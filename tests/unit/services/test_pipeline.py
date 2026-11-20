import asyncio
from collections.abc import Awaitable, Callable
from uuid import uuid4

import pytest

from mixhound.config import PACKAGED_SKILLS_DIR
from mixhound.infrastructure.cache import MemoryCache
from mixhound.infrastructure.skills import load_catalog
from mixhound.memory.context import RunHistory
from mixhound.models.domain import (
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySession,
    DiscoverySettings,
    SonicProfile,
)
from mixhound.models.research import FoundLink, LinkSearchOutput, ResearchOutput, ResearchSource
from mixhound.services.agentic import ORCHESTRATION_NOTE, AgenticWorkspace, DelegateLimits
from mixhound.services.audio import AudioMatcher
from mixhound.services.evidence import ResearchEvidenceError
from mixhound.services.pages import PageInspector
from mixhound.services.pipeline import ResearchPipeline
from mixhound.services.ports import CatalogPage, CatalogRecording
from mixhound.services.research import AgentCall, ResearchError, ResearchSteps
from mixhound.services.run_state import STAGE_ORDER, RunState
from mixhound.services.seed import Seeder
from mixhound.services.verification import ResultVerifier
from tests.fakes import FakeAgents, FakeCatalog, FakeFetch, Outcome
from tests.support.factories import candidate

SKILLS = load_catalog(PACKAGED_SKILLS_DIR)
REVIEW = "https://reviews.example/low-words"
TRACK_PAGE = "https://low.bandcamp.com/track/words"
WORDS = CatalogRecording(
    artist="Low",
    title="Words",
    confirmed_by=("musicbrainz",),
    mbid="mbid-words",
    pages=(CatalogPage("musicbrainz", "https://musicbrainz.org/recording/mbid-words"),),
)
PAGES: dict[str, bytes | Exception | dict[str, object] | list[object]] = {
    "reviews.example/low-words": b"<html><body>Low - Words: slow, brushed, hushed.</body></html>",
    "low.bandcamp.com/track/words": b"<html><body>Low - Words</body></html>",
}


def words(**changes: object) -> ResearchOutput:
    changes.setdefault("similarity_score", 70)
    found = candidate(
        artist="Low",
        title="Words",
        sources=[
            ResearchSource(
                url=REVIEW, claim="Slow, brushed drums and hushed vocals", target_type="review"
            )
        ],
        **changes,
    )
    return ResearchOutput(sonic_profile=SonicProfile(summary="Hushed"), candidates=[found])


def pipeline(
    outcomes: dict[str, Outcome],
    orchestration: Callable[[AgenticWorkspace], Awaitable[None]] | None = None,
) -> tuple[ResearchPipeline, FakeAgents]:
    agents = FakeAgents(outcomes, orchestration)
    catalog = FakeCatalog({("Low", "Words"): WORDS})
    verifier = ResultVerifier(catalog, PageInspector(FakeFetch(dict(PAGES))), MemoryCache())
    return (
        ResearchPipeline(
            Seeder(catalog, None),
            verifier,
            AudioMatcher(None),
            ResearchSteps(agents, verifier, SKILLS),
        ),
        agents,
    )


def run_state(*, cancelled: bool = False, **settings: object) -> tuple[RunState, DiscoverySession]:
    session = DiscoverySession(name="Session")
    run = DiscoveryRun(
        session_id=session.id,
        input=DiscoveryRunInput(settings=DiscoverySettings.model_validate(settings)),
    )
    return RunState(run, cancelled=lambda: cancelled), session


HISTORY = RunHistory(runs=[], tracks=[])


async def test_confirmed_result_with_sonic_evidence_is_published() -> None:
    state, session = run_state(model="test")
    flow, agents = pipeline(
        {
            "research": words(),
            "listening": LinkSearchOutput(
                links=[FoundLink(artist="Low", title="Words", url=TRACK_PAGE)]
            ),
        }
    )

    diagnostic = await flow.execute(state, session, HISTORY)

    run = state.run
    assert run.status == "completed"
    assert [track.title for track in run.results] == ["Words"]
    published = run.results[0]
    assert published.verification == "verified"
    assert published.external_recording_id == "mbid:mbid-words"
    assert TRACK_PAGE in [link.url for link in published.listening_links]
    assert run.sonic_profile == SonicProfile(summary="Hushed")
    assert run.metrics is state.metrics
    assert run.finished_at is not None
    assert state.metrics.candidates == 1
    assert state.metrics.sonic_evidence == 1
    assert state.metrics.verified == 1
    assert state.metrics.llm_calls == 2
    assert set(state.metrics.stages) == set(STAGE_ORDER) - {"profile"}
    assert diagnostic["stage"] == "completed"
    assert diagnostic["mode"] == "pipeline"
    assert diagnostic["model"] == "test"
    assert diagnostic["answer"] == words().model_dump_json()
    assert "Words" in str(diagnostic["followup_answer"])
    assert "failure_code" not in diagnostic
    assert "strategy_answers" not in diagnostic
    assert "notes" not in diagnostic
    assert [call.label for call, _ in agents.calls] == ["research", "listening"]


async def test_low_score_becomes_a_lead() -> None:
    state, session = run_state()
    flow, _ = pipeline(
        {"research": words(similarity_score=20), "listening": LinkSearchOutput(links=[])}
    )

    diagnostic = await flow.execute(state, session, HISTORY)

    assert state.run.status == "completed"
    assert state.run.results == []
    assert [track.title for track in state.run.unverified_leads] == ["Words"]
    assert state.metrics.leads == 1
    # A lead below the publication threshold is not worth a link search.
    assert diagnostic["followup_answer"] == ""


async def test_parallel_strategies_report_each_answer() -> None:
    state, session = run_state(parallel_strategies=True)
    flow, _ = pipeline(
        {
            "profile": SonicProfile(summary="Hushed"),
            "collaborators": words(),
            "label_scene": words(),
            "traits": ResearchError("timeout"),
            "listening": LinkSearchOutput(links=[]),
        }
    )

    diagnostic = await flow.execute(state, session, HISTORY)

    assert state.run.status == "completed"
    assert [track.title for track in state.run.results] == ["Words"]
    assert state.metrics.overlap == 1
    answers = diagnostic["strategy_answers"]
    assert isinstance(answers, dict)
    assert set(answers) == {"collaborators", "label_scene"}


@pytest.mark.parametrize(
    ("error", "code", "detail"),
    [
        (ResearchError("timeout", "ran out"), "timeout", "ran out"),
        (ResearchEvidenceError("bad shape"), "invalid_result", "bad shape"),
        (RuntimeError("boom"), "process_failure", "RuntimeError: boom"),
    ],
)
async def test_failure_is_recorded_with_its_code(
    error: BaseException, code: str, detail: str
) -> None:
    state, session = run_state()
    flow, _ = pipeline({"research": error})

    diagnostic = await flow.execute(state, session, HISTORY)

    assert state.run.status == "failed"
    assert state.run.failure_code == code
    assert state.run.failure_message == detail
    assert diagnostic["stage"] == "research"
    assert diagnostic["failure_code"] == code
    assert diagnostic["detail"] == detail


async def test_failure_without_detail_reports_its_code() -> None:
    state, session = run_state()
    flow, _ = pipeline({"research": ResearchError("limit_reached")})

    await flow.execute(state, session, HISTORY)

    assert state.run.failure_message == "limit_reached"


class CancellingAgents(FakeAgents):
    """Cancels the run while the research is running, which then fails."""

    def __init__(self, cancel: list[bool], error: BaseException) -> None:
        super().__init__({"research": error})
        self.cancel = cancel

    async def research(self, state: RunState, call: AgentCall, prompt: str) -> ResearchOutput:
        self.cancel.append(True)
        return await super().research(state, call, prompt)


@pytest.mark.parametrize("error", [ResearchError("timeout"), asyncio.CancelledError()])
async def test_run_cancelled_during_research_is_marked_cancelled(error: BaseException) -> None:
    cancel: list[bool] = []
    session = DiscoverySession(name="Session")
    state = RunState(
        DiscoveryRun(session_id=session.id, input=DiscoveryRunInput()),
        cancelled=lambda: bool(cancel),
    )
    flow, _ = pipeline({})
    flow.steps.agents = CancellingAgents(cancel, error)

    diagnostic = await flow.execute(state, session, HISTORY)

    assert state.run.status == "cancelled"
    assert state.run.failure_code is None
    assert diagnostic["stage"] == "cancelled"


async def test_run_cancelled_before_research_stops_at_the_first_stage() -> None:
    state, session = run_state(cancelled=True)
    flow, agents = pipeline({})

    await flow.execute(state, session, HISTORY)

    assert state.run.status == "cancelled"
    assert agents.calls == []


async def test_history_excludes_earlier_recordings() -> None:
    state, session = run_state()
    earlier = DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput())
    earlier_output = words()
    flow, _ = pipeline({"research": earlier_output, "listening": LinkSearchOutput(links=[])})
    first_state, _ = run_state()
    await flow.execute(first_state, session, HISTORY)
    earlier.results = first_state.run.results

    await flow.execute(state, session, RunHistory(runs=[earlier], tracks=earlier.results))

    assert state.run.results == []
    assert any("Words" in item.title for item in state.rejections)


async def test_agentic_run_merges_delegated_answers_and_verifies_once() -> None:
    state, session = run_state(mode="agentic", model="test")
    other = words().model_copy(update={"candidates": [candidate(artist="Duster", title="Echo")]})

    async def orchestrate(workspace: AgenticWorkspace) -> None:
        (report,) = await workspace.verify(words().candidates)
        assert report.confirmed
        await workspace.delegate("traits", SonicProfile(), "", DelegateLimits(60, 1.0))

    flow, agents = pipeline(
        {"research": words(), "traits": other, "listening": LinkSearchOutput(links=[])},
        orchestrate,
    )

    diagnostic = await flow.execute(state, session, HISTORY)

    run = state.run
    assert run.status == "completed"
    assert [track.title for track in run.results] == ["Words"]
    assert [track.title for track in run.unverified_leads] == ["Echo"]
    assert state.metrics.candidates == 2
    assert state.metrics.overlap == 0
    assert [call.label for call, _ in agents.calls] == ["traits", "research", "listening"]
    orchestrator, prompt = agents.calls[1]
    assert "# Discovery orchestration" in orchestrator.instructions
    assert "# Music similarity research" in orchestrator.instructions
    assert "## Strategy runs" not in orchestrator.instructions
    assert ORCHESTRATION_NOTE in prompt
    assert '"candidate_pool": []' in prompt
    assert set(state.metrics.stages) == set(STAGE_ORDER) - {"profile"}
    catalog = flow.verifier.catalog
    assert isinstance(catalog, FakeCatalog)
    assert sum(call[2] == "Words" for call in catalog.calls) == 1
    assert diagnostic["mode"] == "agentic"
    assert diagnostic["answer"] == words().model_dump_json()
    answers = diagnostic["strategy_answers"]
    assert isinstance(answers, dict)
    assert set(answers) == {"traits"}


async def test_cut_off_orchestrator_keeps_its_delegated_answers() -> None:
    state, session = run_state(mode="agentic")

    async def orchestrate(workspace: AgenticWorkspace) -> None:
        await workspace.delegate("collaborators", SonicProfile(), "", DelegateLimits(60, 1.0))

    flow, _ = pipeline(
        {
            "research": ResearchError("timeout"),
            "collaborators": words(),
            "listening": LinkSearchOutput(links=[]),
        },
        orchestrate,
    )

    await flow.execute(state, session, HISTORY)

    assert state.run.status == "completed"
    assert state.run.partial
    assert [track.title for track in state.run.results] == ["Words"]


async def test_orchestrator_failing_on_its_answer_keeps_its_delegated_answers() -> None:
    state, session = run_state(mode="agentic")

    async def orchestrate(workspace: AgenticWorkspace) -> None:
        await workspace.delegate("collaborators", SonicProfile(), "", DelegateLimits(60, 1.0))

    flow, _ = pipeline(
        {
            "research": ResearchError("output_limit"),
            "collaborators": words(),
            "listening": LinkSearchOutput(links=[]),
        },
        orchestrate,
    )

    await flow.execute(state, session, HISTORY)

    assert state.run.status == "completed"
    assert state.run.partial
    assert [track.title for track in state.run.results] == ["Words"]


async def test_failed_run_keeps_the_notes_in_its_diagnostic() -> None:
    state, session = run_state()
    state.notes["research"] = ['{"artist": "A", "title": "B"}']
    flow, _ = pipeline({"research": ResearchError("auth_unavailable")})

    diagnostic = await flow.execute(state, session, HISTORY)

    assert state.run.status == "failed"
    assert diagnostic["notes"] == {"research": ['{"artist": "A", "title": "B"}']}


@pytest.mark.parametrize("code", ["timeout", "model_unavailable"])
async def test_orchestrator_failure_without_kept_answers_fails_the_run(code: str) -> None:
    state, session = run_state(mode="agentic")
    flow, _ = pipeline({"research": ResearchError(code)})

    await flow.execute(state, session, HISTORY)

    assert state.run.status == "failed"
    assert state.run.failure_code == code
