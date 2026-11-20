from pathlib import Path
from uuid import uuid4

import pytest

from mixhound.infrastructure.cache import MemoryCache
from mixhound.infrastructure.skills import load_catalog
from mixhound.memory.context import RunHistory
from mixhound.models.domain import (
    DiscoveryListeningLink,
    DiscoveryResearchResult,
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySession,
    DiscoverySettings,
    DiscoveryTrack,
    SonicProfile,
)
from mixhound.models.research import FoundLink, LinkSearchOutput, ResearchOutput
from mixhound.services.budgets import DEPTH_BUDGETS
from mixhound.services.pages import PageInspector
from mixhound.services.research import (
    FOLLOWUP_MAX_USD,
    HELPER_SECONDS,
    MAX_FOLLOWUP_TRACKS,
    NOTES_REMINDER,
    PROFILE_MAX_USD,
    ResearchError,
    ResearchSteps,
)
from mixhound.services.run_state import RunCancelledError, RunState
from mixhound.services.verification import ResultVerifier
from tests.fakes import FakeAgents, FakeCatalog, FakeFetch, Outcome, Reply
from tests.support.factories import candidate

SKILLS = load_catalog(Path(__file__).parents[3] / ".agents" / "skills")
PROFILE = SonicProfile(summary="Dusty, slow, brushed")


def run_state(*, cancelled: bool = False, **settings: object) -> RunState:
    session_id = uuid4()
    run = DiscoveryRun(
        session_id=session_id,
        input=DiscoveryRunInput(settings=DiscoverySettings.model_validate(settings)),
    )
    return RunState(run, cancelled=lambda: cancelled)


def steps(
    outcomes: dict[str, Outcome], pages: dict[str, Reply] | None = None
) -> tuple[ResearchSteps, FakeAgents]:
    agents = FakeAgents(outcomes)
    verifier = ResultVerifier(FakeCatalog(), PageInspector(FakeFetch(pages or {})), MemoryCache())
    return ResearchSteps(agents, verifier, SKILLS), agents


def output(*titles: str) -> ResearchOutput:
    return ResearchOutput(
        sonic_profile=PROFILE, candidates=[candidate(title=title) for title in titles]
    )


def notes(*titles: str) -> list[str]:
    return [candidate(title=title).model_dump_json() for title in titles]


def empty_history() -> RunHistory:
    return RunHistory(runs=[], tracks=[])


def session_for(state: RunState) -> DiscoverySession:
    return DiscoverySession(id=state.run.session_id, name="Session")


async def test_research_runs_one_noted_call_within_the_run_limits() -> None:
    state = run_state(minutes=7, depth="deep", max_usd=4.0, model="test", reasoning_effort="low")
    research, agents = steps({"research": output("One")})

    answer = await research.research(state, "prompt")

    assert [item.title for item in answer.candidates] == ["One"]
    call, prompt = agents.calls[0]
    assert prompt == "prompt" + NOTES_REMINDER
    assert (call.label, call.seconds, call.max_usd, call.model) == ("research", 420, 4.0, "test")
    assert call.reasoning_effort == "low"
    assert call.notes
    assert call.requests == DEPTH_BUDGETS["deep"].requests
    assert call.candidates == DEPTH_BUDGETS["deep"].candidates
    assert "## Strategy runs" not in call.instructions
    assert not state.run.partial


@pytest.mark.parametrize("code", ["timeout", "budget_exceeded", "limit_reached"])
async def test_cut_off_research_falls_back_to_its_notes(code: str) -> None:
    state = run_state()
    research, _ = steps({"research": ResearchError(code, notes=[*notes("Kept"), "not json"])})

    answer = await research.research(state, "prompt")

    assert [item.title for item in answer.candidates] == ["Kept"]
    assert state.run.partial
    assert state.detail == code


@pytest.mark.parametrize(
    ("error", "cancelled"),
    [
        (ResearchError("timeout"), False),
        (ResearchError("timeout", notes=["not json"]), False),
        (ResearchError("invalid_result", notes=notes("Lost")), False),
        (ResearchError("timeout", notes=notes("Lost")), True),
    ],
)
async def test_research_failure_without_usable_notes_propagates(
    error: ResearchError, cancelled: bool
) -> None:
    state = run_state(cancelled=cancelled)
    research, _ = steps({"research": error})

    with pytest.raises(ResearchError) as caught:
        await research.research(state, "prompt")

    assert caught.value is error
    assert not state.run.partial


async def test_profile_reuses_the_sessions_previous_profile() -> None:
    state = run_state()
    earlier = DiscoveryRun(
        session_id=state.run.session_id, input=DiscoveryRunInput(), sonic_profile=PROFILE
    )
    research, agents = steps({})

    profile = await research.profile(
        state, session_for(state), None, RunHistory(runs=[earlier], tracks=[])
    )

    assert profile == PROFILE
    assert agents.calls == []


async def test_profile_pass_is_a_short_capped_call() -> None:
    state = run_state(model="test")
    research, agents = steps({"profile": PROFILE})

    profile = await research.profile(state, session_for(state), None, empty_history())

    assert profile == PROFILE
    call, _ = agents.calls[0]
    assert (call.label, call.seconds, call.max_usd, call.notes) == (
        "profile",
        HELPER_SECONDS,
        PROFILE_MAX_USD,
        False,
    )
    assert call.instructions == SKILLS.get("sonic-profile").instructions()


async def test_failed_profile_is_skipped_unless_the_run_was_cancelled() -> None:
    research, _ = steps({"profile": ResearchError("timeout")})
    state = run_state()
    assert await research.profile(state, session_for(state), None, empty_history()) is None

    cancelled = run_state(cancelled=True)
    with pytest.raises(RunCancelledError):
        await research.profile(cancelled, session_for(cancelled), None, empty_history())


async def test_strategies_merge_answers_and_partial_notes() -> None:
    state = run_state(depth="quick", max_usd=3.0, parallel_strategies=True)
    research, agents = steps(
        {
            "profile": PROFILE,
            "collaborators": output("Shared", "Own"),
            "label_scene": ResearchError("timeout", notes=notes("Shared", "Noted")),
            "traits": ResearchError("invalid_result"),
        }
    )

    merged = await research.research_strategies(state, session_for(state), None, empty_history())

    assert [item.title for item in merged.candidates] == ["Shared", "Own", "Noted"]
    assert merged.sonic_profile == PROFILE
    assert state.metrics.overlap == 1
    assert state.run.partial
    metrics = state.metrics.strategies
    assert metrics["collaborators"].failed is None
    assert metrics["collaborators"].candidates == 2
    assert metrics["label_scene"].failed == "timeout"
    assert metrics["label_scene"].candidates == 2
    assert metrics["traits"].failed == "invalid_result"
    assert metrics["traits"].cost_usd == 0.25
    assert metrics["traits"].web_searches == 3
    strategy_calls = [call for call, _ in agents.calls if call.label != "profile"]
    assert {call.max_usd for call in strategy_calls} == {1.0}
    assert {call.candidates for call in strategy_calls} == {7}
    assert all("## Strategy runs" in call.instructions for call in strategy_calls)
    assert set(state.metrics.stages) == {"profile", "context", "research"}


async def test_strategies_fail_only_when_every_strategy_failed() -> None:
    state = run_state(parallel_strategies=True)
    first = ResearchError("limit_reached")
    research, _ = steps(
        {
            "profile": ResearchError("timeout"),
            "collaborators": first,
            "label_scene": ResearchError("timeout"),
            "traits": ResearchError("invalid_result"),
        }
    )

    with pytest.raises(ResearchError) as caught:
        await research.research_strategies(state, session_for(state), None, empty_history())

    assert caught.value is first


async def test_crashed_strategy_is_recorded_as_process_failure() -> None:
    state = run_state(parallel_strategies=True)
    research, _ = steps(
        {
            "profile": PROFILE,
            "collaborators": RuntimeError("boom"),
            "label_scene": output("One"),
            "traits": output("Two"),
        }
    )

    merged = await research.research_strategies(state, session_for(state), None, empty_history())

    assert len(merged.candidates) == 2
    assert state.metrics.strategies["collaborators"].failed == "process_failure"


def likely(title: str) -> DiscoveryTrack:
    return DiscoveryTrack(artist="Low", title=title, similarities=["Slow"], similarity_score=60)


async def test_followup_verifies_proposed_pages_of_likely_results() -> None:
    state = run_state(helper_model="cheap", helper_reasoning_effort="low")
    words, unlikely = likely("Words"), DiscoveryTrack(artist="Low", title="Shame")
    state.confirmed.add(words.id)
    page = "https://low.bandcamp.com/track/words"
    research, agents = steps(
        {
            "listening": LinkSearchOutput(
                links=[
                    FoundLink(artist="Low", title="Words", url=page),
                    FoundLink(artist="Low", title="Shame", url=page),
                ]
            )
        },
        {"low.bandcamp.com/track/words": b"<html><body>Low - Words</body></html>"},
    )
    result = DiscoveryResearchResult(results=[words], unverified_leads=[unlikely])

    await research.followup(state, result)

    call, prompt = agents.calls[0]
    assert (call.label, call.model, call.reasoning_effort, call.helper) == (
        "listening",
        "cheap",
        "low",
        True,
    )
    assert call.max_usd == FOLLOWUP_MAX_USD
    assert '"Words"' in prompt
    assert '"Shame"' not in prompt
    assert [link.url for link in words.listening_links] == [page]
    assert words.listening_links[0].reachable is True
    assert page in state.trusted_urls[words.id]
    assert unlikely.listening_links == []


async def test_followup_asks_for_at_most_fifteen_recordings() -> None:
    state = run_state()
    tracks = [likely(f"Song {number}") for number in range(MAX_FOLLOWUP_TRACKS + 3)]
    state.confirmed.update(track.id for track in tracks)
    research, agents = steps({"listening": LinkSearchOutput(links=[])})

    await research.followup(state, DiscoveryResearchResult(results=tracks))

    _, prompt = agents.calls[0]
    assert prompt.count('"artist": "Low"') == MAX_FOLLOWUP_TRACKS


async def test_followup_skips_results_with_enough_links_or_no_chance() -> None:
    state = run_state()
    covered = likely("Covered")
    covered.listening_links = [
        DiscoveryListeningLink(service=service, url=url, target_type="track", reachable=True)
        for service, url in [
            ("Bandcamp", "https://low.bandcamp.com/track/covered"),
            ("YouTube", "https://www.youtube.com/watch?v=abcdefghijk"),
        ]
    ]
    state.confirmed.add(covered.id)
    research, agents = steps({})

    await research.followup(
        state, DiscoveryResearchResult(results=[covered, DiscoveryTrack(artist="A", title="B")])
    )

    assert agents.calls == []


async def test_failed_followup_leaves_results_alone_unless_cancelled() -> None:
    track = likely("Words")
    research, _ = steps({"listening": ResearchError("timeout")})
    state = run_state()
    state.confirmed.add(track.id)
    await research.followup(state, DiscoveryResearchResult(results=[track]))
    assert track.listening_links == []

    cancelled = run_state(cancelled=True)
    cancelled.confirmed.add(track.id)
    with pytest.raises(RunCancelledError):
        await research.followup(cancelled, DiscoveryResearchResult(results=[track]))
