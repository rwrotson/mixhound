import pytest

from mixhound.models.domain import SonicProfile
from mixhound.models.research import ResearchOutput, ResearchSource
from mixhound.services.agentic import (
    AgenticWorkspace,
    CandidateReport,
    DelegateReport,
    ReferenceReport,
    SourceCheck,
)
from mixhound.services.audio_features import AudioFeatures
from mixhound.services.preparation import ReferenceFacts
from mixhound.services.research import ResearchError
from mixhound.tools.deps import Toolbox, ToolDeps
from mixhound.tools.orchestration import (
    NOT_AGENTIC,
    _candidate_text,
    _delegate_text,
    _reference_text,
    delegate_research,
    lookup_reference,
    verify_candidates,
)
from tests.fakes import FakeCatalog, FakeFetch, Outcome
from tests.support.agentic import CODEINE, LOW, REVIEW, SWANS, workspace
from tests.support.agents import call, context, tool_deps
from tests.support.factories import candidate

PROFILE = {"summary": "Hushed", "shared_traits": [], "avoid_traits": []}
FEATURES = AudioFeatures(92, 0.3, -20, 5, 2000, 4000, 0.1, 0.2, 0.5, 0.05, 2.0)


def deps(
    *,
    seconds: float = 600.0,
    candidates: int = 4,
    outcomes: dict[str, Outcome] | None = None,
    catalog: FakeCatalog | None = None,
) -> tuple[ToolDeps, AgenticWorkspace]:
    space, used, _ = workspace(outcomes=outcomes, catalog=catalog)
    toolbox = Toolbox(FakeFetch({}), used)
    agent_call = call(seconds=seconds, candidates=candidates)
    return ToolDeps(space.state, agent_call, toolbox, workspace=space), space


async def test_tools_need_an_agentic_run() -> None:
    ctx = context(tool_deps())

    assert await lookup_reference(ctx, str(LOW.id)) == NOT_AGENTIC
    assert await verify_candidates(ctx, [candidate()]) == NOT_AGENTIC
    profile = SonicProfile()
    assert await delegate_research(ctx, "traits", profile) == NOT_AGENTIC
    assert tool_deps().cap("lookup_reference") == 0


async def test_lookup_reference_frames_catalog_facts_and_leads() -> None:
    tool, _ = deps()
    ctx = context(tool)

    text = await lookup_reference(ctx, f" {LOW.id} ")

    assert text.startswith("<catalog_facts")
    assert f"Reference {LOW.id}: Low — Words" in text
    assert "Catalog: found (MusicBrainz r1)." in text
    assert "Tags: slowcore, sadcore, dream pop" in text
    assert "New candidate pool leads (4):" in text
    assert "- Duster — Song; Last.fm similar track of Low — Words" in text
    assert (
        await lookup_reference(ctx, "nope") == "No positive or negative reference has the ID nope."
    )


async def test_lookup_reference_is_capped_by_the_reference_count() -> None:
    tool, _ = deps()
    ctx = context(tool)
    for reference in (LOW, CODEINE, SWANS):
        await lookup_reference(ctx, str(reference.id))

    assert await lookup_reference(ctx, str(LOW.id)) == (
        "lookup_reference limit reached for this run."
    )


@pytest.mark.parametrize(
    ("report", "line"),
    [
        (
            ReferenceReport(LOW, ReferenceFacts(), found=False, catalog_down=True, leads=[]),
            "Catalog: unavailable right now.",
        ),
        (
            ReferenceReport(LOW, ReferenceFacts(), found=False, catalog_down=False, leads=[]),
            "Catalog: not found.",
        ),
        (
            ReferenceReport(
                LOW,
                ReferenceFacts(audio=FEATURES),
                found=True,
                catalog_down=False,
                leads=[{"artist": "Duster", "title": None, "via": "x", "url": "https://d.example"}],
            ),
            "- Duster — (artist); x; https://d.example",
        ),
    ],
)
def test_reference_text_reports_each_catalog_outcome(report: ReferenceReport, line: str) -> None:
    text = _reference_text(report)

    assert line in text
    assert ("Measured audio:" in text) is (report.facts.audio is not None)
    assert ("Catalog: found." in text) is report.found


async def test_verify_candidates_checks_a_batch_within_the_cap() -> None:
    tool, _ = deps(candidates=2)
    ctx = context(tool)
    claimed = candidate(
        artist="Bedhead",
        title="Song",
        sources=[ResearchSource(url=REVIEW, claim="Hushed", target_type="review")],
    )
    found = candidate(artist="Duster", title="Echo")

    text = await verify_candidates(ctx, [found, claimed, candidate()])

    assert text.splitlines() == [
        "1. Duster — Echo: confirmed by musicbrainz, year 2015; "
        "track pages: https://open.spotify.com/track/abc.",
        f"2. Bedhead — Song: not confirmed by the catalogs; source pages: {REVIEW} names it.",
        "1 candidate(s) not checked: verify_candidates limit reached.",
    ]
    assert await verify_candidates(ctx, []) == "No candidates given."
    assert await verify_candidates(ctx, [found]) == (
        "verify_candidates limit reached for this run; answer with what you have."
    )


def test_candidate_text_reports_audio_and_unmatched_pages() -> None:
    report = CandidateReport(
        "A",
        "B",
        confirmed=True,
        confirmed_by=("itunes",),
        sources=[SourceCheck("https://x.example", matches=False)],
        audio_similarity=0.734,
        audio_evidence="Measured",
    )

    assert _candidate_text(3, report) == (
        "3. A — B: confirmed by itunes; source pages: https://x.example does not name it; "
        "audio similarity 0.73 (Measured)."
    )


async def test_delegate_research_summarises_the_strategy_answer() -> None:
    answer = ResearchOutput(
        sonic_profile=SonicProfile(),
        candidates=[
            candidate(title="Low score", similarity_score=40),
            candidate(title="High score", similarity_score=80),
        ],
    )
    tool, space = deps(outcomes={"traits": answer})
    ctx = context(tool)

    text = await delegate_research(ctx, "traits", SonicProfile.model_validate(PROFILE), "brief")

    assert text.splitlines() == [
        "Strategy traits kept 2 candidates; the server merges them with your answer, so repeat "
        "one only to improve it. Best:",
        "- Fogh Depot — High score (80, medium)",
        "- Fogh Depot — Low score (40, medium)",
    ]
    assert space.delegated == {"traits": answer}


async def test_delegate_research_is_refused_without_time_and_capped() -> None:
    tool, _ = deps(seconds=60)
    ctx = context(tool)

    for _ in range(3):
        refused = await delegate_research(ctx, "traits", SonicProfile())
        assert refused == "Strategy traits returned nothing: no_time_or_budget_left."
    assert await delegate_research(ctx, "traits", SonicProfile()) == (
        "delegate_research limit reached for this run."
    )


@pytest.mark.parametrize(
    ("report", "text"),
    [
        (DelegateReport("traits"), "Strategy traits returned nothing: no candidates."),
        (
            DelegateReport("traits", [candidate()], failure="timeout", partial=True),
            "Strategy traits kept 1 candidates (cut off: timeout);",
        ),
    ],
)
def test_delegate_text_reports_empty_and_partial_answers(report: DelegateReport, text: str) -> None:
    assert _delegate_text(report).startswith(text)


async def test_delegate_research_reports_a_failed_strategy() -> None:
    tool, _ = deps(outcomes={"label_scene": ResearchError("model_unavailable")})

    text = await delegate_research(context(tool), "label_scene", SonicProfile())

    assert text == "Strategy label_scene returned nothing: model_unavailable."
