import pytest

from mixhound.infrastructure.cache import MemoryCache
from mixhound.models.domain import (
    SonicProfile,
)
from mixhound.models.research import ResearchOutput, ResearchSource
from mixhound.services.agentic import (
    DelegateLimits,
    delegate_limits,
)
from mixhound.services.audio import AudioAnalyzer
from mixhound.services.ports import CatalogRecording
from mixhound.services.research import ResearchError
from mixhound.services.run_state import LlmUsage, RunCancelledError
from tests.fakes import FakeCatalog, FakeDecoder, FakeFetch
from tests.support.agentic import CODEINE, LOW, REVIEW, SWANS, leads, reference, workspace
from tests.support.factories import candidate

LIMITS = DelegateLimits(seconds=300, max_usd=1.0)
PROFILE = SonicProfile(summary="Hushed")


async def test_lookup_is_memoised_and_adds_a_share_of_leads() -> None:
    space, catalog, _ = workspace()

    low = await space.lookup_reference(LOW)
    again = await space.lookup_reference(LOW)
    codeine = await space.lookup_reference(CODEINE)
    swans = await space.lookup_reference(SWANS)

    assert again is low
    assert (low.found, low.facts.mbid, low.facts.tags) == (
        True,
        "r1",
        ["slowcore", "sadcore", "dream pop"],
    )
    # quick depth: a pool of 15 shared by two positive references, at most 8 each.
    assert [lead["artist"] for lead in low.leads] == ["Duster", "Bedhead", "Idaho", "Spain"]
    assert [lead["artist"] for lead in codeine.leads] == ["Seam", "Rex"]
    assert swans.leads == []
    assert len(space.preparation.pool) == space.state.metrics.pool_size == 6
    assert [call[1] for call in catalog.calls] == ["Low", "Codeine", "Swans"]


async def test_one_reference_cannot_fill_the_whole_pool() -> None:
    many = {("Low", "Words"): CatalogRecording("Low", "Words", leads=leads(*"ABCDEFGHIJKL"))}
    space, _, _ = workspace(catalog=FakeCatalog(many))

    report = await space.lookup_reference(LOW)

    assert len(report.leads) == 8


async def test_catalog_outage_is_reported_and_skips_later_lookups() -> None:
    space, catalog, _ = workspace(catalog=FakeCatalog(down=True))

    first = await space.lookup_reference(LOW)
    second = await space.lookup_reference(CODEINE)

    assert first.catalog_down
    assert not second.found
    assert space.state.catalog_down
    assert len(catalog.calls) == 1


def test_references_are_found_by_id_but_not_neutral_ones() -> None:
    neutral = reference("A", "B", polarity="neutral")
    space, _, _ = workspace()
    space.state.run.input.references.append(neutral)

    assert space.reference(str(LOW.id)) == LOW
    assert space.reference(str(neutral.id)) is None
    assert [item.title for item in space.references] == ["Words", "D", "Loud"]


async def test_verify_reports_what_the_server_found_without_touching_the_run() -> None:
    space, catalog, _ = workspace()
    found = candidate(artist="Duster", title="Echo", year=None)
    claimed = candidate(
        artist="Bedhead",
        title="Song",
        sources=[ResearchSource(url=REVIEW, claim="Hushed", target_type="review")],
    )

    confirmed, unconfirmed = await space.verify([found, claimed])

    assert confirmed.confirmed
    assert confirmed.confirmed_by == ("musicbrainz",)
    assert confirmed.year == 1998
    assert confirmed.listening_urls == ["https://open.spotify.com/track/abc"]
    assert confirmed.sources == []
    assert not unconfirmed.confirmed
    assert [(item.url, item.matches) for item in unconfirmed.sources] == [(REVIEW, True)]
    state = space.state
    assert not state.confirmed
    assert not state.recordings
    assert len(state.catalog_answers) == 2
    await space.verify([found])
    assert len(catalog.calls) == 2


async def test_verify_carries_a_catalog_outage_over_to_the_run() -> None:
    space, _, _ = workspace(catalog=FakeCatalog(down=True))

    (report,) = await space.verify([candidate()])

    assert not report.confirmed
    assert space.state.catalog_down


async def test_delegate_runs_a_strategy_with_its_share_and_brief() -> None:
    answer = ResearchOutput(
        sonic_profile=PROFILE, candidates=[candidate(title="One"), candidate(title="Two")]
    )
    space, _, agents = workspace(outcomes={"collaborators": answer})
    await space.lookup_reference(LOW)

    report = await space.delegate("collaborators", PROFILE, "  Follow Alan Sparhawk  ", LIMITS)

    assert (report.failure, report.partial) == (None, False)
    assert [item.title for item in report.candidates] == ["One", "Two"]
    assert space.delegated == {"collaborators": answer}
    (call, prompt) = agents.calls[0]
    assert (call.label, call.seconds, call.max_usd) == ("collaborators", 300, 1.0)
    assert '"strategy": {"name": "collaborators"' in prompt
    assert "\nOrchestrator brief: Follow Alan Sparhawk\nInterim notes:" in prompt
    assert space.state.metrics.strategies["collaborators"].candidates == 2


async def test_delegate_keeps_notes_of_a_cut_off_strategy() -> None:
    note = candidate(title="Noted").model_dump_json()
    error = ResearchError("timeout", notes=[note])
    space, _, _ = workspace(outcomes={"traits": error})

    report = await space.delegate("traits", PROFILE, "", LIMITS)

    assert (report.failure, report.partial) == ("timeout", True)
    assert [item.title for item in report.candidates] == ["Noted"]
    assert "traits" in space.delegated


async def test_failed_delegate_reports_its_code() -> None:
    space, _, _ = workspace(outcomes={"traits": ResearchError("model_unavailable")})

    report = await space.delegate("traits", PROFILE, "", LIMITS)

    assert report.failure == "model_unavailable"
    assert space.delegated == {}


async def test_failed_delegate_of_a_cancelled_run_stops_it() -> None:
    space, _, _ = workspace(outcomes={"traits": ResearchError("timeout")}, cancelled=True)

    with pytest.raises(RunCancelledError):
        await space.delegate("traits", PROFILE, "", LIMITS)


@pytest.mark.parametrize(
    ("strategy", "limits", "failure"),
    [
        ("unknown", LIMITS, "unknown_strategy"),
        ("traits", None, "no_time_or_budget_left"),
    ],
)
async def test_delegate_is_refused(
    strategy: str, limits: DelegateLimits | None, failure: str
) -> None:
    space, _, agents = workspace()

    report = await space.delegate(strategy, PROFILE, "", limits)

    assert report.failure == failure
    assert agents.calls == []


async def test_each_strategy_is_delegated_once() -> None:
    answer = ResearchOutput(sonic_profile=PROFILE, candidates=[])
    space, _, agents = workspace(outcomes={"traits": answer})

    await space.delegate("traits", PROFILE, "", LIMITS)
    again = await space.delegate("traits", PROFILE, "", LIMITS)

    assert again.failure == "already_delegated"
    assert len(agents.calls) == 1


def test_delegate_limits_share_time_and_the_run_budget() -> None:
    space, _, _ = workspace(max_usd=3.0)
    state = space.state

    assert delegate_limits(state, 60, 600, None) == DelegateLimits(390, 1.0)
    state.usage["traits"] = LlmUsage(cost_usd=1.0)
    state.usage["research"] = LlmUsage(cost_usd=5.0)
    assert delegate_limits(state, 0, 600, 1.6) == DelegateLimits(450, 0.4)
    assert delegate_limits(state, 0, 600, 2.0) is None
    assert delegate_limits(state, 400, 600, None) is None


async def test_complete_looks_up_skipped_references_only_for_audio() -> None:
    space, catalog, _ = workspace()
    await space.lookup_reference(LOW)

    await space.complete()
    assert len(catalog.calls) == 1

    audio = AudioAnalyzer(FakeFetch({}), FakeDecoder(), MemoryCache())
    space, catalog, _ = workspace(audio=audio)
    await space.lookup_reference(LOW)
    await space.complete()
    assert [call[1] for call in catalog.calls] == ["Low", "Codeine", "Swans"]
    assert set(space.preparation.facts) == {LOW.id, CODEINE.id, SWANS.id}
