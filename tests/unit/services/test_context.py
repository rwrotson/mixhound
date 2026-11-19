import json
from uuid import uuid4

from mixhound.memory.context import RunHistory
from mixhound.models.domain import (
    DiscoveryReference,
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySession,
    DiscoverySettings,
    DiscoveryTrack,
    SonicProfile,
)
from mixhound.services.audio_features import AudioFeatures
from mixhound.services.budgets import DEPTH_BUDGETS
from mixhound.services.context import (
    MAX_FEEDBACK_ITEMS,
    StrategyShare,
    build_context,
    excluded_artists,
    exclusions_from_history,
    previous_profile,
    profile_context,
    research_target,
)
from mixhound.services.exclusions import metadata_key
from mixhound.services.preparation import Preparation, ReferenceFacts

NO_HISTORY = RunHistory(runs=[], tracks=[])


def payload(prompt: str) -> dict[str, object]:
    data = json.loads(prompt.split("\n", 1)[1])
    assert isinstance(data, dict)
    return data


def finished(session_id: object, summary: str, *tracks: DiscoveryTrack) -> DiscoveryRun:
    return DiscoveryRun.model_validate(
        {
            "session_id": session_id,
            "input": DiscoveryRunInput(),
            "status": "completed",
            "sonic_profile": SonicProfile(summary=summary),
            "results": list(tracks),
        }
    )


def test_prompt_carries_pool_reference_facts_and_previous_profile() -> None:
    reference = DiscoveryReference(kind="manual", artist="Low", title="Words")
    session_id = uuid4()
    history = RunHistory(
        runs=[finished(session_id, "Slow and hushed"), finished(uuid4(), "Loud")],
        tracks=[],
    )
    audio = AudioFeatures(70, 0.5, -20, 6, 900, 2000, 0.05, 0.5, 0.4, 0.02, 1.0)
    preparation = Preparation(
        facts={reference.id: ReferenceFacts(tags=["slowcore"], audio=audio)},
        pool=[{"artist": "Codeine", "title": None, "via": "ListenBrainz similar artist"}],
    )
    run = DiscoveryRun(session_id=session_id, input=DiscoveryRunInput(references=[reference]))

    prompt, _ = build_context(run, history, preparation)

    data = payload(prompt)
    assert data["previous_profile"] == {
        "summary": "Slow and hushed",
        "shared_traits": [],
        "avoid_traits": [],
    }
    assert data["candidate_pool"] == preparation.pool
    references = data["references"]
    assert isinstance(references, list)
    [sent] = references
    assert sent["catalog_tags"] == ["slowcore"]
    assert sent["audio"]["brightness"] == "dark"


def test_prompt_asks_for_researched_candidates_beyond_desired_count() -> None:
    settings = DiscoverySettings(desired_count=5)
    run = DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput(settings=settings))

    prompt, _ = build_context(run, NO_HISTORY)

    assert "return up to 15 scored candidates" in prompt
    assert "researched: false" in prompt
    assert payload(prompt)["budget"] == {"target": 15, "candidates": 60, "minutes": 10}


def test_strategy_prompt_carries_profile_and_strategy() -> None:
    run = DiscoveryRun(
        session_id=uuid4(),
        input=DiscoveryRunInput(settings=DiscoverySettings(desired_count=5)),
    )
    share = StrategyShare(
        name="traits",
        description="Trait searches",
        profile=SonicProfile(summary="Known"),
        pool=[{"artist": "A"}],
        budget=DEPTH_BUDGETS["quick"],
        shares=3,
    )

    data = payload(build_context(run, NO_HISTORY, share=share)[0])

    assert "previous_profile" not in data
    assert data["profile"]["summary"] == "Known"  # type: ignore[index]
    assert data["strategy"] == {"name": "traits", "description": "Trait searches"}
    assert data["candidate_pool"] == [{"artist": "A"}]
    assert data["budget"]["target"] == 5  # type: ignore[index]


def test_history_becomes_exclusions_and_previous_keys() -> None:
    old = DiscoveryTrack(artist="Duster", title="Inside Out")
    run = DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput())
    history = RunHistory(runs=[], tracks=[old])

    prompt, exclusions = build_context(run, history)

    assert payload(prompt)["previous_recording_keys"] == [metadata_key(old)]
    assert exclusions == exclusions_from_history(history)
    assert exclusions.prior_artists == frozenset({"duster"})


def test_off_references_are_left_out_of_prompts() -> None:
    on = DiscoveryReference(kind="manual", artist="On", title="Kept")
    off = DiscoveryReference(kind="manual", artist="Off", title="Ignored", polarity="neutral")
    run = DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput(references=[on, off]))

    prompt, _ = build_context(run, NO_HISTORY)

    assert [item["artist"] for item in payload(prompt)["references"]] == ["On"]  # type: ignore[attr-defined]
    assert "Ignored" not in profile_context(run)


def test_excluded_artists_are_settled_up_front() -> None:
    settings = DiscoverySettings(
        excluded_artists=["Low", "low"],
        exclude_reference_artists=True,
        exclude_previously_proposed_artists=True,
    )
    reference = DiscoveryReference(kind="manual", artist="Codeine", title="D")
    run = DiscoveryRun(
        session_id=uuid4(),
        input=DiscoveryRunInput(settings=settings, references=[reference]),
    )

    names = excluded_artists(run, [DiscoveryTrack(artist="Duster", title="X")])

    assert names == ["Low", "Codeine", "Duster"]


def test_feedback_is_bounded_and_ranked() -> None:
    feedback: list[dict[str, object]] = [
        {"artist": f"A{index}", "title": "T", "polarity": "neutral"} for index in range(50)
    ]
    feedback[0] |= {"polarity": "negative", "weight": 5, "reason": "x" * 400, "audio": {}}
    run = DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput(feedback=feedback))

    sent = payload(build_context(run, NO_HISTORY)[0])["feedback"]

    assert isinstance(sent, list)
    assert len(sent) == MAX_FEEDBACK_ITEMS
    assert sent[0]["artist"] == "A0"
    assert "audio" not in sent[0]
    assert len(sent[0]["reason"]) == 250


def test_research_target_is_bounded() -> None:
    assert research_target(10, 60) == 30
    assert research_target(10, 20) == 20
    assert research_target(5, 60, shares=3) == 5


def test_previous_profile_uses_the_latest_run_of_the_session() -> None:
    session = DiscoverySession(name="x")
    first, second = finished(session.id, "Old"), finished(session.id, "New")

    found = previous_profile(session, RunHistory(runs=[second, first], tracks=[]))

    assert found is not None
    assert found.summary == "New"
