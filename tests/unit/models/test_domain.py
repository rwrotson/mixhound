from datetime import datetime

import pytest

from mixhound.models.domain import (
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySession,
    DiscoverySettings,
    DiscoveryTrack,
    FeedbackEvent,
    legacy_mark,
)


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (1, ("negative", 5)),
        (2, ("negative", 3)),
        (3, ("neutral", 3)),
        (4, ("positive", 3)),
        (5, ("positive", 5)),
    ],
)
def test_legacy_rating_replays_as_a_mark(rating: int, expected: tuple[str, int]) -> None:
    assert legacy_mark(rating) == expected


def test_mark_requires_polarity_and_weight_unless_off() -> None:
    run_id = DiscoveryRun(session_id=DiscoverySession(name="x").id, input=DiscoveryRunInput()).id
    track_id = DiscoveryTrack(artist="A", title="B").id
    with pytest.raises(ValueError, match="polarity"):
        FeedbackEvent(run_id=run_id, track_id=track_id, action="mark", weight=3)
    with pytest.raises(ValueError, match="weight"):
        FeedbackEvent(run_id=run_id, track_id=track_id, action="mark", polarity="positive")
    off = FeedbackEvent(run_id=run_id, track_id=track_id, action="mark", polarity="neutral")
    assert off.weight is None
    with pytest.raises(ValueError, match="rating"):
        FeedbackEvent(run_id=run_id, track_id=track_id, action="rating")


def test_feedback_state_replays_events_in_order() -> None:
    session = DiscoverySession(name="x")
    run = DiscoveryRun(session_id=session.id, input=DiscoveryRunInput())
    track = DiscoveryTrack(artist="A", title="B")

    def event(**values: object) -> FeedbackEvent:
        return FeedbackEvent.model_validate({"run_id": run.id, "track_id": track.id, **values})

    session.feedback_events = [
        event(action="skip"),
        event(action="rating", rating=5),
        event(action="comment", text="Too loud"),
        event(action="tags", tags=["drone"]),
        event(action="exclude"),
        event(action="mark", polarity="negative", weight=2),
    ]

    state = session.feedback_for(track.id)

    assert (state.polarity, state.weight, state.skipped) == ("negative", 2, False)
    assert (state.text, state.tags, state.excluded) == ("Too loud", ["drone"], True)


def test_settings_reject_reversed_years_and_blank_models() -> None:
    with pytest.raises(ValueError, match="Starting year"):
        DiscoverySettings(year_from=2000, year_to=1990)

    settings = DiscoverySettings(model="  ", helper_model="openrouter:anthropic/claude-haiku-5-5")

    assert settings.model is None
    assert settings.helper_model == "openrouter:anthropic/claude-haiku-5-5"
    assert settings.mode == "pipeline"


def test_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone"):
        DiscoveryTrack(artist="A", title="B", created_at=datetime(2026, 1, 1))  # noqa: DTZ001
    run = DiscoveryRun.model_validate(
        {
            "session_id": DiscoverySession(name="x").id,
            "input": DiscoveryRunInput(),
            "finished_at": "2026-01-01T03:00:00+03:00",
        }
    )
    assert run.finished_at is not None
    assert run.finished_at.hour == 0
