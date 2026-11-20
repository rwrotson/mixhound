from uuid import uuid4

from mixhound.memory.context import feedback_snapshot, run_history
from mixhound.models.domain import (
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySession,
    DiscoveryTrack,
    FeedbackEvent,
)
from tests.fakes import FakeRuns, FakeSessions


async def test_history_holds_every_other_run_and_its_tracks() -> None:
    session = DiscoverySession(name="Test")
    result, lead = DiscoveryTrack(artist="A", title="B"), DiscoveryTrack(artist="C", title="D")
    earlier = DiscoveryRun(
        session_id=session.id, input=DiscoveryRunInput(), results=[result], unverified_leads=[lead]
    )
    current = DiscoveryRun(session_id=session.id, input=DiscoveryRunInput())
    session.run_ids = [earlier.id, uuid4(), current.id]

    history = await run_history(FakeSessions(session), FakeRuns(earlier, current), current)

    assert history.runs == [earlier]
    assert history.tracks == [result, lead]


async def test_latest_mark_wins_and_snapshot_carries_it() -> None:
    features = {"tempo": 120.0}
    liked = DiscoveryTrack(artist="Liked", title="One", audio_features=features)
    off = DiscoveryTrack(artist="Off", title="Two", audio_features=features)
    noted = DiscoveryTrack(artist="Noted", title="Three")
    untouched = DiscoveryTrack(artist="Quiet", title="Four")
    session = DiscoverySession(name="Test")
    run = DiscoveryRun(
        session_id=session.id,
        input=DiscoveryRunInput(),
        status="completed",
        results=[liked, off, untouched],
        unverified_leads=[noted],
    )
    session.run_ids = [uuid4(), run.id]

    def event(track: DiscoveryTrack, **values: object) -> FeedbackEvent:
        return FeedbackEvent.model_validate({"run_id": run.id, "track_id": track.id, **values})

    session.feedback_events = [
        event(liked, action="rating", rating=1),
        event(liked, action="mark", weight=4, polarity="positive"),
        event(off, action="mark", weight=5, polarity="negative"),
        event(off, action="mark", polarity="neutral"),
        event(noted, action="comment", text="too slow"),
    ]

    snapshot = await feedback_snapshot(FakeRuns(run), session)

    assert [(item["artist"], item["polarity"], item["weight"]) for item in snapshot] == [
        ("Liked", "positive", 4),
        ("Noted", "neutral", None),
    ]
    assert snapshot[0]["audio"] == features
    assert snapshot[1] == {
        "artist": "Noted",
        "title": "Three",
        "polarity": "neutral",
        "weight": None,
        "reason": "too slow",
        "excluded": False,
        "avoid_similar": False,
        "tags": [],
        "audio": None,
    }
