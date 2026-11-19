from uuid import uuid4

from mixhound.memory.context import run_history
from mixhound.models.domain import DiscoveryRun, DiscoveryRunInput, DiscoverySession, DiscoveryTrack
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
