import asyncio
from uuid import uuid4

from mixhound.models.domain import DiscoveryRun, DiscoveryRunInput
from mixhound.models.state import RunStatusEvent
from mixhound.services.run_events import RunEvents


def run(**values: object) -> DiscoveryRun:
    return DiscoveryRun.model_validate(
        {"session_id": uuid4(), "input": DiscoveryRunInput(), **values}
    )


def test_history_keeps_the_most_recently_active_runs() -> None:
    events = RunEvents(max_histories=2)
    first, second, third = run(), run(), run()
    events.emit(first, 1)
    events.emit(second)
    events.emit(first)
    events.emit(third)

    assert events.history(second.id) == []
    assert events.history(first.id) == [
        RunStatusEvent(status="queued", position=1),
        RunStatusEvent(status="queued"),
    ]
    assert len(events.history(third.id)) == 1


def test_failure_code_is_reported() -> None:
    events = RunEvents()
    failed = run(status="failed", failure_code="timeout")

    events.emit(failed)

    assert events.history(failed.id)[0].terminal
    assert events.history(failed.id)[0].failure_code == "timeout"


async def test_wait_wakes_on_the_next_event() -> None:
    events = RunEvents()
    current = run()

    waiter = asyncio.create_task(events.wait(current.id, 5))
    await asyncio.sleep(0)
    events.emit(current)

    assert await waiter
    assert not await events.wait(current.id, 0.01)
