import asyncio
from uuid import uuid4

import pytest

from mixhound.infrastructure.queue import AsyncioRunQueue, QueueRejectedError


async def test_jobs_run_one_at_a_time_in_order() -> None:
    queue = AsyncioRunQueue()
    order: list[str] = []
    release = asyncio.Event()

    async def first() -> None:
        order.append("first started")
        await release.wait()
        order.append("first done")

    async def second() -> None:
        order.append("second")

    one, two = uuid4(), uuid4()
    assert await queue.submit(one, first) == 1
    await asyncio.sleep(0)
    assert queue.tracked(one)
    assert await queue.submit(two, second) == 1
    release.set()
    assert queue.worker is not None
    await queue.worker

    assert order == ["first started", "first done", "second"]
    assert not queue.tracked(one)


async def test_duplicate_full_and_closed_submissions_are_rejected() -> None:
    queue = AsyncioRunQueue(max_pending=1)
    blocker = asyncio.Event()

    async def wait() -> None:
        await blocker.wait()

    run = uuid4()
    await queue.submit(run, wait)
    await asyncio.sleep(0)
    with pytest.raises(QueueRejectedError, match="already"):
        await queue.submit(run, wait)
    await queue.submit(uuid4(), wait)
    with pytest.raises(QueueRejectedError, match="full"):
        await queue.submit(uuid4(), wait)
    await queue.close()
    with pytest.raises(QueueRejectedError, match="shutting down"):
        await queue.submit(uuid4(), wait)


async def test_cancel_drops_queued_jobs_and_stops_the_running_one() -> None:
    queue = AsyncioRunQueue()
    started = asyncio.Event()
    finished: list[str] = []

    async def slow() -> None:
        started.set()
        await asyncio.Event().wait()

    async def later() -> None:
        finished.append("later")

    running, waiting = uuid4(), uuid4()
    await queue.submit(running, slow)
    await queue.submit(waiting, later)
    await started.wait()

    assert await queue.cancel(waiting)
    assert await queue.cancel(running)
    assert not await queue.cancel(uuid4())
    assert queue.worker is not None
    await queue.worker
    assert finished == []


async def test_a_failing_job_does_not_stop_the_queue() -> None:
    queue = AsyncioRunQueue()
    done: list[str] = []

    async def fail() -> None:
        raise RuntimeError("boom")

    async def succeed() -> None:
        done.append("ok")

    await queue.submit(uuid4(), fail)
    await queue.submit(uuid4(), succeed)
    assert queue.worker is not None
    await queue.worker

    assert done == ["ok"]


async def test_close_cancels_the_running_job() -> None:
    queue = AsyncioRunQueue()
    started = asyncio.Event()

    async def slow() -> None:
        started.set()
        await asyncio.Event().wait()

    await queue.submit(uuid4(), slow)
    await started.wait()
    await queue.close()

    assert queue.worker is not None
    assert queue.worker.done()
