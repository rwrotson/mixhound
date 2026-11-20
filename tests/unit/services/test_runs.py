import asyncio
from collections.abc import Callable
from uuid import UUID, uuid4

import pytest

from mixhound.infrastructure.queue import AsyncioRunQueue, QueueRejectedError
from mixhound.infrastructure.storage.memory import MemoryRuns
from mixhound.models.domain import (
    DiscoveryReference,
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySettings,
    DiscoveryTrack,
    FeedbackEvent,
)
from mixhound.models.state import RunProgress, RunStatusEvent
from mixhound.services.runs import (
    InvalidResearchInputError,
    RunNotActiveError,
    RunRejectedError,
)
from mixhound.services.sessions import RunNotFoundError, SessionNotFoundError
from tests.support.runs import FakeExecutor, Harness, positive


class FlakyRuns(MemoryRuns):
    def __init__(self) -> None:
        super().__init__()
        self.fail_save = False
        self.fail_get = False
        self.on_get: Callable[[UUID], None] | None = None

    async def get(self, run_id: UUID) -> DiscoveryRun | None:
        if self.fail_get:
            raise OSError("disk gone")
        if self.on_get is not None:
            self.on_get(run_id)
        return await super().get(run_id)

    async def save(self, run: DiscoveryRun) -> None:
        if self.fail_save:
            raise OSError("disk full")
        await super().save(run)


async def statuses(harness: Harness, run_id: UUID) -> list[str]:
    return [event.status for event in harness.events.history(run_id)]


async def stored(harness: Harness, run_id: UUID) -> DiscoveryRun:
    run = await harness.runs.get(run_id)
    assert run is not None
    return run


async def test_run_is_queued_executed_and_stored() -> None:
    harness = Harness()
    session = await harness.session()

    run = await harness.service.create(session.id, " Find more ", [positive()], DiscoverySettings())
    await harness.idle()

    assert run.status == "queued"
    assert run.input.instructions == "Find more"
    assert not run.input.refinement
    assert (await stored(harness, run.id)).status == "completed"
    assert (await harness.sessions.get(session.id)).run_ids == [run.id]  # type: ignore[union-attr]
    assert harness.events.history(run.id)[0] == RunStatusEvent(status="queued", position=1)
    assert await statuses(harness, run.id) == ["queued", "running", "completed"]
    assert harness.diagnostics.payloads[run.id] == {"run_id": str(run.id), "session": "Test"}
    assert harness.executor.calls == [(run.id, 0)]
    assert run.id not in harness.service.active


async def test_refinement_snapshots_feedback_and_drops_neutral_references() -> None:
    harness = Harness()
    liked = DiscoveryTrack(artist="Liked", title="One")
    session = await harness.session()
    earlier = DiscoveryRun(
        session_id=session.id, input=DiscoveryRunInput(), status="completed", results=[liked]
    )
    await harness.runs.save(earlier)
    session.run_ids = [earlier.id]
    session.feedback_events = [
        FeedbackEvent(
            run_id=earlier.id, track_id=liked.id, action="mark", weight=4, polarity="positive"
        )
    ]
    await harness.sessions.save(session)
    off = DiscoveryReference(kind="manual", artist="Off", title="Two", polarity="neutral")

    run = await harness.service.create(session.id, "More", [positive(), off], DiscoverySettings())
    await harness.idle()

    assert run.input.refinement
    assert [item.artist for item in run.input.references] == ["Artist"]
    assert [item["artist"] for item in run.input.feedback] == ["Liked"]
    assert harness.executor.calls == [(run.id, 1)]


@pytest.mark.parametrize(
    ("instructions", "references"),
    [
        ("  ", [positive()]),
        ("More", []),
        ("More", [DiscoveryReference(kind="manual", artist="A", title="B", polarity="negative")]),
    ],
)
async def test_run_needs_instructions_and_a_positive_reference(
    instructions: str, references: list[DiscoveryReference]
) -> None:
    harness = Harness()
    session = await harness.session()

    with pytest.raises(InvalidResearchInputError, match="positive reference"):
        await harness.service.create(session.id, instructions, references, DiscoverySettings())

    assert harness.runs.runs == {}


async def test_unknown_session_is_not_found() -> None:
    with pytest.raises(SessionNotFoundError):
        await Harness().service.create(uuid4(), "More", [positive()], DiscoverySettings())


async def test_session_allows_one_active_run() -> None:
    harness = Harness(executor=FakeExecutor(gated=True))
    session = await harness.session()
    await harness.service.create(session.id, "More", [positive()], DiscoverySettings())

    with pytest.raises(RunRejectedError, match="active run"):
        await harness.service.create(session.id, "More", [positive()], DiscoverySettings())

    harness.executor.gate.set()
    await harness.idle()


async def test_full_queue_stores_nothing() -> None:
    harness = Harness(executor=FakeExecutor(gated=True), queue=AsyncioRunQueue(max_pending=1))
    first, second, third = [await harness.session() for _ in range(3)]
    await harness.service.create(first.id, "More", [positive()], DiscoverySettings())
    await harness.executor.started.wait()
    await harness.service.create(second.id, "More", [positive()], DiscoverySettings())

    with pytest.raises(QueueRejectedError, match="full"):
        await harness.service.create(third.id, "More", [positive()], DiscoverySettings())

    assert len(harness.runs.runs) == 2
    assert (await harness.sessions.get(third.id)).run_ids == []  # type: ignore[union-attr]
    harness.executor.gate.set()
    await harness.idle()


async def test_failed_store_withdraws_the_queued_job() -> None:
    harness = Harness(runs=FlakyRuns())
    session = await harness.session()
    harness.runs.fail_save = True  # type: ignore[attr-defined]

    with pytest.raises(OSError, match="disk full"):
        await harness.service.create(session.id, "More", [positive()], DiscoverySettings())

    assert harness.queue.queued() == []
    assert harness.executor.calls == []


async def test_other_process_owning_the_runs_rejects_new_ones() -> None:
    harness = Harness(owner=False)
    session = await harness.session()

    assert not harness.service.ready
    with pytest.raises(RunRejectedError, match="Another process"):
        await harness.service.create(session.id, "More", [positive()], DiscoverySettings())


async def test_cancel_queued_run() -> None:
    harness = Harness(executor=FakeExecutor(gated=True))
    first, second = await harness.session(), await harness.session()
    await harness.service.create(first.id, "More", [positive()], DiscoverySettings())
    await harness.executor.started.wait()
    queued = await harness.service.create(second.id, "More", [positive()], DiscoverySettings())

    cancelled = await harness.service.cancel(queued.id)

    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None
    assert (await stored(harness, queued.id)).status == "cancelled"
    assert await statuses(harness, queued.id) == ["queued", "cancelled"]
    assert harness.queue.queued() == []
    harness.executor.gate.set()
    await harness.idle()
    assert [run_id for run_id, _ in harness.executor.calls] != [queued.id]


async def test_cancel_running_run() -> None:
    harness = Harness(executor=FakeExecutor(gated=True))
    session = await harness.session()
    run = await harness.service.create(session.id, "More", [positive()], DiscoverySettings())
    await harness.executor.started.wait()
    state = harness.service.active[run.id]

    answer = await harness.service.cancel(run.id)
    assert state.cancelled()
    await harness.idle()

    assert answer.status == "running"
    assert (await stored(harness, run.id)).status == "cancelled"
    assert await statuses(harness, run.id) == ["queued", "running", "cancelled"]
    assert harness.service.cancelled == set()


async def test_cancel_of_a_run_that_is_not_active_here() -> None:
    harness = Harness()
    session = await harness.session()
    finished = DiscoveryRun(session_id=session.id, input=DiscoveryRunInput(), status="completed")
    orphan = DiscoveryRun(session_id=session.id, input=DiscoveryRunInput(), status="running")
    await harness.runs.save(finished)
    await harness.runs.save(orphan)

    for run in (finished, orphan):
        with pytest.raises(RunNotActiveError, match="not active"):
            await harness.service.cancel(run.id)
    with pytest.raises(RunNotFoundError):
        await harness.service.cancel(uuid4())


async def test_shutdown_interrupts_queue_without_starting_new_runs() -> None:
    harness = Harness(executor=FakeExecutor(gated=True))
    first, second = await harness.session(), await harness.session()
    active = await harness.service.create(first.id, "More", [positive()], DiscoverySettings())
    await harness.executor.started.wait()
    queued = await harness.service.create(second.id, "More", [positive()], DiscoverySettings())

    await harness.service.shutdown()

    assert (await stored(harness, active.id)).status == "cancelled"
    interrupted = await stored(harness, queued.id)
    assert (interrupted.status, interrupted.failure_code) == ("failed", "interrupted")
    assert [run_id for run_id, _ in harness.executor.calls] == [active.id]
    assert not harness.service.ready
    with pytest.raises(RunRejectedError, match="shutting down"):
        await harness.service.create(second.id, "More", [positive()], DiscoverySettings())


async def test_shutdown_survives_an_unreadable_queued_run() -> None:
    harness = Harness(executor=FakeExecutor(gated=True), runs=FlakyRuns())
    first, second = await harness.session(), await harness.session()
    await harness.service.create(first.id, "More", [positive()], DiscoverySettings())
    await harness.executor.started.wait()
    await harness.service.create(second.id, "More", [positive()], DiscoverySettings())
    harness.runs.fail_get = True  # type: ignore[attr-defined]

    await harness.service.shutdown()

    assert harness.queue.closing


async def test_reconcile_interrupts_unfinished_runs() -> None:
    harness = Harness()
    session = await harness.session()
    runs = [
        DiscoveryRun(session_id=session.id, input=DiscoveryRunInput(), status=status)
        for status in ("queued", "running", "completed")
    ]
    for run in runs:
        await harness.runs.save(run)

    await harness.service.reconcile()

    assert [(await stored(harness, run.id)).status for run in runs] == [
        "failed",
        "failed",
        "completed",
    ]
    assert (await stored(harness, runs[0].id)).failure_code == "interrupted"
    assert await statuses(harness, runs[1].id) == ["failed"]


async def test_unsaved_run_is_still_reported() -> None:
    harness = Harness(runs=FlakyRuns())
    session = await harness.session()
    run = DiscoveryRun(session_id=session.id, input=DiscoveryRunInput(), status="running")
    await harness.runs.save(run)
    harness.runs.fail_save = True  # type: ignore[attr-defined]

    await harness.service.reconcile()

    assert await statuses(harness, run.id) == ["failed"]
    assert (await stored(harness, run.id)).status == "running"


async def test_diagnostic_failure_does_not_fail_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness()

    async def broken(run_id: UUID, payload: dict[str, object]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(harness.diagnostics, "write", broken)
    session = await harness.session()

    run = await harness.service.create(session.id, "More", [positive()], DiscoverySettings())
    await harness.idle()

    assert (await stored(harness, run.id)).status == "completed"


async def test_run_whose_session_vanished_fails() -> None:
    harness = Harness()
    session = await harness.session()
    run = await harness.service.create(session.id, "More", [positive()], DiscoverySettings())
    del harness.sessions.sessions[session.id]

    await harness.idle()

    failed = await stored(harness, run.id)
    assert (failed.status, failed.failure_code) == ("failed", "process_failure")
    assert failed.failure_message == "SessionNotFoundError: Session not found"
    assert harness.executor.calls == []
    assert run.id not in harness.diagnostics.payloads


async def test_run_cancelled_while_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness()
    session = await harness.session()
    run = await harness.service.create(session.id, "More", [positive()], DiscoverySettings())

    async def cancelled(*_: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr("mixhound.services.runs.run_history", cancelled)
    await harness.idle()

    assert (await stored(harness, run.id)).status == "cancelled"


async def test_job_skips_a_run_that_is_no_longer_queued() -> None:
    harness = Harness()
    session = await harness.session()
    run = await harness.service.create(session.id, "More", [positive()], DiscoverySettings())
    finished = await stored(harness, run.id)
    finished.status = "failed"
    await harness.runs.save(finished)

    await harness.idle()

    assert harness.executor.calls == []


async def test_progress_of_the_active_run() -> None:
    harness = Harness(executor=FakeExecutor(gated=True))
    session = await harness.session()
    run = await harness.service.create(session.id, "More", [positive()], DiscoverySettings())
    await harness.executor.started.wait()

    progress = harness.service.progress(run.id)

    assert progress is not None
    assert (progress.stage, progress.timeout) == ("research", 600)
    assert harness.service.progress(uuid4()) is None
    harness.executor.gate.set()
    await harness.idle()
    assert harness.service.progress(run.id) is None


async def test_follow_streams_status_and_progress_until_the_run_finishes() -> None:
    harness = Harness(executor=FakeExecutor(gated=True))
    session = await harness.session()
    run = await harness.service.create(session.id, "More", [positive()], DiscoverySettings())

    async def collect() -> list[RunStatusEvent | RunProgress]:
        return [event async for event in await harness.service.follow(run.id)]

    follower = asyncio.create_task(collect())
    await harness.executor.started.wait()
    await asyncio.sleep(0.05)
    harness.executor.gate.set()
    events = await follower

    assert [event.status for event in events if isinstance(event, RunStatusEvent)] == [
        "queued",
        "running",
        "completed",
    ]
    assert any(isinstance(event, RunProgress) and event.stage == "research" for event in events)


async def test_follow_of_a_finished_run_replays_its_events() -> None:
    harness = Harness()
    session = await harness.session()
    run = await harness.service.create(session.id, "More", [positive()], DiscoverySettings())
    await harness.idle()

    events = [event async for event in await harness.service.follow(run.id)]

    assert [event.status for event in events] == ["queued", "running", "completed"]  # type: ignore[union-attr]


@pytest.mark.parametrize("status", ["completed", "running"])
async def test_follow_of_an_untracked_run_reads_it_once(status: str) -> None:
    harness = Harness()
    session = await harness.session()
    run = DiscoveryRun.model_validate(
        {"session_id": session.id, "input": DiscoveryRunInput(), "status": status}
    )
    await harness.runs.save(run)

    events = [event async for event in await harness.service.follow(run.id)]

    assert events == [RunStatusEvent.model_validate({"status": status})]


async def test_follow_flushes_events_emitted_while_untracking() -> None:
    harness = Harness(runs=FlakyRuns())
    session = await harness.session()
    run = DiscoveryRun(session_id=session.id, input=DiscoveryRunInput(), status="completed")
    await harness.runs.save(run)
    reads: list[UUID] = []

    def worker_reports(run_id: UUID) -> None:
        # The first read checks the run exists; the worker reports just before the second.
        reads.append(run_id)
        if len(reads) == 2:
            harness.events.emit(run)

    harness.runs.on_get = worker_reports  # type: ignore[attr-defined]

    events = [event async for event in await harness.service.follow(run.id)]

    assert events == [RunStatusEvent(status="completed")]


async def test_follow_of_an_unknown_run_is_not_found() -> None:
    with pytest.raises(RunNotFoundError):
        await Harness().service.follow(uuid4())
