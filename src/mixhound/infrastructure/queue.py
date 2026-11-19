import asyncio
import contextvars
from collections import deque
from uuid import UUID

import structlog

from mixhound.services.errors import ConflictError
from mixhound.services.ports import Job

log = structlog.get_logger(__name__)


class QueueRejectedError(ConflictError):
    """A run cannot be added to the queue."""


class AsyncioRunQueue:
    """Process-local FIFO that runs one job at a time."""

    def __init__(self, *, max_pending: int = 20) -> None:
        self.max_pending = max_pending
        self.pending: deque[tuple[UUID, Job]] = deque()
        self.active: UUID | None = None
        self.task: asyncio.Task[None] | None = None
        self.worker: asyncio.Task[None] | None = None
        self.closing = False

    def tracked(self, run_id: UUID) -> bool:
        """Whether a job is queued or running."""
        return run_id == self.active or any(item == run_id for item, _ in self.pending)

    async def submit(self, run_id: UUID, job: Job) -> int:
        """Queue a job and return its position, 1 being the next to start."""
        if self.closing:
            raise QueueRejectedError("The queue is shutting down")
        if self.tracked(run_id):
            raise QueueRejectedError("Run is already submitted")
        if len(self.pending) >= self.max_pending:
            raise QueueRejectedError("The queue is full")
        self.pending.append((run_id, job))
        if self.worker is None or self.worker.done():
            # A fresh context: the worker outlives the request that started it.
            self.worker = asyncio.create_task(self._drain(), context=contextvars.Context())
        return len(self.pending)

    async def cancel(self, run_id: UUID) -> bool:
        """Drop a queued job or cancel the running one; False when it is not tracked."""
        for item in self.pending:
            if item[0] == run_id:
                self.pending.remove(item)
                return True
        if self.active == run_id and self.task is not None:
            self.task.cancel()
            return True
        return False

    async def close(self) -> None:
        """Stop accepting jobs, drop queued ones and cancel the running one."""
        self.closing = True
        self.pending.clear()
        if self.task is not None:
            self.task.cancel()
        if self.worker is not None:
            await asyncio.gather(self.worker, return_exceptions=True)

    async def _drain(self) -> None:
        """Run queued jobs until the queue empties or closes."""
        while self.pending and not self.closing:
            run_id, job = self.pending.popleft()
            self.active = run_id
            task = self.task = asyncio.create_task(job())
            try:
                await asyncio.wait([task])
            finally:
                self.active, self.task = None, None
            if not task.cancelled() and (error := task.exception()) is not None:
                log.error("run_job_failed", run_id=str(run_id), exc_info=error)
