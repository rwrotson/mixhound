import asyncio
import copy
import logging
from collections.abc import AsyncIterator
from typing import Protocol
from uuid import UUID

from mixhound.memory.context import RunHistory, feedback_snapshot, run_history
from mixhound.memory.repository import DiagnosticsRepository, RunRepository, SessionRepository
from mixhound.models.domain import (
    DiscoveryReference,
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySession,
    DiscoverySettings,
    now_utc,
)
from mixhound.models.state import RunProgress, RunStatusEvent
from mixhound.services.errors import ConflictError, InvalidInputError
from mixhound.services.ports import RunQueue
from mixhound.services.run_events import RunEvents
from mixhound.services.run_state import RunState
from mixhound.services.sessions import require_run, require_session

log = logging.getLogger(__name__)

ACTIVE_STATUSES = frozenset({"queued", "running"})
# How often a follower re-reads the progress of an active run.
FOLLOW_SECONDS = 1.0


class RunExecutor(Protocol):
    """Runs one run's stages and records on the run how it ended."""

    async def execute(
        self, state: RunState, session: DiscoverySession, history: RunHistory
    ) -> dict[str, object]:
        """Run every stage and return the run's diagnostic."""
        ...


class InvalidResearchInputError(InvalidInputError):
    """A run request lacks usable instructions or a positive reference."""


class RunRejectedError(ConflictError):
    """A run cannot be accepted now."""


class RunNotActiveError(ConflictError):
    """A run is no longer queued or running here."""


class RunService:
    """Creates, queues, executes, cancels and follows runs, and recovers abandoned ones."""

    def __init__(  # noqa: PLR0913 - one parameter per injected collaborator
        self,
        sessions: SessionRepository,
        runs: RunRepository,
        diagnostics: DiagnosticsRepository,
        queue: RunQueue,
        executor: RunExecutor,
        *,
        events: RunEvents,
        owner: bool = True,
        follow_seconds: float = FOLLOW_SECONDS,
    ) -> None:
        self.sessions = sessions
        self.runs = runs
        self.diagnostics = diagnostics
        self.queue = queue
        self.executor = executor
        self.events = events
        # False when another process owns the stored runs: this one must not change them.
        self.owner = owner
        self.follow_seconds = follow_seconds
        self.closing = False
        # Serialises run creation, cancellation and shutdown with the start of a queued run.
        self.lock = asyncio.Lock()
        self.active: dict[UUID, RunState] = {}
        self.cancelled: set[UUID] = set()

    @property
    def ready(self) -> bool:
        """Whether new runs are accepted."""
        return self.owner and not self.closing

    async def create(
        self,
        session_id: UUID,
        instructions: str,
        references: list[DiscoveryReference],
        settings: DiscoverySettings,
    ) -> DiscoveryRun:
        """Snapshot the session's feedback and queue a new run."""
        async with self.lock:
            if not self.owner:
                raise RunRejectedError("Another process owns the runs")
            if self.closing:
                raise RunRejectedError("Research is shutting down")
            session = await require_session(self.sessions, session_id)
            if not instructions.strip() or not any(
                reference.polarity == "positive" for reference in references
            ):
                raise InvalidResearchInputError("A prompt and a positive reference are required")
            for run_id in session.run_ids:
                previous = await self.runs.get(run_id)
                if previous is not None and previous.status in ACTIVE_STATUSES:
                    raise RunRejectedError("Session already has an active run")
            run = DiscoveryRun(
                session_id=session_id,
                input=DiscoveryRunInput(
                    references=copy.deepcopy(
                        [item for item in references if item.polarity != "neutral"]
                    ),
                    instructions=instructions.strip(),
                    settings=settings,
                    feedback=await feedback_snapshot(self.runs, session),
                    refinement=bool(session.run_ids),
                ),
            )
            # The job waits for this lock, so it starts only after the run is stored.
            position = await self.queue.submit(run.id, lambda: self._execute(run.id))
            try:
                await self.runs.save(run)
                session.run_ids.append(run.id)
                session.updated_at = now_utc()
                await self.sessions.save(session)
            except BaseException:
                await self.queue.cancel(run.id)
                raise
            self.events.emit(run, position)
            return run

    async def get(self, run_id: UUID) -> DiscoveryRun:
        """Return one run."""
        return await require_run(self.runs, run_id)

    def progress(self, run_id: UUID) -> RunProgress | None:
        """The active run's stage and live research counts; None unless it runs here."""
        state = self.active.get(run_id)
        if state is None:
            return None
        return state.progress(state.run.input.settings.minutes * 60)

    async def cancel(self, run_id: UUID) -> DiscoveryRun:
        """Cancel a queued or running run and return its stored state."""
        async with self.lock:
            run = await require_run(self.runs, run_id)
            if run.status not in ACTIVE_STATUSES:
                raise RunNotActiveError("Run is not active")
            if run_id in self.active:
                # The pipeline records the cancellation and the job stores it.
                self.cancelled.add(run_id)
                await self.queue.cancel(run_id)
                return run
            if not await self.queue.cancel(run_id):
                raise RunNotActiveError("Run is not active")
            run.status, run.finished_at = "cancelled", now_utc()
            await self._store(run)
            return run

    async def follow(self, run_id: UUID) -> AsyncIterator[RunStatusEvent | RunProgress]:
        """Check the run exists and return its event stream."""
        await require_run(self.runs, run_id)
        return self._follow(run_id)

    async def _follow(self, run_id: UUID) -> AsyncIterator[RunStatusEvent | RunProgress]:
        """Yield the run's status events, and its progress while it runs, until it finishes.

        While this process tracks the run, events come from memory; the stored run is read
        once, when the run is no longer tracked and no terminal status was reported.
        """
        sent = 0
        progress: RunProgress | None = None
        while True:
            history = self.events.history(run_id)
            for event in history[sent:]:
                yield event
            sent = len(history)
            if history and history[-1].terminal:
                return
            if self.queue.tracked(run_id) or run_id in self.active:
                current = self.progress(run_id)
                if current is not None and current != progress:
                    progress = current
                    yield current
                await self.events.wait(run_id, self.follow_seconds)
                continue
            run = await self.runs.get(run_id)
            # Events emitted between the history read and untracking.
            history = self.events.history(run_id)
            for event in history[sent:]:
                yield event
            # A finished run without history, or an active status no worker here owns.
            if run is not None and not (history and history[-1].terminal):
                yield RunStatusEvent(status=run.status, failure_code=run.failure_code)
            return

    async def reconcile(self) -> None:
        """Mark runs a previous process left queued or running as interrupted."""
        for run in await self.runs.unfinished():
            if not self.queue.tracked(run.id):
                await self._interrupt(run)

    async def shutdown(self) -> None:
        """Stop accepting runs, cancel the active one and interrupt the queued ones."""
        async with self.lock:
            self.closing = True
            queued = self.queue.queued()
            self.cancelled.update(self.active)
        await self.queue.close()
        for run_id in queued:
            try:
                run = await self.runs.get(run_id)
                if run is not None and run.status == "queued":
                    await self._interrupt(run)
            except Exception:
                log.exception("run_interrupt_failed run=%s", run_id)

    async def _execute(self, run_id: UUID) -> None:
        """Run one queued run and store how it ended."""
        async with self.lock:
            run = await self.runs.get(run_id)
        if run is None or run.status != "queued":
            return
        state = RunState(run, cancelled=lambda: run_id in self.cancelled)
        self.active[run_id] = state
        try:
            diagnostic = await self._run(state)
            # A cancellation arriving now must not leave the run stored as running.
            await asyncio.shield(self._finish(run, diagnostic))
        finally:
            self.active.pop(run_id, None)
            self.cancelled.discard(run_id)

    async def _run(self, state: RunState) -> dict[str, object]:
        """Mark the run running, load its session and history and execute it."""
        run = state.run
        try:
            run.status = "running"
            await self.runs.save(run)
            self.events.emit(run)
            session = await require_session(self.sessions, run.session_id)
            history = await run_history(self.sessions, self.runs, run)
        except asyncio.CancelledError:
            run.status, run.finished_at = "cancelled", now_utc()
            return {}
        except Exception as error:
            log.exception("run_start_failed run=%s", run.id)
            run.status, run.failure_code = "failed", "process_failure"
            run.failure_message = f"{type(error).__name__}: {error}"[:500]
            run.finished_at = now_utc()
            return {}
        return await self.executor.execute(state, session, history)

    async def _finish(self, run: DiscoveryRun, diagnostic: dict[str, object]) -> None:
        """Store the diagnostic, then the run; a diagnostic failure does not fail the run."""
        if diagnostic:
            try:
                await self.diagnostics.write(run.id, diagnostic)
            except Exception:
                log.exception("run_diagnostic_write_failed run=%s", run.id)
        await self._store(run)

    async def _interrupt(self, run: DiscoveryRun) -> None:
        """Mark a run failed as interrupted."""
        run.status, run.failure_code, run.finished_at = "failed", "interrupted", now_utc()
        await self._store(run)

    async def _store(self, run: DiscoveryRun) -> None:
        """Save the run and report its status, even when saving fails."""
        try:
            await self.runs.save(run)
        except Exception:
            log.exception("run_update_failed run=%s", run.id)
        self.events.emit(run)
