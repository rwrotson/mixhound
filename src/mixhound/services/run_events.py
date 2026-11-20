import asyncio
from uuid import UUID

from mixhound.models.domain import DiscoveryRun
from mixhound.models.state import RunStatusEvent

MAX_EVENT_HISTORIES = 200


class RunEvents:
    """In-memory status events of recent runs, which followers can wait for."""

    def __init__(self, max_histories: int = MAX_EVENT_HISTORIES) -> None:
        self.max_histories = max_histories
        self.histories: dict[UUID, list[RunStatusEvent]] = {}
        self.signals: dict[UUID, asyncio.Event] = {}

    def emit(self, run: DiscoveryRun, position: int = 0) -> None:
        """Record the run's current status and wake its followers."""
        history = self.histories.pop(run.id, [])
        history.append(
            RunStatusEvent(status=run.status, position=position, failure_code=run.failure_code)
        )
        # Re-inserted last, so the least recently active run is dropped first.
        self.histories[run.id] = history
        while len(self.histories) > self.max_histories:
            del self.histories[next(iter(self.histories))]
        signal = self.signals.pop(run.id, None)
        if signal is not None:
            signal.set()

    def history(self, run_id: UUID) -> list[RunStatusEvent]:
        """Return the status events reported for a run, oldest first."""
        return list(self.histories.get(run_id, []))

    async def wait(self, run_id: UUID, seconds: float) -> bool:
        """Wait for the run's next event; False when `seconds` pass first."""
        signal = self.signals.setdefault(run_id, asyncio.Event())
        try:
            async with asyncio.timeout(seconds):
                await signal.wait()
        except TimeoutError:
            return False
        return True
