from typing import Literal

from pydantic import BaseModel


class RunProgress(BaseModel):
    """Where an active run is now; kept in memory only, never stored."""

    stage: str
    # The stage's place among the run's stages, 1-based.
    step: int
    steps: int
    # Research only: seconds since the agent started, and its time limit.
    elapsed: float = 0.0
    timeout: float = 0.0
    searches: int = 0
    # Candidates the research has written to its interim notes so far.
    candidates: int = 0


TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class RunStatusEvent(BaseModel):
    """A change of a run's status, as reported to clients."""

    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    # Place in the queue for a queued run, 1 being the next to start; 0 otherwise.
    position: int = 0
    failure_code: str | None = None

    @property
    def terminal(self) -> bool:
        """Whether the run has finished."""
        return self.status in TERMINAL_STATUSES
