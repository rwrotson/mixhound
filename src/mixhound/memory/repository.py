from typing import Protocol
from uuid import UUID

from mixhound.models.domain import DiscoveryRun, DiscoverySession, DiscoverySettings


class SessionRepository(Protocol):
    """Stores sessions in creation order, and the default settings of new sessions."""

    async def default_settings(self) -> DiscoverySettings:
        """Return the settings new sessions start from."""
        ...

    async def save_default_settings(self, settings: DiscoverySettings) -> None:
        """Replace the settings new sessions start from."""
        ...

    async def list(self) -> list[DiscoverySession]:
        """Return every session, oldest first."""
        ...

    async def get(self, session_id: UUID) -> DiscoverySession | None:
        """Return a session, or None when it does not exist."""
        ...

    async def save(self, session: DiscoverySession) -> None:
        """Create or replace a session."""
        ...


class RunRepository(Protocol):
    """Stores runs; a run's input never changes after creation."""

    async def get(self, run_id: UUID) -> DiscoveryRun | None:
        """Return a run, or None when it does not exist."""
        ...

    async def save(self, run: DiscoveryRun) -> None:
        """Create or replace a run."""
        ...

    async def unfinished(self) -> list[DiscoveryRun]:
        """Return runs left queued or running."""
        ...


class DiagnosticsRepository(Protocol):
    """Keeps one diagnostic document per run for a limited time."""

    async def write(self, run_id: UUID, payload: dict[str, object]) -> None:
        """Store the diagnostic of a run."""
        ...
