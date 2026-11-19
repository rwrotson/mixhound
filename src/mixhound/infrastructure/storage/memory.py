from uuid import UUID

from mixhound.infrastructure.storage.records import ACTIVE_STATUSES, check_run_update
from mixhound.models.domain import DiscoveryRun, DiscoverySession, DiscoverySettings


class MemorySessions:
    """Sessions kept in process memory."""

    def __init__(self) -> None:
        self.settings = DiscoverySettings()
        self.sessions: dict[UUID, DiscoverySession] = {}

    async def default_settings(self) -> DiscoverySettings:
        """Return the settings new sessions start from."""
        return self.settings.model_copy(deep=True)

    async def save_default_settings(self, settings: DiscoverySettings) -> None:
        """Replace the settings new sessions start from."""
        self.settings = settings.model_copy(deep=True)

    async def list(self) -> list[DiscoverySession]:
        """Return every session, oldest first."""
        return [session.model_copy(deep=True) for session in self.sessions.values()]

    async def get(self, session_id: UUID) -> DiscoverySession | None:
        """Return a session, or None when it does not exist."""
        session = self.sessions.get(session_id)
        return session.model_copy(deep=True) if session else None

    async def save(self, session: DiscoverySession) -> None:
        """Create or replace a session."""
        self.sessions[session.id] = session.model_copy(deep=True)


class MemoryRuns:
    """Runs kept in process memory."""

    def __init__(self) -> None:
        self.runs: dict[UUID, DiscoveryRun] = {}

    async def get(self, run_id: UUID) -> DiscoveryRun | None:
        """Return a run, or None when it does not exist."""
        run = self.runs.get(run_id)
        return run.model_copy(deep=True) if run else None

    async def save(self, run: DiscoveryRun) -> None:
        """Create or replace a run; its input never changes after creation."""
        check_run_update(self.runs.get(run.id), run)
        self.runs[run.id] = run.model_copy(deep=True)

    async def unfinished(self) -> list[DiscoveryRun]:
        """Return runs left queued or running."""
        return [
            run.model_copy(deep=True) for run in self.runs.values() if run.status in ACTIVE_STATUSES
        ]


class MemoryDiagnostics:
    """Diagnostics kept in process memory."""

    def __init__(self) -> None:
        self.payloads: dict[UUID, dict[str, object]] = {}

    async def write(self, run_id: UUID, payload: dict[str, object]) -> None:
        """Store the diagnostic of a run."""
        self.payloads[run_id] = payload
