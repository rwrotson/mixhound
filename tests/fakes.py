from uuid import UUID

from mixhound.models.domain import DiscoveryRun, DiscoverySession, DiscoverySettings


class FakeSessions:
    def __init__(self, *sessions: DiscoverySession) -> None:
        self.sessions = {session.id: session for session in sessions}
        self.settings = DiscoverySettings()

    async def default_settings(self) -> DiscoverySettings:
        return self.settings

    async def save_default_settings(self, settings: DiscoverySettings) -> None:
        self.settings = settings

    async def list(self) -> list[DiscoverySession]:
        return list(self.sessions.values())

    async def get(self, session_id: UUID) -> DiscoverySession | None:
        return self.sessions.get(session_id)

    async def save(self, session: DiscoverySession) -> None:
        self.sessions[session.id] = session


class FakeRuns:
    def __init__(self, *runs: DiscoveryRun) -> None:
        self.runs = {run.id: run for run in runs}

    async def get(self, run_id: UUID) -> DiscoveryRun | None:
        return self.runs.get(run_id)

    async def save(self, run: DiscoveryRun) -> None:
        self.runs[run.id] = run

    async def unfinished(self) -> list[DiscoveryRun]:
        return [run for run in self.runs.values() if run.status in {"queued", "running"}]
