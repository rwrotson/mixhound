import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import UUID

from fastapi import FastAPI

from mixhound.api.app import build_api
from mixhound.config import ApiSettings
from mixhound.infrastructure.queue import AsyncioRunQueue
from mixhound.infrastructure.storage.memory import MemoryDiagnostics, MemoryRuns, MemorySessions
from mixhound.memory.context import RunHistory
from mixhound.models.domain import DiscoveryReference, DiscoverySession, now_utc
from mixhound.models.responses import ProvidersOut
from mixhound.services.pages import PageInspector
from mixhound.services.run_events import RunEvents
from mixhound.services.run_state import RESEARCH_STAGE, RunState
from mixhound.services.runs import RunService
from mixhound.services.sessions import SessionService
from tests.fakes import FakeFetch


class FakeExecutor:
    """Completes runs; while `gate` is clear, a run waits in the research stage."""

    def __init__(self, *, gated: bool = False) -> None:
        self.gate = asyncio.Event()
        if not gated:
            self.gate.set()
        self.started = asyncio.Event()
        self.calls: list[tuple[UUID, int]] = []

    async def execute(
        self, state: RunState, session: DiscoverySession, history: RunHistory
    ) -> dict[str, object]:
        run = state.run
        self.calls.append((run.id, len(history.runs)))
        try:
            with state.enter(RESEARCH_STAGE):
                self.started.set()
                await self.gate.wait()
        except asyncio.CancelledError:
            run.status = "cancelled"
        else:
            run.status = "completed"
        run.finished_at = now_utc()
        return {"run_id": str(run.id), "session": session.name}


@dataclass
class Harness:
    """A run and a session service over memory storage, a real queue and a fake executor."""

    executor: FakeExecutor = field(default_factory=FakeExecutor)
    sessions: MemorySessions = field(default_factory=MemorySessions)
    runs: MemoryRuns = field(default_factory=MemoryRuns)
    diagnostics: MemoryDiagnostics = field(default_factory=MemoryDiagnostics)
    queue: AsyncioRunQueue = field(default_factory=AsyncioRunQueue)
    events: RunEvents = field(default_factory=RunEvents)
    owner: bool = True
    providers: ProvidersOut = field(
        default_factory=lambda: ProvidersOut(
            model="test",
            helper_model=None,
            fallback_model=None,
            web_search=False,
            audio=False,
            catalog_keys={},
        )
    )
    fetch: FakeFetch = field(default_factory=lambda: FakeFetch({}))

    def __post_init__(self) -> None:
        self.service = RunService(
            self.sessions,
            self.runs,
            self.diagnostics,
            self.queue,
            self.executor,
            events=self.events,
            owner=self.owner,
            follow_seconds=0.01,
        )
        self.session_service = SessionService(self.sessions, self.runs, PageInspector(self.fetch))

    async def session(self, *references: DiscoveryReference) -> DiscoverySession:
        """Store a session with the given references."""
        session = DiscoverySession(name="Test", references=list(references))
        await self.sessions.save(session)
        return session

    async def idle(self) -> None:
        """Wait until the queue has run every job."""
        worker = self.queue.worker
        if worker is not None:
            await worker


def positive() -> DiscoveryReference:
    """A positive manual reference."""
    return DiscoveryReference(kind="manual", artist="Artist", title="Title")


@dataclass(frozen=True)
class Resources:
    """What the API routes read from the lifespan state."""

    sessions: SessionService
    runs: RunService
    providers: ProvidersOut


def api(harness: Harness) -> FastAPI:
    """The HTTP application over a harness's services."""

    @asynccontextmanager
    async def lifespan() -> AsyncIterator[Resources]:
        yield Resources(harness.session_service, harness.service, harness.providers)

    return build_api(ApiSettings(), lifespan)
