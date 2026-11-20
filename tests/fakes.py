import json
from collections.abc import Awaitable, Callable, Mapping
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel

from mixhound.models.domain import (
    DiscoveryRun,
    DiscoverySession,
    DiscoverySettings,
    SonicProfile,
)
from mixhound.models.research import LinkSearchOutput, ResearchOutput
from mixhound.services.agentic import AgenticWorkspace
from mixhound.services.ports import (
    HTML,
    JSON,
    PAGE_BYTES,
    Accept,
    CatalogRecording,
    CatalogUnavailableError,
    FetchError,
    Samples,
)
from mixhound.services.research import AgentCall
from mixhound.services.run_state import LlmUsage, RunState


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


class FakeCatalog:
    """Answers catalog requests from a map keyed by (artist, title)."""

    def __init__(
        self,
        recordings: dict[tuple[str, str], CatalogRecording] | None = None,
        *,
        down: bool = False,
    ) -> None:
        self.recordings = recordings or {}
        self.down = down
        self.calls: list[tuple[str, str, str, str | None, str | None]] = []

    async def _answer(
        self, profile: str, artist: str, title: str, version: str | None, isrc: str | None
    ) -> CatalogRecording | None:
        self.calls.append((profile, artist, title, version, isrc))
        if self.down:
            raise CatalogUnavailableError("down")
        return self.recordings.get((artist, title))

    async def reference(
        self, artist: str, title: str, *, version: str | None = None, isrc: str | None = None
    ) -> CatalogRecording | None:
        return await self._answer("reference", artist, title, version, isrc)

    async def candidate(
        self, artist: str, title: str, *, version: str | None = None, isrc: str | None = None
    ) -> CatalogRecording | None:
        return await self._answer("candidate", artist, title, version, isrc)


type Reply = bytes | dict[str, object] | list[object] | Exception


class FakeFetch:
    """Serves page bodies, JSON or errors by host and path, ignoring the query."""

    def __init__(self, replies: dict[str, Reply]) -> None:
        self.replies = replies
        self.requests: list[tuple[str, Accept]] = []

    def _reply(self, url: str, accept: Accept) -> Reply:
        self.requests.append((url, accept))
        parts = urlsplit(url)
        reply = self.replies.get(f"{parts.hostname}{parts.path}", FetchError("unreachable", 404))
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def page(
        self,
        url: str,
        *,
        max_bytes: int = PAGE_BYTES,
        truncate: bool = False,
        accept: Accept = HTML,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[str, bytes]:
        reply = self._reply(url, accept)
        return url, reply if isinstance(reply, bytes) else json.dumps(reply).encode()

    async def json(
        self,
        url: str,
        *,
        max_bytes: int = PAGE_BYTES,
        headers: Mapping[str, str] | None = None,
    ) -> object:
        reply = self._reply(url, JSON)
        return json.loads(reply) if isinstance(reply, bytes) else reply


class FakeDecoder:
    """Decodes a clip body to the samples registered for it."""

    def __init__(self, samples: dict[bytes, Samples | Exception | None] | None = None) -> None:
        self.samples = samples or {}
        self.decoded: list[bytes] = []

    def available(self) -> bool:
        return True

    async def decode(self, data: bytes) -> Samples | None:
        self.decoded.append(data)
        found = self.samples.get(data)
        if isinstance(found, Exception):
            raise found
        return found


type Outcome = BaseModel | BaseException


class FakeAgents:
    """Answers agent calls with a queued output or error per call label."""

    def __init__(
        self,
        outcomes: Mapping[str, Outcome] | None = None,
        orchestration: Callable[[AgenticWorkspace], Awaitable[None]] | None = None,
    ) -> None:
        self.outcomes = dict(outcomes or {})
        # What the orchestrator does through its workspace before it answers.
        self.orchestration = orchestration
        self.calls: list[tuple[AgentCall, str]] = []

    def _answer(self, state: RunState, call: AgentCall, prompt: str) -> BaseModel:
        self.calls.append((call, prompt))
        state.add_usage(call.label, LlmUsage(cost_usd=0.25, web_searches=3, turns=2))
        outcome = self.outcomes[call.label]
        if isinstance(outcome, BaseException):
            raise outcome
        state.answers[call.label] = outcome.model_dump_json()
        return outcome

    async def research(self, state: RunState, call: AgentCall, prompt: str) -> ResearchOutput:
        answer = self._answer(state, call, prompt)
        assert isinstance(answer, ResearchOutput)
        return answer

    async def profile(self, state: RunState, call: AgentCall, prompt: str) -> SonicProfile:
        answer = self._answer(state, call, prompt)
        assert isinstance(answer, SonicProfile)
        return answer

    async def find_links(self, state: RunState, call: AgentCall, prompt: str) -> LinkSearchOutput:
        answer = self._answer(state, call, prompt)
        assert isinstance(answer, LinkSearchOutput)
        return answer

    async def orchestrate(
        self, state: RunState, call: AgentCall, prompt: str, workspace: AgenticWorkspace
    ) -> ResearchOutput:
        if self.orchestration is not None:
            await self.orchestration(workspace)
        answer = self._answer(state, call, prompt)
        assert isinstance(answer, ResearchOutput)
        return answer
