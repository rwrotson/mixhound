import json
from collections.abc import Mapping
from urllib.parse import urlsplit
from uuid import UUID

from mixhound.models.domain import DiscoveryRun, DiscoverySession, DiscoverySettings
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
