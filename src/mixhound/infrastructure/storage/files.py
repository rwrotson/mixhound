import asyncio
import fcntl
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError

from mixhound.infrastructure.storage.records import ACTIVE_STATUSES, check_run_update
from mixhound.models.domain import DiscoveryIndex, DiscoveryRun, DiscoverySession, DiscoverySettings
from mixhound.services.errors import ConflictError

DIAGNOSTICS_DAYS = 30


class StoredDataError(ConflictError):
    """A stored file cannot be read."""


class FileStore:
    """JSON files under one directory, written atomically under an exclusive lock."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Hold the process and file locks of the store."""
        with self._lock:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            with (self.root / ".lock").open("a+b") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

    def read[M: BaseModel](self, path: Path, kind: type[M]) -> M | None:
        """Return a stored model, or None when its file does not exist."""
        try:
            return kind.model_validate_json(path.read_bytes())
        except FileNotFoundError:
            return None
        except (OSError, ValueError, ValidationError) as error:
            raise StoredDataError(f"Unreadable file: {path.name}") from error

    @staticmethod
    def write(path: Path, content: str, mode: int = 0o600) -> None:
        """Replace a file atomically and durably."""
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    async def run[T](self, work: Callable[[], T]) -> T:
        """Run blocking file work inside a transaction, off the event loop."""

        def locked() -> T:
            with self.transaction():
                return work()

        return await asyncio.to_thread(locked)


class FileSessions:
    """Sessions in `sessions/<id>.json`, ordered and defaulted by `index.json`."""

    def __init__(self, store: FileStore) -> None:
        self.store = store

    def _index(self) -> DiscoveryIndex:
        return self.store.read(self.store.root / "index.json", DiscoveryIndex) or DiscoveryIndex()

    def _path(self, session_id: UUID) -> Path:
        return self.store.root / "sessions" / f"{session_id}.json"

    async def default_settings(self) -> DiscoverySettings:
        """Return the settings new sessions start from."""
        return await self.store.run(lambda: self._index().settings)

    async def save_default_settings(self, settings: DiscoverySettings) -> None:
        """Replace the settings new sessions start from."""

        def save() -> None:
            index = self._index()
            index.settings = settings
            self.store.write(self.store.root / "index.json", index.model_dump_json(indent=2))

        await self.store.run(save)

    async def list(self) -> list[DiscoverySession]:
        """Return every session, oldest first."""

        def read() -> list[DiscoverySession]:
            found = (
                self.store.read(self._path(item), DiscoverySession)
                for item in self._index().session_ids
            )
            return [session for session in found if session is not None]

        return await self.store.run(read)

    async def get(self, session_id: UUID) -> DiscoverySession | None:
        """Return a session, or None when it does not exist."""
        return await self.store.run(
            lambda: self.store.read(self._path(session_id), DiscoverySession)
        )

    async def save(self, session: DiscoverySession) -> None:
        """Create or replace a session; the session file is written before the index."""

        def save() -> None:
            self.store.write(self._path(session.id), session.model_dump_json(indent=2))
            index = self._index()
            if session.id not in index.session_ids:
                index.session_ids.append(session.id)
                self.store.write(self.store.root / "index.json", index.model_dump_json(indent=2))

        await self.store.run(save)


class FileRuns:
    """Runs in `runs/<id>.json`."""

    def __init__(self, store: FileStore) -> None:
        self.store = store

    def _path(self, run_id: UUID) -> Path:
        return self.store.root / "runs" / f"{run_id}.json"

    async def get(self, run_id: UUID) -> DiscoveryRun | None:
        """Return a run, or None when it does not exist."""
        return await self.store.run(lambda: self.store.read(self._path(run_id), DiscoveryRun))

    async def save(self, run: DiscoveryRun) -> None:
        """Create or replace a run; its input never changes after creation."""

        def save() -> None:
            check_run_update(self.store.read(self._path(run.id), DiscoveryRun), run)
            self.store.write(self._path(run.id), run.model_dump_json(indent=2))

        await self.store.run(save)

    async def unfinished(self) -> list[DiscoveryRun]:
        """Return runs left queued or running."""

        def read() -> list[DiscoveryRun]:
            found = (
                self.store.read(path, DiscoveryRun)
                for path in sorted((self.store.root / "runs").glob("*.json"))
            )
            return [run for run in found if run is not None and run.status in ACTIVE_STATUSES]

        return await self.store.run(read)


class FileDiagnostics:
    """One diagnostic per run in `diagnostics/<id>.json`, readable only by the owner."""

    def __init__(self, store: FileStore, *, clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.clock = clock

    async def write(self, run_id: UUID, payload: dict[str, object]) -> None:
        """Store the diagnostic of a run and delete those older than 30 days."""
        directory = self.store.root / "diagnostics"

        def write() -> None:
            self._prune(directory)
            content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            self.store.write(directory / f"{run_id}.json", content)

        await self.store.run(write)

    def _prune(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        cutoff = self.clock() - DIAGNOSTICS_DAYS * 86_400
        for path in directory.glob("*.json"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
