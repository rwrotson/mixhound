import asyncio
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)

MAX_ENTRIES = 20_000


class DiskCache:
    """Small on-disk cache with a time-to-live per entry."""

    def __init__(
        self, root: Path, *, clock: Callable[[], float] = time.time, enabled: bool = True
    ) -> None:
        self.root = root
        self.clock = clock
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.root / namespace / digest[:2] / f"{digest}.json"

    async def get(self, namespace: str, key: str) -> tuple[bool, object]:
        """Return (found, value); expired or unreadable entries count as missing."""
        if not self.enabled:
            return False, None
        found = await asyncio.to_thread(self._get, self._path(namespace, key))
        if found[0]:
            self.hits += 1
        else:
            self.misses += 1
        return found

    def _get(self, path: Path) -> tuple[bool, object]:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return False, None
        if not isinstance(entry, dict) or entry.get("expires", 0) < self.clock():
            path.unlink(missing_ok=True)
            return False, None
        return True, entry.get("value")

    async def set(self, namespace: str, key: str, value: object, ttl: float) -> None:
        """Store a value for a time-to-live; write failures are only logged."""
        if self.enabled:
            await asyncio.to_thread(self._set, namespace, key, value, ttl)

    def _set(self, namespace: str, key: str, value: object, ttl: float) -> None:
        path = self._path(namespace, key)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps({"expires": self.clock() + ttl, "value": value}, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError) as error:
            log.info("cache_write_failed", namespace=namespace, error=str(error))
        finally:
            temporary.unlink(missing_ok=True)

    async def prune(self) -> None:
        """Delete expired entries and, past MAX_ENTRIES, the oldest ones."""
        await asyncio.to_thread(self._prune)

    def _prune(self) -> None:
        if not self.root.is_dir():
            return
        files = sorted(self.root.glob("*/*/*.json"), key=lambda path: path.stat().st_mtime)
        now = self.clock()
        for index, path in enumerate(files):
            try:
                expired = json.loads(path.read_text(encoding="utf-8")).get("expires", 0) < now
            except OSError, ValueError, AttributeError:
                expired = True
            if expired or index < len(files) - MAX_ENTRIES:
                path.unlink(missing_ok=True)


class MemoryCache:
    """In-process cache with a time-to-live per entry."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self.entries: dict[tuple[str, str], tuple[float, object]] = {}

    async def get(self, namespace: str, key: str) -> tuple[bool, object]:
        """Return (found, value); expired entries count as missing."""
        entry = self.entries.get((namespace, key))
        if entry is None or entry[0] < self.clock():
            self.entries.pop((namespace, key), None)
            return False, None
        return True, entry[1]

    async def set(self, namespace: str, key: str, value: object, ttl: float) -> None:
        """Store a value for a time-to-live."""
        self.entries[namespace, key] = (self.clock() + ttl, value)
