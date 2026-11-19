import json
import os
from pathlib import Path

import pytest

from mixhound.infrastructure import cache as disk
from mixhound.infrastructure.cache import DiskCache, MemoryCache


async def test_disk_cache_keeps_values_until_they_expire(tmp_path: Path) -> None:
    now = [1000.0]
    cache = DiskCache(tmp_path, clock=lambda: now[0])

    await cache.set("ns", "a", {"v": 1}, 10)

    assert await cache.get("ns", "a") == (True, {"v": 1})
    assert await cache.get("ns", "missing") == (False, None)
    now[0] += 11
    assert await cache.get("ns", "a") == (False, None)
    assert (cache.hits, cache.misses) == (1, 2)
    assert list(tmp_path.glob("ns/*/*.json")) == []


async def test_disabled_disk_cache_stores_nothing(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, enabled=False)

    await cache.set("ns", "a", 1, 10)

    assert await cache.get("ns", "a") == (False, None)
    assert not any(tmp_path.iterdir())


async def test_unreadable_and_unserializable_entries_are_misses(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    await cache.set("ns", "a", 1, 10)
    next(tmp_path.glob("ns/*/*.json")).write_text("[1]")

    assert await cache.get("ns", "a") == (False, None)
    await cache.set("ns", "b", object(), 10)
    assert await cache.get("ns", "b") == (False, None)
    assert list(tmp_path.glob("ns/*/.*.tmp")) == []


async def test_prune_drops_expired_unreadable_and_oldest_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    cache = DiskCache(tmp_path, clock=lambda: now[0])
    await cache.prune()
    for key, ttl in (("old", 1), ("a", 100), ("b", 100), ("c", 100)):
        await cache.set("ns", key, key, ttl)
    for index, key in enumerate(("old", "a", "b", "c")):
        os.utime(cache._path("ns", key), (index, index))
    broken = tmp_path / "ns" / "zz" / "broken.json"
    broken.parent.mkdir()
    broken.write_text("{")
    monkeypatch.setattr(disk, "MAX_ENTRIES", 3)
    now[0] += 50

    await cache.prune()

    kept = [json.loads(path.read_text())["value"] for path in tmp_path.glob("ns/*/*.json")]
    assert sorted(kept) == ["b", "c"]


async def test_memory_cache_expires_entries() -> None:
    now = [0.0]
    cache = MemoryCache(clock=lambda: now[0])

    await cache.set("ns", "a", [1], 5)

    assert await cache.get("ns", "a") == (True, [1])
    now[0] = 6
    assert await cache.get("ns", "a") == (False, None)
