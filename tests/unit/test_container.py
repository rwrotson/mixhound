from pathlib import Path

import pytest
from pydantic import SecretStr

from mixhound.config import CacheSettings, SearchSettings, Settings, StorageSettings
from mixhound.container import open_container
from mixhound.infrastructure.cache import DiskCache, MemoryCache
from mixhound.infrastructure.search.brave import BraveSearch
from mixhound.infrastructure.storage.files import FileRuns
from mixhound.infrastructure.storage.memory import MemoryRuns
from mixhound.models.domain import DiscoveryRun, DiscoveryRunInput, DiscoverySession
from mixhound.services.skills import RESEARCH_SKILL


async def test_container_opens_file_storage_and_skills(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, cache=CacheSettings(enabled=False))

    async with open_container(settings) as container:
        assert RESEARCH_SKILL in container.skills.skills
        assert isinstance(container.repositories.runs, FileRuns)
        assert isinstance(container.cache, DiskCache)
        assert not container.cache.enabled
        assert container.search is None


async def test_container_can_keep_everything_in_memory_and_search(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        storage=StorageSettings(backend="memory"),
        search=SearchSettings(brave_api_key=SecretStr("key")),
    )

    async with open_container(settings) as container:
        assert isinstance(container.repositories.runs, MemoryRuns)
        assert isinstance(container.cache, MemoryCache)
        assert isinstance(container.search, BraveSearch)


async def test_container_reconciles_runs_a_previous_process_left(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, cache=CacheSettings(enabled=False))
    async with open_container(settings) as container:
        session = DiscoverySession(name="Test")
        await container.repositories.sessions.save(session)
        left = DiscoveryRun(session_id=session.id, input=DiscoveryRunInput(), status="running")
        await container.repositories.runs.save(left)

    async with open_container(settings) as container:
        run = await container.runs.get(left.id)
        assert container.runs.ready
        assert container.providers.model == settings.llm.model
        assert container.providers.catalog_keys == {
            "lastfm": False,
            "discogs": False,
            "youtube": False,
            "spotify": False,
        }
        assert await container.sessions.list_all() == [session]

    assert (run.status, run.failure_code) == ("failed", "interrupted")


async def test_second_process_does_not_own_the_runs(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, cache=CacheSettings(enabled=False))
    async with open_container(settings) as first, open_container(settings) as second:
        assert first.runs.ready
        assert not second.runs.ready


async def test_failed_cache_prune_does_not_stop_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(self: DiskCache) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(DiskCache, "prune", broken)

    async with open_container(Settings(data_dir=tmp_path)) as container:
        assert container.runs.ready
