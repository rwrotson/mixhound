from pathlib import Path

from pydantic import SecretStr

from mixhound.config import CacheSettings, SearchSettings, Settings, StorageSettings
from mixhound.container import open_container
from mixhound.infrastructure.cache import DiskCache, MemoryCache
from mixhound.infrastructure.search.brave import BraveSearch
from mixhound.infrastructure.storage.files import FileRuns
from mixhound.infrastructure.storage.memory import MemoryRuns
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
