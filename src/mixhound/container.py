import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import httpx
import structlog

from mixhound.config import Settings
from mixhound.infrastructure.cache import DiskCache, MemoryCache
from mixhound.infrastructure.catalogs.deadwax import open_catalog
from mixhound.infrastructure.ffmpeg import FfmpegDecoder
from mixhound.infrastructure.http import FetchPolicy, PageFetcher, open_http_client
from mixhound.infrastructure.llm.factory import ModelFactory
from mixhound.infrastructure.logs import setup_logging
from mixhound.infrastructure.queue import AsyncioRunQueue
from mixhound.infrastructure.search.brave import BraveSearch
from mixhound.infrastructure.skills import load_catalog
from mixhound.infrastructure.storage.files import FileDiagnostics, FileRuns, FileSessions, FileStore
from mixhound.infrastructure.storage.memory import MemoryDiagnostics, MemoryRuns, MemorySessions
from mixhound.memory.repository import DiagnosticsRepository, RunRepository, SessionRepository
from mixhound.services.ports import (
    AudioDecoder,
    Cache,
    PageFetch,
    RunQueue,
    TrackCatalog,
    WebSearch,
)
from mixhound.services.skills import SkillCatalog


@dataclass(frozen=True)
class Repositories:
    """Where sessions, runs and diagnostics are stored."""

    sessions: SessionRepository
    runs: RunRepository
    diagnostics: DiagnosticsRepository


@dataclass(frozen=True)
class Container:
    """Resources owned by one application process."""

    settings: Settings
    skills: SkillCatalog
    repositories: Repositories
    catalog: TrackCatalog
    pages: PageFetch
    cache: Cache
    decoder: AudioDecoder
    search: WebSearch | None
    queue: RunQueue
    models: ModelFactory


def open_repositories(settings: Settings) -> Repositories:
    """Create the repositories of the configured storage backend."""
    if settings.storage.backend == "memory":
        return Repositories(MemorySessions(), MemoryRuns(), MemoryDiagnostics())
    store = FileStore(settings.data_dir / "store")
    return Repositories(FileSessions(store), FileRuns(store), FileDiagnostics(store))


@asynccontextmanager
async def open_container(settings: Settings) -> AsyncIterator[Container]:
    """Create application resources and release them on exit."""
    setup_logging(settings.log)
    log = structlog.get_logger("mixhound")
    async with AsyncExitStack() as stack:
        skills = load_catalog(settings.skills_dir)
        client = await stack.enter_async_context(open_http_client(settings.http))
        catalog = await stack.enter_async_context(open_catalog(settings.deadwax, settings.data_dir))
        search: WebSearch | None = None
        if settings.search.brave_api_key is not None:
            search_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=settings.search.timeout)
            )
            search = BraveSearch(
                search_client,
                settings.search.brave_api_key.get_secret_value(),
                results=settings.search.results,
            )
        cache: Cache = (
            DiskCache(settings.data_dir / "cache", enabled=settings.cache.enabled)
            if settings.storage.backend == "files"
            else MemoryCache()
        )
        queue = AsyncioRunQueue(max_pending=settings.queue.max_pending)
        stack.push_async_callback(queue.close)
        models = ModelFactory(settings.llm, os.environ)
        stack.push_async_callback(models.aclose)
        container = Container(
            settings=settings,
            skills=skills,
            repositories=open_repositories(settings),
            catalog=catalog,
            pages=PageFetcher(client, FetchPolicy.from_settings(settings.http)),
            cache=cache,
            decoder=FfmpegDecoder(),
            search=search,
            queue=queue,
            models=models,
        )
        log.info("application_started", skills=len(skills.skills), web_search=search is not None)
        try:
            yield container
        finally:
            log.info("application_stopped")
