import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import httpx
import structlog

from mixhound.agents.runner import PydanticResearchAgents
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
from mixhound.infrastructure.storage.owner import claim_owner
from mixhound.memory.repository import DiagnosticsRepository, RunRepository, SessionRepository
from mixhound.models.responses import ProvidersOut
from mixhound.services.audio import AudioAnalyzer, AudioMatcher
from mixhound.services.pages import PageInspector
from mixhound.services.pipeline import ResearchPipeline
from mixhound.services.ports import (
    AudioDecoder,
    Cache,
    PageFetch,
    RunQueue,
    TrackCatalog,
    WebSearch,
)
from mixhound.services.research import ResearchSteps
from mixhound.services.run_events import RunEvents
from mixhound.services.runs import RunService
from mixhound.services.seed import Seeder
from mixhound.services.sessions import SessionService
from mixhound.services.skills import SkillCatalog
from mixhound.services.verification import ResultVerifier
from mixhound.tools.deps import Toolbox


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
    pipeline: ResearchPipeline
    sessions: SessionService
    runs: RunService
    providers: ProvidersOut


def open_repositories(settings: Settings) -> Repositories:
    """Create the repositories of the configured storage backend."""
    if settings.storage.backend == "memory":
        return Repositories(MemorySessions(), MemoryRuns(), MemoryDiagnostics())
    store = FileStore(settings.data_dir / "store")
    return Repositories(FileSessions(store), FileRuns(store), FileDiagnostics(store))


def providers_status(settings: Settings, *, search: WebSearch | None, audio: bool) -> ProvidersOut:
    """What the process is configured to use, without revealing any key."""
    deadwax = settings.deadwax
    return ProvidersOut(
        model=settings.llm.model,
        helper_model=settings.llm.helper_model,
        fallback_model=settings.llm.fallback_model,
        web_search=search is not None,
        audio=audio,
        catalog_keys={
            "lastfm": deadwax.lastfm_api_key is not None,
            "discogs": deadwax.discogs_token is not None,
            "youtube": deadwax.youtube_api_key is not None,
            "spotify": deadwax.spotify_client_id is not None
            and deadwax.spotify_client_secret is not None,
        },
    )


async def _prune(cache: Cache) -> None:
    """Drop expired entries of a disk cache; a failure only leaves them for next time."""
    if isinstance(cache, DiskCache):
        try:
            await cache.prune()
        except OSError:
            structlog.get_logger("mixhound").exception("cache_prune_failed")


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
        owner = settings.storage.backend == "memory" or stack.enter_context(
            claim_owner(settings.data_dir / "store" / "owner.lock")
        )
        queue = AsyncioRunQueue(max_pending=settings.queue.max_pending)
        models = ModelFactory(settings.llm, os.environ)
        stack.push_async_callback(models.aclose)
        pages = PageFetcher(client, FetchPolicy.from_settings(settings.http))
        decoder = FfmpegDecoder()
        verifier = ResultVerifier(catalog, PageInspector(pages), cache)
        audio = AudioAnalyzer(pages, decoder, cache) if decoder.available() else None
        agents = PydanticResearchAgents(models, Toolbox(pages, catalog, search))
        repositories = open_repositories(settings)
        pipeline = ResearchPipeline(
            Seeder(catalog, audio),
            verifier,
            AudioMatcher(audio),
            ResearchSteps(agents, verifier, skills),
        )
        runs = RunService(
            repositories.sessions,
            repositories.runs,
            repositories.diagnostics,
            queue,
            pipeline,
            events=RunEvents(),
            owner=owner,
        )
        stack.push_async_callback(runs.shutdown)
        if owner:
            await _prune(cache)
            await runs.reconcile()
        container = Container(
            settings=settings,
            skills=skills,
            repositories=repositories,
            catalog=catalog,
            pages=pages,
            cache=cache,
            decoder=decoder,
            search=search,
            queue=queue,
            models=models,
            pipeline=pipeline,
            sessions=SessionService(repositories.sessions, repositories.runs, PageInspector(pages)),
            runs=runs,
            providers=providers_status(settings, search=search, audio=audio is not None),
        )
        log.info(
            "application_started",
            skills=len(skills.skills),
            web_search=search is not None,
            owner=owner,
        )
        try:
            yield container
        finally:
            log.info("application_stopped")
