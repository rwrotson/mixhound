from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import structlog

from mixhound.config import Settings
from mixhound.infrastructure.logs import setup_logging


@dataclass(frozen=True)
class Container:
    """Resources owned by one application process."""

    settings: Settings


@asynccontextmanager
async def open_container(settings: Settings) -> AsyncIterator[Container]:
    """Create application resources and release them on exit."""
    setup_logging(settings.log)
    log = structlog.get_logger("mixhound")
    async with AsyncExitStack():
        log.info("application_started")
        try:
            yield Container(settings=settings)
        finally:
            log.info("application_stopped")
