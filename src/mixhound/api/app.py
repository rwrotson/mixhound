from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from importlib.metadata import version
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from mixhound.api.health import router as health_router
from mixhound.api.problems import register_error_handlers
from mixhound.config import ApiSettings

DISTRIBUTION = "mixhound"

Lifespan = Callable[[], AbstractAsyncContextManager[Any]]


def _operation_id(route: APIRoute) -> str:
    """Generate stable OpenAPI operation IDs such as `runs-create`."""
    return f"{route.tags[0]}-{route.name}" if route.tags else route.name


def build_api(settings: ApiSettings, resources: Lifespan) -> FastAPI:
    """Build the HTTP application around a lifespan that owns the application resources."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[dict[str, Any]]:
        async with resources() as state:
            yield {"container": state}

    application = FastAPI(
        title="mixhound",
        version=version(DISTRIBUTION),
        lifespan=lifespan,
        generate_unique_id_function=_operation_id,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    register_error_handlers(application)
    application.include_router(health_router)
    return application
