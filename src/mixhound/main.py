from fastapi import FastAPI

from mixhound.api.app import build_api
from mixhound.config import Settings, load_settings
from mixhound.container import open_container


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application; run with `uvicorn mixhound.main:create_app --factory`."""
    config = settings if settings is not None else load_settings()
    return build_api(config.api, lambda: open_container(config))


def serve() -> None:
    """Run the HTTP server."""
    import uvicorn  # noqa: PLC0415 - imported only by the console script

    uvicorn.run(create_app, factory=True, host="127.0.0.1", port=8000, access_log=False)
