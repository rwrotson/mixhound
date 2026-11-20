from fastapi import APIRouter
from starlette.exceptions import HTTPException

from mixhound.api.deps import RunsDep

router = APIRouter(tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    """Report that the HTTP process is alive."""
    return {"status": "ok"}


@router.get("/ready", responses={503: {"description": "Runs are not accepted"}})
async def ready(runs: RunsDep) -> dict[str, str]:
    """Report that the application can accept runs."""
    if not runs.ready:
        raise HTTPException(503, "Runs are not accepted: shutting down or not the owner")
    return {"status": "ok"}
