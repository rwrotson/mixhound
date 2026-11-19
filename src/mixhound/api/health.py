from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    """Report that the HTTP process is alive."""
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict[str, str]:
    """Report that the application can accept runs."""
    return {"status": "ok"}
