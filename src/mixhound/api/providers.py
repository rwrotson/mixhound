from fastapi import APIRouter

from mixhound.api.deps import ServicesDep, SessionsDep
from mixhound.models.requests import URLIn
from mixhound.models.responses import ProvidersOut, ResolvedUrlOut

router = APIRouter(tags=["providers"])


@router.get("/providers")
async def providers(services: ServicesDep) -> ProvidersOut:
    """Return the configured models and which optional services are available."""
    return services.providers


@router.post("/resolve-url")
async def resolve_url(payload: URLIn, sessions: SessionsDep) -> ResolvedUrlOut:
    """Read the recording a public URL describes."""
    return ResolvedUrlOut.model_validate(await sessions.resolve_url(payload.url))
