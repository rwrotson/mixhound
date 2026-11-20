from uuid import UUID

from fastapi import APIRouter

from mixhound.api.deps import SessionsDep
from mixhound.models.domain import DiscoveryReference, DiscoverySettings
from mixhound.models.requests import ManualIn, ReferenceIn, SessionIn, SessionNameIn
from mixhound.models.responses import SessionOut, SettingsOut

router = APIRouter(tags=["sessions"])


@router.get("/sessions")
async def list_sessions(sessions: SessionsDep) -> list[SessionOut]:
    """List every session, oldest first."""
    return [SessionOut.model_validate(item) for item in await sessions.list_all()]


@router.post("/sessions", status_code=201)
async def create(payload: SessionIn, sessions: SessionsDep) -> SessionOut:
    """Create a session."""
    return SessionOut.model_validate(
        await sessions.create(
            payload.name, payload.instructions, payload.references, payload.settings
        )
    )


@router.get("/sessions/{session_id}")
async def get(session_id: UUID, sessions: SessionsDep) -> SessionOut:
    """Return one session."""
    return SessionOut.model_validate(await sessions.get(session_id))


@router.patch("/sessions/{session_id}")
async def rename(session_id: UUID, payload: SessionNameIn, sessions: SessionsDep) -> SessionOut:
    """Rename a session."""
    return SessionOut.model_validate(await sessions.rename(session_id, payload.name))


@router.post("/sessions/{session_id}/references", status_code=201)
async def add_reference(
    session_id: UUID, payload: ReferenceIn, sessions: SessionsDep
) -> SessionOut:
    """Add a reference to a session."""
    reference = DiscoveryReference(**payload.model_dump())
    return SessionOut.model_validate(await sessions.add_reference(session_id, reference))


@router.post("/sessions/{session_id}/references/manual", status_code=201)
async def add_manual(session_id: UUID, payload: ManualIn, sessions: SessionsDep) -> SessionOut:
    """Add references typed as `Artist — Track` lines."""
    return SessionOut.model_validate(
        await sessions.add_manual(session_id, payload.lines, payload.polarity)
    )


@router.patch("/sessions/{session_id}/references/{reference_id}")
async def update_reference(
    session_id: UUID, reference_id: UUID, payload: ReferenceIn, sessions: SessionsDep
) -> SessionOut:
    """Replace a session reference."""
    reference = DiscoveryReference(id=reference_id, **payload.model_dump())
    return SessionOut.model_validate(await sessions.update_reference(session_id, reference))


@router.delete("/sessions/{session_id}/references/{reference_id}")
async def delete_reference(
    session_id: UUID, reference_id: UUID, sessions: SessionsDep
) -> SessionOut:
    """Remove a session reference."""
    return SessionOut.model_validate(await sessions.delete_reference(session_id, reference_id))


@router.get("/settings")
async def default_settings(sessions: SessionsDep) -> SettingsOut:
    """Return the settings new sessions start from."""
    return SettingsOut.model_validate(await sessions.default_settings())


@router.patch("/settings")
async def save_default_settings(payload: DiscoverySettings, sessions: SessionsDep) -> SettingsOut:
    """Replace the settings new sessions start from."""
    return SettingsOut.model_validate(await sessions.save_default_settings(payload))
