from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent

from mixhound.api.deps import RunsDep, SessionsDep
from mixhound.models.requests import FeedbackIn, RunRequest
from mixhound.models.responses import FeedbackStateOut, RunOut
from mixhound.models.state import RunProgress, RunStatusEvent

router = APIRouter(tags=["runs"])

type RunEventStream = AsyncIterator[RunStatusEvent | RunProgress]


async def run_events(run_id: UUID, runs: RunsDep) -> RunEventStream:
    """Open a run's event stream; a missing run fails before the response starts."""
    return await runs.follow(run_id)


@router.post("/sessions/{session_id}/runs", status_code=202)
async def create(session_id: UUID, payload: RunRequest, runs: RunsDep) -> RunOut:
    """Queue a research run for a session."""
    return RunOut.model_validate(
        await runs.create(session_id, payload.instructions, payload.references, payload.settings)
    )


@router.get("/runs/{run_id}")
async def get(run_id: UUID, runs: RunsDep) -> RunOut:
    """Return one run."""
    return RunOut.model_validate(await runs.get(run_id))


@router.get("/runs/{run_id}/progress")
async def progress(run_id: UUID, runs: RunsDep) -> RunProgress | None:
    """The active run's stage and live research counts; null unless it runs in this process."""
    await runs.get(run_id)
    return runs.progress(run_id)


@router.get("/runs/{run_id}/events", response_class=EventSourceResponse)
async def events(
    stream: Annotated[RunEventStream, Depends(run_events)],
) -> AsyncIterator[ServerSentEvent]:
    """Stream `status` events, and `progress` events while the run runs, until it finishes."""
    async for event in stream:
        kind = "status" if isinstance(event, RunStatusEvent) else "progress"
        yield ServerSentEvent(event=kind, data=event)


@router.post("/runs/{run_id}/cancel")
async def cancel(run_id: UUID, runs: RunsDep) -> RunOut:
    """Cancel a queued or running run."""
    return RunOut.model_validate(await runs.cancel(run_id))


@router.post("/runs/{run_id}/tracks/{track_id}/feedback")
async def feedback(
    run_id: UUID, track_id: UUID, payload: FeedbackIn, sessions: SessionsDep
) -> FeedbackStateOut:
    """Record feedback on a result and return its new feedback state."""
    return FeedbackStateOut.model_validate(
        await sessions.record_feedback(run_id, track_id, payload.model_dump())
    )
