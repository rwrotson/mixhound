from dataclasses import dataclass

from mixhound.memory.repository import RunRepository, SessionRepository
from mixhound.models.domain import DiscoveryRun, DiscoveryTrack


@dataclass(frozen=True)
class RunHistory:
    """Earlier runs and every track they proposed."""

    runs: list[DiscoveryRun]
    tracks: list[DiscoveryTrack]


async def run_history(
    sessions: SessionRepository,
    runs: RunRepository,
    run: DiscoveryRun,
) -> RunHistory:
    """Every earlier run of every session and the tracks they proposed."""
    previous: list[DiscoveryRun] = []
    for session in await sessions.list():
        for run_id in session.run_ids:
            if run_id == run.id:
                continue
            found = await runs.get(run_id)
            if found is not None:
                previous.append(found)
    return RunHistory(
        runs=previous,
        tracks=[track for old in previous for track in old.results + old.unverified_leads],
    )
