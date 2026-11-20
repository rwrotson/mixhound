from dataclasses import dataclass

from mixhound.memory.repository import RunRepository, SessionRepository
from mixhound.models.domain import DiscoveryRun, DiscoverySession, DiscoveryTrack


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


async def feedback_snapshot(
    runs: RunRepository, session: DiscoverySession
) -> list[dict[str, object]]:
    """Marks, notes and exclusions on earlier results of the session."""
    snapshot: list[dict[str, object]] = []
    for run_id in session.run_ids:
        prior = await runs.get(run_id)
        if prior is None:
            continue
        for track in prior.results + prior.unverified_leads:
            state = session.feedback_for(track.id)
            marked = state.polarity != "neutral"
            if marked or state.text or state.excluded or state.avoid_similar or state.tags:
                snapshot.append(
                    {
                        "artist": track.artist,
                        "title": track.title,
                        "polarity": state.polarity,
                        "weight": state.weight if marked else None,
                        "reason": state.text,
                        "excluded": state.excluded,
                        "avoid_similar": state.avoid_similar,
                        "tags": state.tags,
                        "audio": track.audio_features if marked else None,
                    },
                )
    return snapshot
