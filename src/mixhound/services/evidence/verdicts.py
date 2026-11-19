from collections.abc import Collection
from uuid import UUID

from mixhound.models.domain import (
    DiscoveryResearchResult,
    DiscoveryRunInput,
    DiscoverySettings,
    DiscoveryTrack,
)
from mixhound.services.evidence.links import has_sonic_evidence
from mixhound.services.evidence.screening import Rejection
from mixhound.services.identity import canonical_name, normalize

# Below this similarity estimate a candidate is shown only as a lead.
MIN_VERIFIED_SCORE = 40
# Weight of the measured audio similarity in the ranking score, when one exists.
AUDIO_WEIGHT = 0.3


def _publication_gap(track: DiscoveryTrack, settings: DiscoverySettings) -> str | None:
    """Return why a recording cannot be published as a verified result."""
    if not track.researched:
        return "Not researched"
    if track.year is None and (settings.year_from is not None or settings.year_to is not None):
        return "Year not confirmed"
    return _evidence_gap(track) or _rationale_gap(track)


def _evidence_gap(track: DiscoveryTrack) -> str | None:
    """Return what evidence a verified result lacks."""
    if not any(
        source.reachable is True and source.matches_recording is True and source.claim.strip()
        for source in track.sources
    ):
        return "Verified result lacks recording evidence"
    if not has_sonic_evidence(track):
        return "No sonic evidence"
    return None


def _rationale_gap(track: DiscoveryTrack) -> str | None:
    """Return what the musical rationale of a verified result lacks."""
    if not track.similarities or track.similarity_score is None:
        return "Verified result lacks musical rationale"
    if track.similarity_score < MIN_VERIFIED_SCORE:
        return "Similarity below threshold"
    return None


def rank_score(track: DiscoveryTrack) -> float:
    """The research estimate, blended with measured audio similarity when there is one."""
    score = float(track.similarity_score or 0)
    if track.audio_similarity is not None:
        score = (1 - AUDIO_WEIGHT) * score + AUDIO_WEIGHT * track.audio_similarity
    return score


def _split_verified(
    result: DiscoveryResearchResult,
    settings: DiscoverySettings,
    confirmed: Collection[UUID],
    rejections: list[Rejection],
) -> tuple[list[DiscoveryTrack], list[DiscoveryTrack]]:
    """Mark each candidate verified or a lead, noting demoted ones; return both lists."""
    verified: list[DiscoveryTrack] = []
    leads: list[DiscoveryTrack] = []
    for track in result.results + result.unverified_leads:
        gap = None if track.id in confirmed else "Identity not confirmed by an independent source"
        if gap is None:
            gap = _publication_gap(track, settings)
        if gap is None:
            track.verification = "verified"
            verified.append(track)
            continue
        if track.verification == "verified":
            rejections.append(Rejection(track.artist, track.title, gap, "demoted"))
        track.verification = "unverified"
        leads.append(track)
    return verified, leads


def publish_result(
    result: DiscoveryResearchResult,
    run_input: DiscoveryRunInput,
    confirmed: Collection[UUID],
) -> list[Rejection]:
    """Decide verified results and leads from server-side evidence.

    A candidate is verified when the server confirmed its identity (`confirmed`) and it has
    a similarity estimate of at least MIN_VERIFIED_SCORE with reasons. Results are ranked
    by `rank_score`, limited to `max_per_artist` per artist and to `desired_count`.
    """
    settings = run_input.settings
    rejections: list[Rejection] = []
    verified, leads = _split_verified(result, settings, confirmed, rejections)
    verified.sort(key=rank_score, reverse=True)
    per_artist: dict[str, int] = {}
    published: list[DiscoveryTrack] = []
    for track in verified:
        artist = normalize(canonical_name(track.artist))
        if per_artist.get(artist, 0) >= settings.max_per_artist:
            reason = "More recordings by this artist than allowed"
        elif len(published) >= settings.desired_count:
            reason = "More verified results than requested"
        else:
            per_artist[artist] = per_artist.get(artist, 0) + 1
            published.append(track)
            continue
        rejections.append(Rejection(track.artist, track.title, reason, "demoted"))
        track.verification = "unverified"
        leads.insert(0, track)

    kept_leads: list[DiscoveryTrack] = []
    for track in sorted(leads, key=rank_score, reverse=True):
        artist = normalize(canonical_name(track.artist))
        if per_artist.get(artist, 0) >= settings.max_per_artist:
            rejections.append(
                Rejection(
                    track.artist,
                    track.title,
                    "More recordings by this artist than allowed",
                    "dropped",
                ),
            )
            continue
        per_artist[artist] = per_artist.get(artist, 0) + 1
        kept_leads.append(track)
    result.results, result.unverified_leads = published, kept_leads
    return rejections
