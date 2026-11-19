from dataclasses import dataclass, field
from urllib.parse import urlsplit

from mixhound.models.domain import (
    DiscoveryResearchResult,
    DiscoveryRunInput,
    DiscoverySettings,
    DiscoveryTrack,
)
from mixhound.services.errors import MixhoundError
from mixhound.services.exclusions import NO_EXCLUSIONS, Exclusions, metadata_key
from mixhound.services.identity import normalize, recording_key


class ResearchEvidenceError(MixhoundError, ValueError):
    """A research answer violates the publication contract."""


def _http_url(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme in {"http", "https"} and bool(parts.hostname) and not parts.username


def _key(track: DiscoveryTrack) -> str:
    return recording_key(
        track.artist,
        track.title,
        qualifier=track.version_qualifier,
        external_recording_id=track.external_recording_id,
    )


@dataclass
class Rejection:
    """A candidate the screening dropped, demoted or corrected, and why."""

    artist: str
    title: str
    reason: str
    action: str  # "dropped", "demoted" or "corrected"

    def as_dict(self) -> dict[str, str]:
        """Return the rejection for the run diagnostic."""
        return {
            "artist": self.artist,
            "title": self.title,
            "reason": self.reason,
            "action": self.action,
        }


@dataclass
class _Screen:
    run_input: DiscoveryRunInput
    exclusions: Exclusions
    strict: bool
    rejections: list[Rejection] = field(default_factory=list)

    def note(self, track: DiscoveryTrack, reason: str, action: str) -> None:
        if self.strict:
            raise ResearchEvidenceError(reason)
        self.rejections.append(Rejection(track.artist, track.title, reason, action))


def _correct(track: DiscoveryTrack, screen: _Screen) -> None:
    """Repair claims that can be withdrawn without discarding the recording."""
    _correct_claims(track, screen)
    _correct_urls(track, screen)
    _correct_playback(track, screen)
    if track.preferred_listening_url and track.preferred_listening_url not in {
        link.url for link in track.listening_links
    }:
        screen.note(track, "Preferred listening URL is absent", "corrected")
        track.preferred_listening_url = None


def _correct_claims(track: DiscoveryTrack, screen: _Screen) -> None:
    """Withdraw an unsupported audio analysis claim and unknown closest references."""
    reference_ids = {ref.id for ref in screen.run_input.references}
    if track.audio_analyzed and not track.audio_analysis_evidence:
        screen.note(track, "Audio analysis claim lacks evidence", "corrected")
        track.audio_analyzed = False
    if any(ref_id not in reference_ids for ref_id in track.closest_reference_ids):
        screen.note(track, "Unknown closest reference", "corrected")
        track.closest_reference_ids = [
            ref_id for ref_id in track.closest_reference_ids if ref_id in reference_ids
        ]


def _correct_urls(track: DiscoveryTrack, screen: _Screen) -> None:
    """Drop incomplete discovery paths and entries with invalid URLs."""
    for path in list(track.discovery_paths):
        if not path.strategy.strip() or not path.explanation.strip():
            screen.note(track, "Incomplete discovery path", "corrected")
            track.discovery_paths.remove(path)
        elif path.source_url and not _http_url(path.source_url):
            screen.note(track, "Invalid discovery URL", "corrected")
            track.discovery_paths.remove(path)
    for source in list(track.sources):
        if not _http_url(source.url):
            screen.note(track, "Invalid source URL", "corrected")
            track.sources.remove(source)
    for link in list(track.listening_links):
        if not _http_url(link.url):
            screen.note(track, "Invalid listening URL", "corrected")
            track.listening_links.remove(link)


def _correct_playback(track: DiscoveryTrack, screen: _Screen) -> None:
    """Withdraw playback claims for unreachable links and unconfirmed album pages."""
    for link in track.listening_links:
        if not link.playback_available:
            continue
        album_recording_confirmed = link.target_type == "album" and any(
            source.url == link.url
            and source.target_type == "album"
            and source.reachable is True
            and source.matches_recording is True
            and source.claim.strip()
            for source in track.sources
        )
        if link.reachable is not True or (
            link.target_type != "track" and not album_recording_confirmed
        ):
            screen.note(track, "Unsupported playback claim", "corrected")
            link.playback_available = None


def _exclusion(track: DiscoveryTrack, screen: _Screen, seen: set[str]) -> str | None:
    """Return why a recording must not be shown at all."""
    artist = normalize(track.artist)
    key, meta = _key(track), metadata_key(track)
    if not artist or not normalize(track.title) or key in seen or meta in seen:
        return "Missing recording identity or duplicate"
    seen.update((key, meta))
    if "cover" in normalize(track.version_qualifier or "") and not (
        track.original_artist and track.original_title
    ):
        cover_gap: str | None = "Cover lacks original-work attribution"
    else:
        cover_gap = None
    return (
        _settings_exclusion(track, artist, screen.run_input.settings)
        or _repeat_exclusion(artist, {key, meta}, screen)
        or cover_gap
    )


def _settings_exclusion(
    track: DiscoveryTrack,
    artist: str,
    settings: DiscoverySettings,
) -> str | None:
    """Return why the run settings exclude a recording."""
    if artist in {normalize(value) for value in settings.excluded_artists}:
        return "Excluded artist"
    # An undated recording stays; `_publication_gap` keeps it a lead.
    if track.year is not None and (
        (settings.year_from is not None and track.year < settings.year_from)
        or (settings.year_to is not None and track.year > settings.year_to)
    ):
        return "Outside requested year range"
    return None


def _repeat_exclusion(artist: str, keys: set[str], screen: _Screen) -> str | None:
    """Return why a recording repeats a reference or an earlier proposal."""
    settings = screen.run_input.settings
    references = screen.run_input.references
    reference_keys = {
        recording_key(ref.artist, ref.title, qualifier=ref.version_qualifier) for ref in references
    }
    if keys & reference_keys:
        return "Reference recording"
    if keys & screen.exclusions.prior_keys:
        return "Previously proposed recording"
    if settings.exclude_reference_artists and artist in {
        normalize(ref.artist) for ref in references
    }:
        return "Reference artist"
    if settings.exclude_previously_proposed_artists and artist in screen.exclusions.prior_artists:
        return "Previously proposed artist"
    return None


def _screen(result: DiscoveryResearchResult, screen: _Screen) -> DiscoveryResearchResult:
    """Drop excluded candidates and correct withdrawable claims, keeping lists as they are."""
    seen: set[str] = set()
    for name in ("results", "unverified_leads"):
        kept: list[DiscoveryTrack] = []
        for track in getattr(result, name):
            reason = _exclusion(track, screen, seen)
            if reason:
                screen.note(track, reason, "dropped")
                continue
            _correct(track, screen)
            kept.append(track)
        setattr(result, name, kept)
    return result


def trim_to_target(result: DiscoveryResearchResult, target: int) -> list[Rejection]:
    """Keep the `target` most promising candidates before the costly server stages.

    Researched candidates come first, then the higher research estimate; the research
    returns everything plausible and the server decides how many are worth checking.
    """
    tracks = result.results + result.unverified_leads
    if len(tracks) <= target:
        return []
    ranked = sorted(
        tracks,
        key=lambda track: (track.researched, track.similarity_score or 0),
        reverse=True,
    )
    kept = {id(track) for track in ranked[:target]}
    result.results = [track for track in result.results if id(track) in kept]
    result.unverified_leads = [track for track in result.unverified_leads if id(track) in kept]
    return [
        Rejection(track.artist, track.title, "Below research target", "dropped")
        for track in ranked[target:]
    ]


def validate_research_result(
    result: DiscoveryResearchResult,
    run_input: DiscoveryRunInput,
    exclusions: Exclusions = NO_EXCLUSIONS,
) -> DiscoveryResearchResult:
    """Raise on the first contract violation; used where an answer must already be clean."""
    from mixhound.services.evidence.verdicts import _publication_gap  # noqa: PLC0415

    screen = _Screen(run_input, exclusions, strict=True)
    _screen(result, screen)
    for track in result.results:
        gap = (
            _publication_gap(track, run_input.settings)
            if track.verification == "verified"
            else None
        )
        if gap:
            raise ResearchEvidenceError(gap)
    if len(result.results) > run_input.settings.desired_count:
        raise ResearchEvidenceError("More verified results than requested")
    return result


def screen_research_result(
    result: DiscoveryResearchResult,
    run_input: DiscoveryRunInput,
    exclusions: Exclusions = NO_EXCLUSIONS,
) -> list[Rejection]:
    """Drop or correct individual candidates instead of failing the whole run."""
    screen = _Screen(run_input, exclusions, strict=False)
    _screen(result, screen)
    return screen.rejections
