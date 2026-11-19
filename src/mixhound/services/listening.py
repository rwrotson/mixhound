import json
from urllib.parse import urlsplit

from mixhound.models.domain import DiscoveryListeningLink, DiscoveryResearchResult, DiscoveryTrack
from mixhound.models.research import LinkSearchOutput
from mixhound.services.identity import WantedRecording, identity_matches
from mixhound.services.urls import host_matches

MIN_LISTENING_SERVICES = 2

# A SoundCloud track path is /<user>/<track>.
_SOUNDCLOUD_TRACK_PARTS = 2

_SOUNDCLOUD_NON_TRACK = {
    "sets",
    "albums",
    "tracks",
    "reposts",
    "likes",
    "followers",
    "following",
    "popular-tracks",
    "comments",
    "discover",
    "search",
}


def _track_path(service: str, host: str, path: str, query: str) -> bool:
    parts = [part for part in path.split("/") if part]
    if service == "spotify":
        return "track" in parts
    if service == "bandcamp":
        return parts[-2:-1] == ["track"]
    if service == "youtube":
        return (host == "youtu.be" and len(parts) == 1) or (parts == ["watch"] and "v=" in query)
    if service == "soundcloud":
        return len(parts) == _SOUNDCLOUD_TRACK_PARTS and parts[1] not in _SOUNDCLOUD_NON_TRACK
    return False


SUPPORTED_SERVICES = {
    "spotify": ("open.spotify.com",),
    "youtube": ("youtube.com", "youtu.be"),
    "soundcloud": ("soundcloud.com",),
    "bandcamp": ("bandcamp.com",),
}


def supported_link(url: str, service: str, target_type: str) -> bool:
    """Whether a link is a specific track page on a service Mixtable can play."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if target_type != "track" or parts.scheme != "https":
        return False
    for name, domains in SUPPORTED_SERVICES.items():
        if name in service.casefold() and any(host_matches(host, domain) for domain in domains):
            return _track_path(name, host, parts.path, parts.query)
    return False


def missing_links(result: DiscoveryResearchResult) -> list[DiscoveryTrack]:
    """Results and unverified leads with fewer than two playable services."""
    return [
        track
        for track in result.results + result.unverified_leads
        if len(
            {
                link.service.casefold()
                for link in track.listening_links
                if link.reachable is True
                and link.playback_available is not False
                and supported_link(link.url, link.service, link.target_type)
            },
        )
        < MIN_LISTENING_SERVICES
    ]


def followup_prompt(tracks: list[DiscoveryTrack]) -> str:
    """Build the follow-up prompt asking for track pages of the given results."""
    identities = [
        {
            "artist": track.artist,
            "title": track.title,
            "artist_alt": track.artist_alt,
            "title_alt": track.title_alt,
            "album": track.album,
            "year": track.year,
            "version_qualifier": track.version_qualifier,
            "existing_services": sorted(
                {
                    link.service
                    for link in track.listening_links
                    if link.reachable is True and link.playback_available is not False
                },
            ),
        }
        for track in tracks
    ]
    return "Find track pages for these recordings.\n" + json.dumps(identities, ensure_ascii=False)


def service_name(url: str) -> str | None:
    """The playable service a URL belongs to, if any."""
    host = (urlsplit(url).hostname or "").lower()
    for name, domains in SUPPORTED_SERVICES.items():
        if any(host_matches(host, domain) for domain in domains):
            return {
                "spotify": "Spotify",
                "youtube": "YouTube",
                "soundcloud": "SoundCloud",
                "bandcamp": "Bandcamp",
            }[name]
    return None


def proposed_links(
    tracks: list[DiscoveryTrack],
    found: LinkSearchOutput,
) -> list[tuple[DiscoveryTrack, DiscoveryListeningLink]]:
    """Track-page URLs the follow-up proposed, paired with their recording; still unverified."""
    proposals: list[tuple[DiscoveryTrack, DiscoveryListeningLink]] = []
    for item in found.links:
        service = service_name(item.url)
        if service is None or not supported_link(item.url, service, "track"):
            continue
        target = next(
            (
                track
                for track in tracks
                if identity_matches(
                    item.artist,
                    item.title,
                    WantedRecording(
                        track.artist,
                        track.title,
                        track.version_qualifier,
                        (track.artist_alt, track.title_alt),
                    ),
                )
            ),
            None,
        )
        if target is None or any(link.url == item.url for link in target.listening_links):
            continue
        proposals.append(
            (target, DiscoveryListeningLink(url=item.url, service=service, target_type="track")),
        )
    return proposals
