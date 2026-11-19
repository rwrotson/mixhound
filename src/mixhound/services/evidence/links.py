from urllib.parse import urlsplit

from mixhound.models.domain import DiscoveryTrack
from mixhound.services.identity import canonical_name, name_variants
from mixhound.services.listening import supported_link
from mixhound.services.urls import host_matches

_STREAMING_HOSTS = {
    "open.spotify.com": "Spotify",
    "bandcamp.com": "Bandcamp",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "soundcloud.com": "SoundCloud",
    "music.apple.com": "Apple Music",
    "deezer.com": "Deezer",
    "musicbrainz.org": "MusicBrainz",
    "discogs.com": "Discogs",
}


def _service(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    for domain, name in _STREAMING_HOSTS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return host.removeprefix("www.") or "Other"


def _link_target(url: str, service: str) -> str:
    if supported_link(url, service, "track"):
        return "track"
    path = urlsplit(url).path
    if service == "Bandcamp" and path.startswith("/album/"):
        return "album"
    if service in {"Apple Music", "Deezer"} and ("/track/" in path or "?i=" in url):
        return "track"
    return "other"


def _split_alternate(value: str, alternate: str | None) -> tuple[str, str | None]:
    """Move a transliteration written in brackets beside a name into its own field."""
    variants = name_variants(value)
    if len(variants) > 1:
        return canonical_name(value), alternate or variants[-1]
    return value.strip(), alternate.strip() if alternate and alternate.strip() else None


# Catalogs, similarity services, and players: they identify a recording but don't describe
# how it sounds. Bandcamp pages, reviews, articles, and label text do.
SONIC_EVIDENCE_EXCLUDED_HOSTS = (
    "listenbrainz.org",
    "last.fm",
    "musicbrainz.org",
    "discogs.com",
    "open.spotify.com",
    "spotify.com",
    "deezer.com",
    "music.apple.com",
    "itunes.apple.com",
    "youtube.com",
    "youtu.be",
)
SONIC_EVIDENCE_TARGETS = {"track", "album", "review", "other"}


def has_sonic_evidence(track: DiscoveryTrack) -> bool:
    """Whether the research cites a source that can describe how this recording sounds."""
    for source in track.sources:
        host = (urlsplit(source.url).hostname or "").lower()
        if (
            source.claim.strip()
            and source.target_type in SONIC_EVIDENCE_TARGETS
            and not any(host_matches(host, domain) for domain in SONIC_EVIDENCE_EXCLUDED_HOSTS)
        ):
            return True
    return False
