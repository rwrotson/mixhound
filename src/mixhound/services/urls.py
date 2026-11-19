from urllib.parse import urlsplit

SERVICES = {
    "bandcamp.com": "Bandcamp",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "soundcloud.com": "SoundCloud",
    "musicbrainz.org": "MusicBrainz",
    "spotify.com": "Spotify",
}


def host_matches(host: str, domain: str) -> bool:
    """Return whether a host is the domain or one of its subdomains."""
    return host == domain or host.endswith("." + domain)


def service_for(url: str) -> str:
    """Return the music service a URL belongs to, or `Other`."""
    host = (urlsplit(url).hostname or "").lower()
    return next((name for domain, name in SERVICES.items() if host_matches(host, domain)), "Other")
