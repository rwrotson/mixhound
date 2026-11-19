import pytest

from mixhound.models.domain import DiscoveryListeningLink, DiscoveryTrack
from mixhound.services.links import choose_preferred, playable_link
from mixhound.services.listening import service_name, supported_link
from mixhound.services.urls import service_for


def link(
    url: str, service: str, target: str = "track", **changes: object
) -> DiscoveryListeningLink:
    return DiscoveryListeningLink.model_validate(
        {"url": url, "service": service, "target_type": target, **changes}
    )


def test_playable_links_are_known_track_pages_or_bandcamp_albums() -> None:
    assert playable_link(link("https://open.spotify.com/track/1", "Spotify"))
    assert playable_link(link("https://x.bandcamp.com/album/y", "Bandcamp", "album"))
    assert not playable_link(link("https://music.apple.com/x", "Apple Music"))
    assert not playable_link(link("https://open.spotify.com/track/1", "Spotify", reachable=False))
    assert not playable_link(
        link("https://open.spotify.com/track/1", "Spotify", playback_available=False)
    )


def test_preferred_link_follows_service_priority_and_stays_while_playable() -> None:
    youtube = link("https://youtu.be/x", "YouTube")
    spotify = link("https://open.spotify.com/track/1", "Spotify")
    track = DiscoveryTrack(artist="A", title="B", listening_links=[youtube, spotify])

    choose_preferred(track)
    assert track.preferred_listening_url == spotify.url

    track.preferred_listening_url = youtube.url
    choose_preferred(track)
    assert track.preferred_listening_url == youtube.url

    track.listening_links = [link("https://music.apple.com/x", "Apple Music")]
    choose_preferred(track)
    assert track.preferred_listening_url is None


@pytest.mark.parametrize(
    ("url", "service", "expected"),
    [
        ("https://open.spotify.com/track/1", "Spotify", True),
        ("https://open.spotify.com/album/1", "Spotify", False),
        ("https://x.bandcamp.com/track/y", "Bandcamp", True),
        ("https://www.youtube.com/watch?v=abc", "YouTube", True),
        ("https://www.youtube.com/channel/abc", "YouTube", False),
        ("https://youtu.be/abc", "YouTube", True),
        ("https://soundcloud.com/user/song", "SoundCloud", True),
        ("https://soundcloud.com/user/sets", "SoundCloud", False),
        ("http://open.spotify.com/track/1", "Spotify", False),
        ("https://example.org/track/1", "Example", False),
    ],
)
def test_supported_track_pages(url: str, service: str, expected: bool) -> None:
    assert supported_link(url, service, "track") is expected


def test_service_names() -> None:
    assert service_name("https://m.soundcloud.com/a/b") == "SoundCloud"
    assert service_name("https://example.org") is None
    assert service_for("https://musicbrainz.org/recording/1") == "MusicBrainz"
    assert service_for("https://example.org") == "Other"
