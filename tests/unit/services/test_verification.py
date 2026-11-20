from uuid import uuid4

import pytest

from mixhound.infrastructure.cache import MemoryCache
from mixhound.models.domain import (
    DiscoveryListeningLink,
    DiscoveryResearchResult,
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySource,
    DiscoveryTrack,
)
from mixhound.services import verification
from mixhound.services.pages import PageInspector
from mixhound.services.ports import CatalogLink, CatalogPage, CatalogRecording, FetchError
from mixhound.services.run_state import RunCancelledError, RunState
from mixhound.services.verification import PAGE_MATCHES, ResultVerifier, likely_result
from tests.fakes import FakeCatalog, FakeFetch, Reply

PAGE = b"<html><body>Low - Words</body></html>"


def state() -> RunState:
    return RunState(DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput()))


def verifier(
    catalog: FakeCatalog | None = None, pages: dict[str, Reply] | None = None
) -> tuple[ResultVerifier, FakeFetch, MemoryCache]:
    fetch, cache = FakeFetch(pages or {}), MemoryCache()
    return ResultVerifier(catalog or FakeCatalog(), PageInspector(fetch), cache), fetch, cache


def test_only_confirmed_bandcamp_album_becomes_listening_link() -> None:
    accepted = "https://artist.bandcamp.com/album/record"
    rejected = [
        ("http://artist.bandcamp.com/album/insecure", True, True),
        ("https://artist.bandcamp.com/track/song", True, True),
        ("https://other.example/album/record", True, True),
        ("https://artist.bandcamp.com/album/unknown", None, True),
        ("https://artist.bandcamp.com/album/wrong", True, False),
    ]
    sources = [
        DiscoverySource(
            url=url,
            source_type="bandcamp",
            target_type="album",
            claim="Album contains recording",
            reachable=reachable,
            matches_recording=matches,
        )
        for url, reachable, matches in [(accepted, True, True), *rejected]
    ]
    track = DiscoveryTrack(artist="Artist", title="Song", sources=sources)
    result = DiscoveryResearchResult(results=[track])

    verification.promote_album_sources(result)
    verification.promote_album_sources(result)

    assert [(link.url, link.target_type) for link in track.listening_links] == [
        (accepted, "album"),
    ]


def test_likely_result_needs_confirmation_reasons_and_score() -> None:
    run = state()
    track = DiscoveryTrack(artist="A", title="B", similarities=["Slow"], similarity_score=70)
    assert not likely_result(run, track)
    run.trust(track, "https://example.org")
    assert likely_result(run, track)
    track.similarity_score = 10
    assert not likely_result(run, track)


async def test_catalog_confirmation_adds_sources_links_and_identifiers() -> None:
    recording = CatalogRecording(
        "Low",
        "Words",
        confirmed_by=("musicbrainz",),
        mbid="r1",
        isrcs=("USX", "USY"),
        year=1994,
        pages=(CatalogPage("musicbrainz", "https://musicbrainz.org/recording/r1"),),
        links=(
            CatalogLink("spotify", "https://open.spotify.com/track/1", "track"),
            CatalogLink("itunes", "https://music.apple.com/us/song/1", "track"),
            CatalogLink("bandcamp", "https://low.bandcamp.com/album/lp", "album"),
        ),
    )
    catalog = FakeCatalog({("Low", "Words"): recording})
    known = DiscoveryListeningLink(
        url="https://open.spotify.com/track/1", service="Spotify", target_type="track"
    )
    track = DiscoveryTrack(
        artist="Low", title="Words", version_qualifier="Live", listening_links=[known]
    )
    run = state()
    checker, _, _ = verifier(catalog)

    await checker.confirm_catalog(run, DiscoveryResearchResult(unverified_leads=[track]))

    assert catalog.calls == [("candidate", "Low", "Words", "Live", None)]
    assert (track.external_recording_id, track.year, track.isrc) == ("mbid:r1", 1994, "USX")
    assert [(s.url, s.reachable, s.matches_recording) for s in track.sources] == [
        ("https://musicbrainz.org/recording/r1", True, True)
    ]
    assert [link.url for link in track.listening_links] == ["https://open.spotify.com/track/1"]
    assert run.trusted_urls[track.id] == {
        "https://musicbrainz.org/recording/r1",
        "https://open.spotify.com/track/1",
    }
    assert run.recordings[track.id] is recording


async def test_unconfirmed_or_unknown_recordings_add_nothing() -> None:
    unconfirmed = CatalogRecording(
        "Low", "Words", pages=(CatalogPage("itunes", "https://music.apple.com/1"),)
    )
    track, ghost = (
        DiscoveryTrack(artist="Low", title="Words"),
        DiscoveryTrack(artist="G", title="H"),
    )
    run = state()
    checker, _, _ = verifier(FakeCatalog({("Low", "Words"): unconfirmed}))

    await checker.confirm_catalog(run, DiscoveryResearchResult(results=[track, ghost]))

    assert track.sources == []
    assert run.confirmed == set()
    assert run.recordings == {track.id: unconfirmed}


async def test_catalog_outage_skips_the_remaining_results() -> None:
    catalog = FakeCatalog(down=True)
    tracks = [DiscoveryTrack(artist=f"A{index}", title="T") for index in range(3)]
    run = state()
    checker, _, _ = verifier(catalog)

    await checker.confirm_catalog(run, DiscoveryResearchResult(results=tracks[:1]))
    await checker.confirm_catalog(run, DiscoveryResearchResult(results=tracks[1:]))

    assert run.catalog_down
    assert len(catalog.calls) == 1


async def test_cancelled_run_stops_verification() -> None:
    run = state()
    run.cancelled = lambda: True
    checker, _, _ = verifier()

    with pytest.raises(RunCancelledError):
        await checker.confirm_catalog(
            run, DiscoveryResearchResult(results=[DiscoveryTrack(artist="A", title="B")])
        )


def source(url: str, target: str) -> DiscoverySource:
    return DiscoverySource.model_validate(
        {"url": url, "source_type": "web", "claim": "c", "target_type": target}
    )


async def test_sources_are_checked_most_specific_first_until_one_matches() -> None:
    track = DiscoveryTrack(
        artist="Low",
        title="Words",
        sources=[
            source("https://example.org/artist", "artist"),
            source("https://example.org/review", "review"),
            source("https://example.org/down", "track"),
            source("https://example.org/other", "album"),
        ],
    )
    run = state()
    checker, fetch, cache = verifier(
        pages={
            "example.org/down": FetchError("unreachable", 503),
            "example.org/other": b"<p>Nothing relevant</p>",
            "example.org/review": PAGE,
        }
    )

    await checker.confirm_sources(run, DiscoveryResearchResult(results=[track]))

    by_url = {item.url: (item.reachable, item.matches_recording) for item in track.sources}
    assert by_url == {
        "https://example.org/artist": (None, None),
        "https://example.org/review": (True, True),
        "https://example.org/down": (None, None),
        "https://example.org/other": (True, False),
    }
    assert track.id in run.confirmed
    assert len(cache.entries) == 2  # Unfetchable pages are not cached.
    assert all(key[0] == PAGE_MATCHES.name for key in cache.entries)
    assert "https://example.org/artist" not in [url for url, _ in fetch.requests]


async def test_sources_of_catalog_confirmed_results_stay_claims() -> None:
    track = DiscoveryTrack(
        artist="Low", title="Words", sources=[source("https://example.org/a", "track")]
    )
    run = state()
    run.trust(track, "https://musicbrainz.org/recording/r1")
    checker, fetch, _ = verifier()

    await checker.confirm_sources(run, DiscoveryResearchResult(results=[track]))

    assert fetch.requests == []


async def test_page_matches_is_cached_and_unknown_without_pages() -> None:
    track = DiscoveryTrack(artist="Low", title="Words")
    checker, fetch, _ = verifier(pages={"example.org/a": PAGE})

    assert await checker.page_matches("https://example.org/a", track) is True
    assert await checker.page_matches("https://example.org/a", track) is True
    assert len(fetch.requests) == 1
    assert await ResultVerifier(FakeCatalog(), None, MemoryCache()).page_matches("u", track) is None


def link(url: str, target: str = "track", service: str = "Spotify") -> DiscoveryListeningLink:
    return DiscoveryListeningLink.model_validate(
        {"url": url, "service": service, "target_type": target}
    )


async def test_links_are_checked_and_dead_ones_withdrawn() -> None:
    trusted = link("https://open.spotify.com/track/trusted")
    matching = link("https://open.spotify.com/track/match")
    gone = link("https://open.spotify.com/track/gone")
    unclear = link("https://open.spotify.com/track/unclear")
    album_gone = link("https://x.bandcamp.com/album/gone", "album", "Bandcamp")
    album_ok = link("https://x.bandcamp.com/album/ok", "album", "Bandcamp")
    track = DiscoveryTrack(
        artist="Low",
        title="Words",
        listening_links=[trusted, matching, gone, unclear, album_gone, album_ok],
    )
    run = state()
    run.trust(track, trusted.url)
    checker, _, _ = verifier(
        pages={
            "open.spotify.com/track/match": PAGE,
            "open.spotify.com/track/gone": FetchError("unreachable", 410),
            "open.spotify.com/track/unclear": FetchError("unreachable", 503),
            "x.bandcamp.com/album/gone": FetchError("unreachable", 404),
            "x.bandcamp.com/album/ok": b"<p>album</p>",
            "open.spotify.com/oembed": FetchError("unreachable", 503),
        }
    )

    await checker.confirm_links(run, DiscoveryResearchResult(results=[track]))

    assert [item.reachable for item in track.listening_links] == [
        True,
        True,
        False,
        None,
        False,
        None,
    ]
    assert all(item.playback_available is None for item in track.listening_links)
    assert matching.url in run.trusted_urls[track.id]
    assert track.preferred_listening_url == trusted.url


async def test_links_stay_unknown_without_a_page_inspector() -> None:
    track = DiscoveryTrack(
        artist="Low",
        title="Words",
        listening_links=[
            link("https://open.spotify.com/track/1"),
            link("https://x.bandcamp.com/album/a", "album", "Bandcamp"),
        ],
    )
    checker = ResultVerifier(FakeCatalog(), None, MemoryCache())

    await checker.confirm_links(state(), DiscoveryResearchResult(results=[track]))

    assert [item.reachable for item in track.listening_links] == [None, None]


async def test_unparsable_track_page_leaves_the_link_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(*_: object, **__: object) -> bool:
        raise ValueError("bad page")

    checker, _, _ = verifier()
    assert checker.pages is not None
    monkeypatch.setattr(checker.pages, "verify_recording", broken)
    track = DiscoveryTrack(
        artist="Low", title="Words", listening_links=[link("https://open.spotify.com/track/1")]
    )

    await checker.confirm_links(state(), DiscoveryResearchResult(results=[track]))

    assert track.listening_links[0].reachable is None


async def test_verify_proposed_adds_only_matching_pages() -> None:
    track = DiscoveryTrack(artist="Low", title="Words")
    good = link("https://open.spotify.com/track/good")
    bad = link("https://open.spotify.com/track/bad")
    run = state()
    checker, _, _ = verifier(
        pages={
            "open.spotify.com/track/good": PAGE,
            "open.spotify.com/track/bad": b"<p>Someone else</p>",
            "open.spotify.com/oembed": FetchError("unreachable", 503),
        }
    )

    await checker.verify_proposed(run, [(track, good), (track, bad), (track, good)])

    assert [item.url for item in track.listening_links] == [good.url]
    assert good.reachable is True
    assert run.trusted_urls[track.id] == {good.url}
