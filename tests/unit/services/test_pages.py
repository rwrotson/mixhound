import json

import pytest

from mixhound.services.pages import PageInspector, PageRecording, _parse
from mixhound.services.ports import FetchError
from tests.fakes import FakeFetch, Reply


def inspector(replies: dict[str, Reply]) -> PageInspector:
    return PageInspector(FakeFetch(replies))


async def test_independent_page_identity_checks_version() -> None:
    pages = inspector(
        {"example.org/track": b"<html><body>The Coral - In the Morning (Live)</body></html>"}
    )

    url = "https://example.org/track"
    assert await pages.verify_recording(url, "The Coral", "In the Morning", "Live")
    assert not await pages.verify_recording(url, "The Coral", "In the Morning", "Remix")


async def test_verify_uses_oembed_for_client_rendered_pages() -> None:
    pages = inspector(
        {
            "www.youtube.com/oembed": {
                "title": "Low - Words (Official Audio)",
                "author_name": "Low - Topic",
            },
            "www.youtube.com/watch": b"<html><body>JavaScript required</body></html>",
        }
    )

    assert await pages.verify_recording("https://www.youtube.com/watch?v=x", "Low", "Words")
    assert not await pages.verify_recording("https://www.youtube.com/watch?v=x", "Low", "Other")


async def test_oembed_answer_survives_an_unreachable_page() -> None:
    pages = inspector(
        {
            "soundcloud.com/oembed": {"title": "Words", "author_name": "Low"},
            "soundcloud.com/low/words": FetchError("unreachable", 403),
        }
    )

    assert await pages.verify_recording("https://soundcloud.com/low/words", "Low", "Words")


@pytest.mark.parametrize("oembed", [FetchError("unreachable", 500), ["not", "an", "object"]])
async def test_failed_oembed_falls_back_to_the_page(oembed: Reply) -> None:
    pages = inspector(
        {
            "open.spotify.com/oembed": oembed,
            "open.spotify.com/track/1": b"<title>Words - song by Low</title>",
        }
    )

    assert await pages.verify_recording("https://open.spotify.com/track/1", "Low", "Words")


async def test_unreachable_page_without_oembed_raises() -> None:
    with pytest.raises(FetchError, match="unreachable"):
        await inspector({}).verify_recording("https://example.org/none", "Low", "Words")


async def test_verify_reads_open_graph_and_respects_word_boundaries() -> None:
    pages = inspector(
        {
            "example.org/og": b'<meta property="og:title" content="Words">'
            b'<meta property="og:description" content="Low \xc2\xb7 Album \xc2\xb7 Song">',
            "example.org/text": b"<body>Below the surface: Words</body>",
        }
    )

    assert await pages.verify_recording("https://example.org/og", "Low", "Words")
    assert not await pages.verify_recording("https://example.org/text", "Low", "Words")


async def test_hidden_text_and_link_metadata_are_ignored() -> None:
    pages = inspector(
        {
            "example.org/page": b'<meta name="music:musician" content="https://x.example/low">'
            b"<style>Low Words</style><script>var a = 'Low Words';</script><p>Nothing here</p>"
        }
    )

    assert not await pages.verify_recording("https://example.org/page", "Low", "Words")


def test_json_ld_type_lists_and_item_lists_are_read() -> None:
    album = {
        "@context": "https://schema.org",
        "@type": ["MusicAlbum", "Product"],
        "name": "LP",
        "track": {
            "@type": "ItemList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "item": {
                        "@type": ["MusicRecording"],
                        "name": "Inside Out",
                        "byArtist": {"name": "Duster"},
                    },
                },
            ],
        },
    }
    odd = {"@type": [["nested"], 3], "name": "Not a recording"}
    html = (
        "<html><head>"
        f'<script type="application/ld+json">{json.dumps(album)}</script>'
        f'<script type="application/ld+json">{json.dumps(odd)}</script>'
        '<script type="application/ld+json">{broken</script>'
        "</head><body></body></html>"
    ).encode()
    assert _parse(html).recordings() == [("Duster", "Inside Out")]


@pytest.mark.parametrize(
    ("reply", "gone"),
    [
        (FetchError("unreachable", 404), True),
        (FetchError("unreachable", 410), True),
        (FetchError("unreachable", 503), False),
        (OSError("network down"), False),
        (b"<html></html>", False),
    ],
)
async def test_page_gone_only_for_definitive_absence(reply: Reply, gone: bool) -> None:
    assert await inspector({"example.com/song": reply}).gone("https://example.com/song") is gone


@pytest.mark.parametrize(
    ("url", "body", "expected"),
    [
        (
            "https://low.bandcamp.com/track/words",
            b'<meta property="og:title" content="Words, by Low by Low">',
            PageRecording(
                "https://low.bandcamp.com/track/words", "Bandcamp", "Low", "Words, by Low"
            ),
        ),
        (
            "https://example.org/a",
            b'<meta property="music:song" content="Words">'
            b'<meta property="music:musician" content="https://x.example/low">',
            PageRecording("https://example.org/a", "Other", None, "Words"),
        ),
        (
            "https://example.org/b",
            b'<title>Page</title><script type="application/ld+json">'
            b'{"@type": "MusicRecording", "name": "Words", "byArtist": [{"name": "Low"}]}'
            b"</script>",
            PageRecording("https://example.org/b", "Other", "Low", "Words"),
        ),
        ("https://example.org/c", b"<p>plain</p>", PageRecording("https://example.org/c", "Other")),
    ],
)
async def test_resolve_reads_the_recording_a_page_describes(
    url: str, body: bytes, expected: PageRecording
) -> None:
    pages = inspector({url.removeprefix("https://"): body})

    assert await pages.resolve(url) == expected
