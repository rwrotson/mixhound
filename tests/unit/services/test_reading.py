import json

from mixhound.services.reading import page_text

PAGE = b"""<html><head><title> Words  | Low </title>
<meta name="description" content="Single from the album.">
<script type="application/ld+json">{"@type": "MusicRecording", "name": "Words",
 "byArtist": {"name": "Low"}, "keywords": ["slowcore", "dream pop"], "duration": "PT5M"}</script>
<style>body { color: red }</style><script>alert(1)</script></head>
<body><nav>Home</nav><h1>Words</h1><p>Brushed   drums,
hushed vocals.</p><ul><li>tags:</li><li>slowcore</li></ul>&amp; more</body></html>"""


def test_page_text_keeps_metadata_and_visible_text() -> None:
    text = page_text(PAGE)

    assert text.startswith("Title: Words | Low\nDescription: Single from the album.\n")
    assert 'name: "Words"' in text
    assert 'byArtist: "Low"' in text
    assert 'keywords: ["slowcore", "dream pop"]' in text
    assert "duration" not in text
    assert "Brushed drums,\nhushed vocals." in text
    assert "slowcore" in text.split("\n\n", 1)[1]
    assert "& more" in text
    assert "alert" not in text
    assert "color: red" not in text


def test_page_text_reads_json_ld_graphs_and_skips_bad_blocks() -> None:
    graph = json.dumps({"@graph": [{"genre": "Ambient"}, "noise"]})
    page = (
        f'<script type="application/ld+json">{graph}</script>'
        '<script type="application/ld+json">{broken</script>'
        '<script type="application/ld+json">42</script><p>Body</p>'
    ).encode()

    assert page_text(page) == 'genre: "Ambient"\n\nBody'


def test_page_text_without_metadata_is_just_the_body() -> None:
    assert page_text(b"<p>Only text</p>") == "Only text"


def test_long_page_is_truncated() -> None:
    text = page_text(b"<p>" + b"word " * 100 + b"</p>", limit=50)

    assert len(text) <= 50 + len("\n[truncated]")
    assert text.endswith("\n[truncated]")
