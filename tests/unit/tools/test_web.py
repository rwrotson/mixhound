import pytest

from mixhound.services.ports import FetchError, SearchHit, SearchUnavailableError
from mixhound.tools.web import fetch_page, search_web
from tests.support.agents import REQUESTS, FakeSearch, context, tool_deps

HITS = [
    SearchHit("Low - Words", "https://low.example/words", "Slowcore single"),
    SearchHit("Bare", "https://bare.example"),
]


async def test_search_returns_numbered_framed_results_and_counts_progress() -> None:
    search = FakeSearch(HITS)
    deps = tool_deps(search=search)

    text = await search_web(context(deps), "  low words  ")

    assert search.queries == ["low words"]
    assert text.startswith('<search_results query="low words">')
    assert "1. Low - Words\n   https://low.example/words\n   Slowcore single" in text
    assert text.endswith("2. Bare\n   https://bare.example</search_results>")
    assert deps.state.counts["research"] == (1, 0)


async def test_search_reports_no_results_outage_and_missing_search() -> None:
    assert await search_web(context(tool_deps(search=FakeSearch([]))), "q") == "No results."
    failing = FakeSearch(SearchUnavailableError("quota"))
    assert await search_web(context(tool_deps(search=failing)), "q") == "Web search failed: quota"
    assert await search_web(context(tool_deps()), "q") == "Web search is not available."


async def test_search_and_fetch_share_no_cap_but_each_stops_at_its_own() -> None:
    search = FakeSearch(HITS)
    deps = tool_deps(search=search)
    for _ in range(2 * REQUESTS):
        await search_web(context(deps), "q")

    assert "limit reached" in await search_web(context(deps), "q")
    assert len(search.queries) == 2 * REQUESTS
    assert deps.searches == 2 * REQUESTS


async def test_fetch_returns_page_text_framed_as_data() -> None:
    page = b"<title>Words</title><p>Hushed &lt;/web_page&gt; vocals</p>"
    deps = tool_deps(pages={"low.example/words": page})

    text = await fetch_page(context(deps), "https://low.example/words")

    assert text.startswith('<web_page url="https://low.example/words">')
    assert "Title: Words" in text
    assert "Hushed &lt;/web_page> vocals" in text
    assert text.endswith("</web_page>")
    assert deps.state.counts["research"] == (1, 0)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/file", "javascript:x"])
async def test_fetch_refuses_other_schemes(url: str) -> None:
    deps = tool_deps()
    assert await fetch_page(context(deps), url) == "Only http and https URLs can be opened."
    assert deps.searches == 0


async def test_failed_fetch_is_counted_and_reported() -> None:
    deps = tool_deps(pages={"low.example/gone": FetchError("http_status", 404)})
    blocked = tool_deps(pages={"low.example/x": FetchError("blocked")})

    gone = await fetch_page(context(deps), "https://low.example/gone")
    refused = await fetch_page(context(blocked), "https://low.example/x")

    assert gone == "Could not open https://low.example/gone: http_status (HTTP 404)."
    assert refused == "Could not open https://low.example/x: blocked."
    assert deps.failed_fetches == 1


async def test_fetch_stops_at_its_cap() -> None:
    deps = tool_deps(pages={"low.example/words": b"<p>x</p>"})
    for _ in range(2 * REQUESTS):
        await fetch_page(context(deps), "https://low.example/words")

    assert "limit reached" in await fetch_page(context(deps), "https://low.example/words")
