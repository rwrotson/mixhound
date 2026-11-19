import httpx
import pytest

from mixhound.infrastructure.search.brave import ENDPOINT, BraveSearch
from mixhound.services.ports import SearchHit, SearchUnavailableError


def search(handler: httpx.MockTransport) -> BraveSearch:
    return BraveSearch(httpx.AsyncClient(transport=handler), "key", results=5)


async def test_results_are_hits_with_web_urls_only() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {"title": "Reckoner", "url": "https://example.org/a", "description": "x"},
                        {"url": "javascript:alert(1)"},
                    ]
                }
            },
        )

    hits = await search(httpx.MockTransport(handle)).search("radiohead reckoner")

    assert hits == [SearchHit("Reckoner", "https://example.org/a", "x")]
    assert str(seen[0].url).startswith(ENDPOINT)
    assert seen[0].url.params["count"] == "5"
    assert seen[0].headers["X-Subscription-Token"] == "key"


async def test_a_response_without_web_results_is_empty() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={}))

    assert await search(transport).search("q", count=2) == []


@pytest.mark.parametrize(
    "response", [httpx.Response(429), httpx.Response(200, content=b"not json")]
)
async def test_failures_are_reported(response: httpx.Response) -> None:
    transport = httpx.MockTransport(lambda _: response)

    with pytest.raises(SearchUnavailableError):
        await search(transport).search("q")
