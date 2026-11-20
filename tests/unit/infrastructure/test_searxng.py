import httpx
import pytest

from mixhound.infrastructure.search.searxng import Pacer, SearxngSearch
from mixhound.services.ports import SearchHit, SearchUnavailableError


class Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def search(handler: httpx.MockTransport, clock: Clock | None = None) -> SearxngSearch:
    clock = clock or Clock()
    return SearxngSearch(
        httpx.AsyncClient(transport=handler),
        "http://searx.local/",
        results=2,
        pacer=Pacer(1.0, clock=clock, sleep=clock.sleep),
    )


async def test_results_are_unique_web_hits_cut_to_the_count() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Archangel", "url": "https://example.org/a", "content": "x"},
                    {"title": "Again", "url": "https://example.org/a"},
                    {"url": "javascript:alert(1)"},
                    {"url": "http://example.org/b"},
                    {"url": "https://example.org/c"},
                ]
            },
        )

    hits = await search(httpx.MockTransport(handle)).search("burial archangel")

    assert hits == [
        SearchHit("Archangel", "https://example.org/a", "x"),
        SearchHit("", "http://example.org/b"),
    ]
    assert str(seen[0].url).startswith("http://searx.local/search?")
    assert seen[0].url.params["q"] == "burial archangel"
    assert seen[0].url.params["format"] == "json"


async def test_an_explicit_count_overrides_the_default() -> None:
    results = [{"url": f"https://example.org/{index}"} for index in range(5)]
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"results": results}))

    assert len(await search(transport).search("q", count=4)) == 4


async def test_queries_are_spaced_out() -> None:
    clock = Clock()
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    engine = search(transport, clock)

    await engine.search("a")
    clock.now += 0.25
    await engine.search("b")
    clock.now += 5
    await engine.search("c")

    assert clock.slept == [0.75]


async def test_disabled_json_output_is_explained() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(403))

    with pytest.raises(SearchUnavailableError, match=r"settings\.yml"):
        await search(transport).search("q")


@pytest.mark.parametrize(
    "response", [httpx.Response(429), httpx.Response(200, content=b"not json")]
)
async def test_failures_are_reported(response: httpx.Response) -> None:
    transport = httpx.MockTransport(lambda _: response)

    with pytest.raises(SearchUnavailableError):
        await search(transport).search("q")
