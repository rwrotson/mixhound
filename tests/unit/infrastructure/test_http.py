import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from mixhound.config import HttpSettings
from mixhound.infrastructure import http
from mixhound.infrastructure.http import FetchPolicy, PageFetcher, open_http_client
from mixhound.services.ports import FetchError
from tests.support.fake_http import FakeNetwork, FakeResponse

HTML = {"Content-Type": "text/html; charset=utf-8"}
JSON = {"Content-Type": "application/json"}
BASE = "http://music.example"
URL = f"{BASE}/song"


async def test_redirects_content_types_and_size_limits() -> None:
    fetcher = FakeNetwork(
        {
            "/start": FakeResponse(302, {"Location": "/final"}),
            "/final": FakeResponse(200, HTML, b"done"),
            "/loop": FakeResponse(302, {"Location": "/loop"}),
            "/nowhere": FakeResponse(302),
            "/missing": FakeResponse(404, HTML),
            "/image": FakeResponse(200, {"Content-Type": "image/png"}),
            "/large": FakeResponse(200, HTML, b"x" * 20),
        }
    ).fetcher()

    assert await fetcher.page(f"{BASE}/start") == (f"{BASE}/final", b"done")
    for path, code in (
        ("/loop", "too_many_redirects"),
        ("/nowhere", "unreachable"),
        ("/missing", "unreachable"),
        ("/image", "unsupported_content"),
    ):
        with pytest.raises(FetchError, match=code):
            await fetcher.page(BASE + path)
    with pytest.raises(FetchError, match="page_too_large"):
        await fetcher.page(f"{BASE}/large", max_bytes=10)
    assert (await fetcher.page(f"{BASE}/large", max_bytes=10, truncate=True))[1] == b"x" * 10


@pytest.mark.parametrize(
    "url", ["ftp://music.example/", "http://user@music.example/", "http://music.example:99999/"]
)
async def test_invalid_urls_are_rejected(url: str) -> None:
    with pytest.raises(FetchError, match="invalid_url"):
        await FakeNetwork({}).fetcher().page(url)


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.1.1", "10.0.0.1", "::1"])
async def test_private_destinations_are_rejected_before_connecting(address: str) -> None:
    network = FakeNetwork({}, {"music.example": ["8.8.8.8", address]})

    with pytest.raises(FetchError, match="unsafe_destination"):
        await network.fetcher().page(URL)

    assert network.requests == []


async def test_redirect_to_private_address_is_blocked() -> None:
    network = FakeNetwork(
        {"/song": FakeResponse(302, {"Location": "http://internal.example/private"})},
        {"internal.example": ["127.0.0.1"]},
    )

    with pytest.raises(FetchError, match="unsafe_destination"):
        await network.fetcher().page(URL)

    assert [sent.host for sent in network.requests] == ["music.example"]


async def test_requests_go_to_the_checked_address_with_the_original_host() -> None:
    network = FakeNetwork(
        {"/song?id=1": FakeResponse(200, HTML, b"ok")},
        {"music.example": ["2606:4700::6810:84e5"]},
    )

    assert await network.fetcher().page("https://music.example/song?id=1") == (
        "https://music.example/song?id=1",
        b"ok",
    )
    sent = network.requests[0]
    assert sent.address == "2606:4700::6810:84e5"
    assert sent.request.headers["host"] == "music.example"
    assert sent.request.extensions["sni_hostname"] == "music.example"


async def test_a_custom_port_stays_in_the_host_header() -> None:
    network = FakeNetwork({"/song": FakeResponse(200, HTML, b"ok")})

    await network.fetcher().page("http://music.example:8080/song")

    assert network.requests[0].request.headers["host"] == "music.example:8080"
    assert network.requests[0].request.url.port == 8080
    assert "sni_hostname" not in network.requests[0].request.extensions


async def test_dns_failure_is_unreachable() -> None:
    async def resolve(host: str, port: int) -> list[str]:
        raise OSError("no such host")

    fetcher = PageFetcher(FakeNetwork({}).client(), FetchPolicy(max_retries=0), resolve=resolve)

    with pytest.raises(FetchError, match="unreachable"):
        await fetcher.page(URL)


async def test_next_public_address_is_tried_after_connection_failure() -> None:
    network = FakeNetwork(
        {"/song": [OSError("first address failed"), FakeResponse(200, HTML, b"ok")]},
        {"music.example": ["8.8.8.8", "1.1.1.1"]},
    )

    assert await network.fetcher().page(URL) == (URL, b"ok")
    assert [sent.address for sent in network.requests] == ["8.8.8.8", "1.1.1.1"]


async def test_json_parses_bodies_and_rejects_invalid_ones() -> None:
    fetcher = FakeNetwork(
        {"/ok": FakeResponse(200, JSON, b'{"a": 1}'), "/bad": FakeResponse(200, JSON, b"{")}
    ).fetcher()

    assert await fetcher.json(f"{BASE}/ok") == {"a": 1}
    with pytest.raises(FetchError, match="invalid_json"):
        await fetcher.json(f"{BASE}/bad")


def test_backoff_doubles_up_to_cap_and_retry_after_wins_within_its_cap() -> None:
    policy = FetchPolicy(jitter=lambda _low, high: high)

    assert [policy.delay(attempt, None) for attempt in range(5)] == [0.5, 1.0, 2.0, 4.0, 4.0]
    assert policy.delay(0, 2.0) == 2.0
    assert policy.delay(0, 30.0) == 5.0
    assert policy.delay(0, -3.0) == 0.0


def test_jitter_draws_between_zero_and_the_backoff() -> None:
    bounds: list[tuple[float, float]] = []

    def jitter(low: float, high: float) -> float:
        bounds.append((low, high))
        return low

    assert FetchPolicy(jitter=jitter).delay(2, None) == 0.0
    assert bounds == [(0.0, 2.0)]


def test_policy_reads_http_settings() -> None:
    settings = HttpSettings(max_retries=5, backoff_base=1, backoff_cap=8, retry_after_cap=9)

    policy = FetchPolicy.from_settings(settings)

    assert (policy.max_retries, policy.backoff_base, policy.backoff_cap) == (5, 1, 8)
    assert policy.retry_after_cap == 9


async def test_service_unavailable_waits_for_retry_after() -> None:
    network = FakeNetwork(
        {"/song": [FakeResponse(503, {"Retry-After": "2"}), FakeResponse(200, HTML, b"ok")]}
    )

    assert await network.fetcher(max_retries=2).page(URL) == (URL, b"ok")
    assert network.sleeps == [2.0]


async def test_missing_page_is_not_retried() -> None:
    network = FakeNetwork({"/song": FakeResponse(404, HTML)})

    with pytest.raises(FetchError) as raised:
        await network.fetcher(max_retries=2).page(URL)

    assert raised.value.status == 404
    assert (len(network.requests), network.sleeps) == (1, [])


async def test_exhausted_retries_report_the_last_status() -> None:
    network = FakeNetwork({"/song": [FakeResponse(429), FakeResponse(502), FakeResponse(503)]})

    with pytest.raises(FetchError) as raised:
        await network.fetcher(max_retries=2).page(URL)

    assert (raised.value.code, raised.value.status) == ("unreachable", 503)
    assert len(network.requests) == 3
    assert network.sleeps == [0.5, 1.0]


async def test_unreachable_host_is_retried_after_every_address_fails() -> None:
    network = FakeNetwork(
        {"/song": [OSError("down"), OSError("down"), FakeResponse(200, HTML, b"ok")]},
        {"music.example": ["8.8.8.8", "1.1.1.1"]},
    )

    assert await network.fetcher(max_retries=2).page(URL) == (URL, b"ok")
    assert [sent.address for sent in network.requests] == ["8.8.8.8", "1.1.1.1", "8.8.8.8"]
    assert network.sleeps == [0.5]


async def test_no_retries_when_disabled() -> None:
    network = FakeNetwork({"/song": FakeResponse(503)})

    with pytest.raises(FetchError):
        await network.fetcher(max_retries=0).page(URL)

    assert network.sleeps == []


def test_retry_after_reads_seconds_and_http_dates() -> None:
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    past = format_datetime(datetime.now(UTC) - timedelta(seconds=30), usegmt=True)
    naive = (datetime.now(UTC) + timedelta(seconds=30)).strftime("%d %b %Y %H:%M:%S")

    assert http._retry_after("7") == 7.0
    assert 25.0 < (http._retry_after(later) or 0.0) <= 30.0
    assert http._retry_after(past) == 0.0
    assert http._retry_after(f"Mon, {naive}") is not None
    assert http._retry_after("soon") is None
    assert http._retry_after(None) is None


async def test_http_client_follows_no_redirects_and_ignores_proxies() -> None:
    async with open_http_client(HttpSettings(timeout=3, user_agent="test/1")) as client:
        assert client.follow_redirects is False
        assert client.headers["User-Agent"] == "test/1"
        assert client.timeout.read == 3


async def test_default_resolver_uses_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    async def getaddrinfo(host: str, port: int, **_: object) -> list[tuple[object, ...]]:
        return [(2, 1, 6, "", ("8.8.8.8", port))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)

    assert await http.resolve_addresses("music.example", 443) == ["8.8.8.8"]
