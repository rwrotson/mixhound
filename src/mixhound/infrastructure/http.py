import asyncio
import ipaddress
import json
import random
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from urllib.parse import urljoin, urlsplit

import httpx
import structlog

from mixhound.config import HttpSettings
from mixhound.services.ports import HTML, JSON, PAGE_BYTES, Accept, FetchError

log = structlog.get_logger(__name__)

MAX_REDIRECTS = 4
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
RETRY_STATUSES = frozenset({429, 502, 503, 504})

type Resolve = Callable[[str, int], Awaitable[Sequence[str]]]


async def resolve_addresses(host: str, port: int) -> list[str]:
    """Return the addresses DNS gives for a host."""
    entries = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [str(entry[4][0]) for entry in entries]


def open_http_client(settings: HttpSettings) -> httpx.AsyncClient:
    """Create the client of public page fetches; the fetcher follows redirects itself."""
    return httpx.AsyncClient(
        timeout=settings.timeout,
        follow_redirects=False,
        # Proxies from the environment would bypass the pinned, checked address.
        trust_env=False,
        headers={"User-Agent": settings.user_agent},
    )


@dataclass(frozen=True)
class FetchPolicy:
    """How a page fetch retries an unreachable host or a transient HTTP status."""

    max_retries: int = 2
    backoff_base: float = 0.5
    backoff_cap: float = 4.0
    retry_after_cap: float = 5.0
    retry_statuses: frozenset[int] = RETRY_STATUSES
    sleep: Callable[[float], Awaitable[None]] = field(
        default=asyncio.sleep, repr=False, compare=False
    )
    jitter: Callable[[float, float], float] = field(
        default=random.uniform,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_settings(cls, settings: HttpSettings) -> FetchPolicy:
        """Build the policy from HTTP settings."""
        return cls(
            max_retries=settings.max_retries,
            backoff_base=settings.backoff_base,
            backoff_cap=settings.backoff_cap,
            retry_after_cap=settings.retry_after_cap,
        )

    def retries(self, error: FetchError) -> bool:
        """Whether a failed attempt is worth repeating."""
        return error.code == "unreachable" and (
            error.status is None or error.status in self.retry_statuses
        )

    def delay(self, attempt: int, retry_after: float | None) -> float:
        """Seconds to wait before the retry after the zero-based `attempt`.

        A server's Retry-After wins over the jittered exponential backoff, up to a cap.
        """
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.retry_after_cap)
        return self.jitter(0.0, min(self.backoff_cap, self.backoff_base * 2**attempt))


@dataclass(frozen=True)
class _Request:
    """What a page fetch accepts back."""

    accept: Accept
    headers: Mapping[str, str]
    max_bytes: int
    truncate: bool


@dataclass(frozen=True)
class _Destination:
    scheme: str
    host: str
    port: int
    addresses: list[str]


class PageFetcher:
    """GET public URLs through pinned, SSRF-checked connections under one retry policy."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        policy: FetchPolicy | None = None,
        *,
        resolve: Resolve = resolve_addresses,
    ) -> None:
        self.client = client
        self.policy = policy or FetchPolicy()
        self.resolve = resolve

    async def page(
        self,
        url: str,
        *,
        max_bytes: int = PAGE_BYTES,
        truncate: bool = False,
        accept: Accept = HTML,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[str, bytes]:
        """Return (final URL, body) of a public URL, following redirects.

        With `truncate`, an oversized body is cut at `max_bytes` instead of being rejected.
        """
        request = _Request(accept, headers or {}, max_bytes, truncate)
        for _ in range(MAX_REDIRECTS):
            redirect, content = await self._hop(url, request)
            if redirect is None:
                return url, content
            url = redirect
        raise FetchError("too_many_redirects")

    async def json(
        self, url: str, *, max_bytes: int = PAGE_BYTES, headers: Mapping[str, str] | None = None
    ) -> object:
        """Fetch a public URL and parse its JSON body."""
        _, body = await self.page(url, max_bytes=max_bytes, accept=JSON, headers=headers)
        try:
            return json.loads(body)
        except ValueError as error:
            raise FetchError("invalid_json") from error

    async def destination(self, url: str) -> _Destination:
        """Return where a URL that is safe to fetch points: only public addresses."""
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise FetchError("invalid_url")
        host = parsed.hostname.rstrip(".").lower()
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as error:
            raise FetchError("invalid_url") from error
        try:
            public = list(await self.resolve(host, port))
        except OSError as error:
            raise FetchError("unreachable") from error
        if not public or any(not ipaddress.ip_address(address).is_global for address in public):
            raise FetchError("unsafe_destination")
        return _Destination(parsed.scheme, host, port, list(dict.fromkeys(public)))

    async def _hop(self, url: str, request: _Request) -> tuple[str | None, bytes]:
        """Fetch one URL of a redirect chain, retrying transient failures."""
        attempt = 0
        while True:
            try:
                return await self._attempt(url, request)
            except FetchError as error:
                if attempt >= self.policy.max_retries or not self.policy.retries(error):
                    raise
                delay = self.policy.delay(attempt, error.retry_after)
                log.info(
                    "url_fetch_retry",
                    host=urlsplit(url).hostname,
                    attempt=attempt + 1,
                    delay=round(delay, 2),
                    status=error.status,
                )
                await self.policy.sleep(delay)
                attempt += 1

    async def _attempt(self, url: str, request: _Request) -> tuple[str | None, bytes]:
        """Resolve and check the host again, then try each of its public addresses in turn."""
        destination = await self.destination(url)
        for address in destination.addresses:
            try:
                return await self._get(url, destination, address, request)
            except httpx.TransportError as error:
                log.warning(
                    "url_fetch_connection_error",
                    host=destination.host,
                    address=address,
                    error=type(error).__name__,
                )
                if address == destination.addresses[-1]:
                    raise FetchError("unreachable") from error
        raise FetchError("unreachable")  # pragma: no cover - destination() never returns none.

    async def _get(
        self, url: str, destination: _Destination, address: str, request: _Request
    ) -> tuple[str | None, bytes]:
        """Send one GET to a checked address; return (redirect URL, b"") or (None, body)."""
        host = destination.host
        default_port = 443 if destination.scheme == "https" else 80
        authority = host if destination.port == default_port else f"{host}:{destination.port}"
        pinned = httpx.URL(url).copy_with(host=address, port=destination.port)
        # TLS still verifies the certificate of `host`, not of the pinned address.
        extensions = {"sni_hostname": host} if destination.scheme == "https" else {}
        headers = {"Host": authority, "Accept": request.accept.header, **request.headers}
        async with self.client.stream(
            "GET", pinned, headers=headers, extensions=extensions
        ) as response:
            if response.status_code in REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if not location:
                    raise FetchError("unreachable", response.status_code)
                return urljoin(url, location), b""
            if response.status_code != HTTPStatus.OK:
                log.warning("url_fetch_http_error", host=host, status=response.status_code)
                raise FetchError(
                    "unreachable",
                    response.status_code,
                    retry_after=_retry_after(response.headers.get("Retry-After")),
                )
            content_type = response.headers.get("Content-Type", "").lower()
            if not any(kind in content_type for kind in request.accept.content_types):
                raise FetchError("unsupported_content")
            content = await _read(response, request.max_bytes + 1)
        if len(content) > request.max_bytes:
            if not request.truncate:
                raise FetchError("page_too_large")
            content = content[: request.max_bytes]
        return None, content


async def _read(response: httpx.Response, limit: int) -> bytes:
    """Read at most `limit` bytes of a streamed body."""
    chunks = bytearray()
    async for chunk in response.aiter_bytes():
        chunks += chunk
        if len(chunks) >= limit:
            break
    return bytes(chunks[:limit])


def _retry_after(value: str | None) -> float | None:
    """Seconds named by a Retry-After header: a delay or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)
