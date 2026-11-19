from dataclasses import dataclass, field

import httpx

from mixhound.infrastructure.http import FetchPolicy, PageFetcher

PUBLIC_ADDRESS = "93.184.216.34"


@dataclass
class FakeResponse:
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


type Reply = FakeResponse | Exception


@dataclass
class Sent:
    host: str
    address: str
    path: str
    request: httpx.Request


class FakeNetwork:
    """Serves replies by path, or by host and path, through pinned addresses."""

    def __init__(
        self,
        replies: dict[str, Reply | list[Reply]],
        addresses: dict[str, list[str]] | None = None,
    ) -> None:
        self.replies = replies
        self.addresses = addresses or {}
        self.requests: list[Sent] = []
        self.sleeps: list[float] = []

    async def resolve(self, host: str, port: int) -> list[str]:
        return self.addresses.get(host, [PUBLIC_ADDRESS])

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def handle(self, request: httpx.Request) -> httpx.Response:
        host = request.headers["host"].split(":")[0]
        path = request.url.raw_path.decode()
        self.requests.append(Sent(host, request.url.host, path, request))
        key = path if path in self.replies else f"{host}{path}"
        reply = self.replies.get(key, FakeResponse(404))
        if isinstance(reply, list):
            reply = reply.pop(0)
        if isinstance(reply, Exception):
            raise httpx.ConnectError(str(reply), request=request)
        return httpx.Response(reply.status, headers=reply.headers, content=reply.body)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def fetcher(self, max_retries: int = 0) -> PageFetcher:
        policy = FetchPolicy(max_retries=max_retries, sleep=self.sleep, jitter=lambda _, high: high)
        return PageFetcher(self.client(), policy, resolve=self.resolve)
