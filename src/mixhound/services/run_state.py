import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from uuid import UUID

from mixhound.models.domain import DiscoveryRun, DiscoveryTrack, RunMetrics
from mixhound.services.errors import MixhoundError
from mixhound.services.ports import CatalogRecording

CONCURRENCY = 4


async def bounded[T](
    items: Iterable[T],
    work: Callable[[T], Awaitable[None]],
    limit: int = CONCURRENCY,
) -> None:
    """Run `work` for every item with at most `limit` in flight; the first error propagates."""
    semaphore = asyncio.Semaphore(limit)

    async def one(item: T) -> None:
        async with semaphore:
            await work(item)

    tasks = [asyncio.ensure_future(one(item)) for item in items]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()


def alternates(track: DiscoveryTrack) -> tuple[str | None, str | None]:
    """Return a result's alternate artist and title."""
    return track.artist_alt, track.title_alt


class RunCancelledError(MixhoundError):
    """The active run was cancelled between stages."""


def _never() -> bool:
    return False


@dataclass
class RunState:
    """What one run accumulates while its results are verified."""

    run: DiscoveryRun
    cancelled: Callable[[], bool] = _never
    metrics: RunMetrics = field(default_factory=RunMetrics)
    confirmed: set[UUID] = field(default_factory=set)
    # URLs whose identity the server already established (catalog or page match).
    trusted_urls: dict[UUID, set[str]] = field(default_factory=dict)
    # Catalog answers per result, read again by the audio match and the agents.
    recordings: dict[UUID, CatalogRecording] = field(default_factory=dict)
    catalog_down: bool = False

    def check(self) -> None:
        """Stop the run between steps once it was cancelled."""
        if self.cancelled():
            raise RunCancelledError

    def trust(self, track: DiscoveryTrack, url: str) -> None:
        """Mark a result confirmed and a URL as establishing its identity."""
        self.confirmed.add(track.id)
        self.trusted_urls.setdefault(track.id, set()).add(url)
