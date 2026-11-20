import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from uuid import UUID

from mixhound.models.domain import DiscoveryRun, DiscoveryTrack, RunMetrics
from mixhound.models.state import RunProgress
from mixhound.services.errors import MixhoundError
from mixhound.services.evidence import Rejection
from mixhound.services.ports import CatalogRecording

CONCURRENCY = 4
# LLM calls that support the research rather than research the references themselves.
RESEARCH_HELPER_LABELS = {"profile", "listening"}
RESEARCH_STAGE = "research"
# Every stage in run order, for the progress step; a run skips the ones it does not need.
STAGE_ORDER = (
    "prepare",
    "context",
    "profile",
    RESEARCH_STAGE,
    "evidence",
    "catalog",
    "independent_verification",
    "link_search",
    "listening",
    "listening_verification",
    "audio",
    "publish",
)


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
class LlmUsage:
    """Token, cost and tool counts of one agent run."""

    cost_usd: float | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0
    turns: int = 0
    failed_fetches: int = 0


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
    # Catalog answers by recording key, so a recording checked twice costs one request.
    catalog_answers: dict[str, CatalogRecording | None] = field(default_factory=dict)
    stage: str = "prepare"
    # What the run reports when it fails: the failure code or detail of the stage.
    detail: str = ""
    rejections: list[Rejection] = field(default_factory=list)
    # Each agent call's final answer (JSON or text) by label, for the diagnostic.
    answers: dict[str, str] = field(default_factory=dict)
    # Interim candidate notes of each failed agent call, by label, for the diagnostic.
    notes: dict[str, list[str]] = field(default_factory=dict)
    # Live searches and noted candidates of each running agent call, by label.
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    research_started: float | None = None
    # Each agent call's usage by label.
    usage: dict[str, LlmUsage] = field(default_factory=dict)

    def check(self) -> None:
        """Stop the run between steps once it was cancelled."""
        if self.cancelled():
            raise RunCancelledError

    def trust(self, track: DiscoveryTrack, url: str) -> None:
        """Mark a result confirmed and a URL as establishing its identity."""
        self.confirmed.add(track.id)
        self.trusted_urls.setdefault(track.id, set()).add(url)

    @contextmanager
    def enter(self, name: str) -> Iterator[None]:
        """Name the current stage and add its wall time to the run metrics."""
        self.check()
        self.stage = name
        started = time.monotonic()
        if name == RESEARCH_STAGE and self.research_started is None:
            self.research_started = started
        try:
            yield
        finally:
            stages = self.metrics.stages
            stages[name] = round(stages.get(name, 0.0) + time.monotonic() - started, 2)

    def listener(self, label: str) -> Callable[[int, int], None]:
        """Record one agent call's live searches and noted candidates under `label`."""

        def hear(searches: int, candidates: int) -> None:
            self.counts[label] = (searches, candidates)

        return hear

    def progress(self, timeout: float) -> RunProgress:
        """Return the progress reported to the client."""
        elapsed = (
            time.monotonic() - self.research_started if self.research_started is not None else 0.0
        )
        step = STAGE_ORDER.index(self.stage) + 1 if self.stage in STAGE_ORDER else 0
        return RunProgress(
            stage=self.stage,
            step=step,
            steps=len(STAGE_ORDER),
            elapsed=round(elapsed, 1),
            timeout=timeout,
            searches=sum(count[0] for count in self.counts.values()),
            candidates=sum(count[1] for count in self.counts.values()),
        )

    def add_usage(self, label: str, usage: LlmUsage) -> None:
        """Add one agent call's usage to the run metrics."""
        self.usage[label] = usage
        metrics = self.metrics
        metrics.llm_calls += 1
        if usage.cost_usd is not None:
            metrics.cost_usd = round((metrics.cost_usd or 0) + usage.cost_usd, 4)
        metrics.input_tokens += usage.input_tokens
        metrics.cached_input_tokens += usage.cached_input_tokens
        metrics.output_tokens += usage.output_tokens
        metrics.web_searches += usage.web_searches
        if label not in RESEARCH_HELPER_LABELS:
            metrics.research_searches += usage.web_searches
        metrics.turns += usage.turns
        metrics.failed_fetches += usage.failed_fetches
