import asyncio
import logging
import math
from dataclasses import dataclass, field
from uuid import UUID

from mixhound.memory.context import RunHistory
from mixhound.models.domain import DiscoveryReference, SonicProfile
from mixhound.models.research import ResearchCandidate, ResearchOutput
from mixhound.services.audio import AudioMatcher
from mixhound.services.budgets import DEPTH_BUDGETS
from mixhound.services.context import StrategyShare, build_context
from mixhound.services.evidence import research_result
from mixhound.services.exclusions import Exclusions
from mixhound.services.preparation import Preparation, ReferenceFacts
from mixhound.services.research import ResearchError, ResearchSteps
from mixhound.services.run_state import RunState
from mixhound.services.seed import (
    MAX_SEED_REFERENCES,
    LeadPool,
    Seeder,
    reference_leads,
    seed_order,
)
from mixhound.services.strategies import STRATEGIES, split_pool, strategy_budget, strategy_max_usd
from mixhound.services.verification import ResultVerifier

log = logging.getLogger(__name__)

# A delegated research ends by this share of the orchestrator's time, leaving the rest to
# verify and answer.
DELEGATE_SHARE = 0.75
MIN_DELEGATE_SECONDS = 60.0
MAX_BRIEF = 1000
# Appended to the agentic prompt, whose references carry no catalog facts or pool yet.
ORCHESTRATION_NOTE = (
    "\nCatalog facts, measured audio and the candidate pool are not prepared yet: call "
    "lookup_reference for each reference you need them for."
)


@dataclass(frozen=True)
class ReferenceReport:
    """What the orchestrator learns from looking a reference up."""

    reference: DiscoveryReference
    facts: ReferenceFacts
    found: bool
    catalog_down: bool
    # Leads this lookup added to the run's candidate pool.
    leads: list[dict[str, object]]


@dataclass(frozen=True)
class SourceCheck:
    """One source page the server opened and whether it names the recording."""

    url: str
    matches: bool


@dataclass(frozen=True)
class CandidateReport:
    """What the server established about one candidate."""

    artist: str
    title: str
    confirmed: bool
    confirmed_by: tuple[str, ...] = ()
    year: int | None = None
    listening_urls: list[str] = field(default_factory=list)
    sources: list[SourceCheck] = field(default_factory=list)
    audio_similarity: float | None = None
    audio_evidence: str | None = None


@dataclass(frozen=True)
class DelegateReport:
    """How a delegated strategy research ended."""

    strategy: str
    candidates: list[ResearchCandidate] = field(default_factory=list)
    # A failure code, or why the delegation was refused.
    failure: str | None = None
    partial: bool = False


@dataclass(frozen=True)
class DelegateLimits:
    """Time and money a delegated research may spend."""

    seconds: float
    max_usd: float


def delegated_cost(state: RunState) -> float:
    """Estimated cost of the strategy researches that finished so far."""
    return sum(state.usage[name].cost_usd or 0.0 for name in STRATEGIES if name in state.usage)


def delegate_limits(
    state: RunState, elapsed: float, limit: float, own_cost: float | None
) -> DelegateLimits | None:
    """Limits of a new delegated research, or None when too little time or money is left.

    The run's spending cap covers the orchestrator and every delegated research together.
    """
    max_usd = state.run.input.settings.max_usd
    seconds = limit * DELEGATE_SHARE - elapsed
    left = max_usd - (own_cost or 0.0) - delegated_cost(state)
    usd = round(min(strategy_max_usd(max_usd), left), 2)
    if seconds < MIN_DELEGATE_SECONDS or usd <= 0:
        return None
    return DelegateLimits(seconds, usd)


@dataclass(frozen=True)
class RunServices:
    """The services a run's stages and its orchestrator's tools share."""

    seeder: Seeder
    verifier: ResultVerifier
    audio_matcher: AudioMatcher
    steps: ResearchSteps


class AgenticWorkspace:
    """What the orchestrator of one agentic run seeds, verifies and delegates through."""

    def __init__(
        self,
        state: RunState,
        history: RunHistory,
        exclusions: Exclusions,
        services: RunServices,
    ) -> None:
        self.state = state
        self.history = history
        self.seeder = services.seeder
        self.verifier = services.verifier
        self.audio_matcher = services.audio_matcher
        self.steps = services.steps
        self.pool = LeadPool(state.run.input, exclusions)
        self.preparation = Preparation(pool=self.pool.leads)
        # Strategy answers kept for the server-side merge with the orchestrator's answer.
        self.delegated: dict[str, ResearchOutput] = {}
        self._reports: dict[UUID, ReferenceReport] = {}
        self._started: set[str] = set()
        self._seeded = 0
        self._lookups = asyncio.Lock()

    @property
    def references(self) -> list[DiscoveryReference]:
        """The references worth looking up, highest priority first."""
        return seed_order(self.state.run.input)

    def reference(self, reference_id: str) -> DiscoveryReference | None:
        """A non-neutral reference of the run by its ID."""
        return next((item for item in self.references if str(item.id) == reference_id), None)

    def _pool_share(self) -> int:
        """How many leads one positive reference may add to the pool."""
        size = DEPTH_BUDGETS[self.state.run.input.settings.depth].pool
        positives = sum(item.polarity == "positive" for item in self.references)
        return math.ceil(size / max(min(positives, MAX_SEED_REFERENCES), 1))

    async def lookup_reference(self, reference: DiscoveryReference) -> ReferenceReport:
        """Look a reference up once; a positive one adds its share of leads to the pool."""
        async with self._lookups:
            known = self._reports.get(reference.id)
            if known is not None:
                return known
            self.state.check()
            found = await self.seeder.lookup(reference, catalog=not self.state.catalog_down)
            self.state.catalog_down = self.state.catalog_down or found.catalog_down
            self.preparation.facts[reference.id] = found.facts
            leads: list[dict[str, object]] = []
            if reference.polarity == "positive" and self._seeded < MAX_SEED_REFERENCES:
                self._seeded += 1
                size = DEPTH_BUDGETS[self.state.run.input.settings.depth].pool
                limit = min(len(self.pool.leads) + self._pool_share(), size)
                leads = self.pool.interleave([(reference, reference_leads(found.recording))], limit)
            self.state.metrics.pool_size = len(self.pool.leads)
            report = ReferenceReport(
                reference,
                found.facts,
                found=found.recording is not None,
                catalog_down=found.catalog_down,
                leads=leads,
            )
            self._reports[reference.id] = report
            return report

    async def verify(self, candidates: list[ResearchCandidate]) -> list[CandidateReport]:
        """Confirm candidates as the server will: catalogs, source pages and audio.

        The checks run on a scratch state, so only the catalog answers carry over to the run;
        the final verification repeats them from the catalog memo and the page and audio caches.
        """
        state = self.state
        scratch = RunState(
            state.run,
            state.cancelled,
            catalog_down=state.catalog_down,
            catalog_answers=state.catalog_answers,
        )
        output = ResearchOutput(sonic_profile=SonicProfile(), candidates=candidates)
        result = research_result(output, state.run.input)
        # The research's own sources; catalog pages added below are reported as confirmation.
        claimed = {track.id: {source.url for source in track.sources} for track in result.results}
        await self.verifier.confirm_catalog(scratch, result)
        await self.verifier.confirm_sources(scratch, result)
        await self.audio_matcher.compare(scratch, result, self.preparation)
        state.catalog_down = scratch.catalog_down
        reports = []
        for track in result.results:
            recording = scratch.recordings.get(track.id)
            confirmed = recording is not None and recording.confirmed
            reports.append(
                CandidateReport(
                    track.artist,
                    track.title,
                    confirmed=confirmed,
                    confirmed_by=recording.confirmed_by if recording and confirmed else (),
                    year=track.year,
                    listening_urls=[
                        link.url for link in track.listening_links if link.reachable is True
                    ],
                    sources=[
                        SourceCheck(source.url, source.matches_recording)
                        for source in track.sources
                        if source.matches_recording is not None and source.url in claimed[track.id]
                    ],
                    audio_similarity=track.audio_similarity,
                    audio_evidence=track.audio_analysis_evidence,
                )
            )
        return reports

    def strategy_prompt(self, strategy: str, profile: SonicProfile, brief: str) -> str:
        """The prompt of a delegated strategy research: its share of the pool and a brief."""
        run = self.state.run
        share = StrategyShare(
            strategy,
            STRATEGIES[strategy],
            profile,
            split_pool(self.pool.leads)[strategy],
            strategy_budget(run.input.settings.depth),
            len(STRATEGIES),
        )
        prompt = build_context(run, self.history, self.preparation, share)[0]
        brief = brief.strip()[:MAX_BRIEF]
        return f"{prompt}\nOrchestrator brief: {brief}" if brief else prompt

    async def delegate(
        self, strategy: str, profile: SonicProfile, brief: str, limits: DelegateLimits | None
    ) -> DelegateReport:
        """Run one strategy research; its answer joins the run's candidates server-side."""
        if strategy not in STRATEGIES:
            return DelegateReport(strategy, failure="unknown_strategy")
        if strategy in self._started:
            return DelegateReport(strategy, failure="already_delegated")
        if limits is None:
            return DelegateReport(strategy, failure="no_time_or_budget_left")
        self._started.add(strategy)
        prompt = self.strategy_prompt(strategy, profile, brief)
        try:
            output = await self.steps.delegate(
                self.state, strategy, prompt, limits.seconds, limits.max_usd
            )
        except ResearchError as error:
            self.state.check()
            log.warning(
                "discovery_delegate_failed run=%s strategy=%s code=%s",
                self.state.run.id,
                strategy,
                error.code,
            )
            return DelegateReport(strategy, failure=error.code)
        self.delegated[strategy] = output
        failed = self.state.metrics.strategies[strategy].failed
        return DelegateReport(
            strategy, output.candidates, failure=failed, partial=failed is not None
        )

    async def complete(self) -> None:
        """Look up the references the orchestrator skipped, for their measured audio."""
        if self.seeder.audio is None:
            return
        for reference in self.references:
            if reference.id not in self.preparation.facts:
                self.state.check()
                found = await self.seeder.lookup(reference, catalog=not self.state.catalog_down)
                self.state.catalog_down = self.state.catalog_down or found.catalog_down
                self.preparation.facts[reference.id] = found.facts
