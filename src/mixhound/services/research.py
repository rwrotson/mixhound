import asyncio
import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol

from mixhound.memory.context import RunHistory
from mixhound.models.domain import (
    DiscoveryListeningLink,
    DiscoveryResearchResult,
    DiscoverySession,
    DiscoveryTrack,
    SonicProfile,
    StrategyMetrics,
)
from mixhound.models.research import LinkSearchOutput, ResearchOutput
from mixhound.services.budgets import DEPTH_BUDGETS, HELPER_REQUESTS
from mixhound.services.context import (
    StrategyShare,
    build_context,
    previous_profile,
    profile_context,
)
from mixhound.services.errors import MixhoundError
from mixhound.services.evidence import merge_outputs, notes_output
from mixhound.services.listening import followup_prompt, missing_links, proposed_links
from mixhound.services.preparation import Preparation
from mixhound.services.run_state import RESEARCH_STAGE, RunState
from mixhound.services.skills import (
    LINK_SEARCH_SKILL,
    ORCHESTRATION_SKILL,
    PROFILE_SKILL,
    SkillCatalog,
)
from mixhound.services.strategies import (
    STRATEGIES,
    split_pool,
    strategy_budget,
    strategy_max_usd,
)
from mixhound.services.verification import ResultVerifier, likely_result

if TYPE_CHECKING:
    from mixhound.services.agentic import AgenticWorkspace

log = logging.getLogger(__name__)

# Time limit of the short helper calls (profile pass, listening-link search).
HELPER_SECONDS = 300
PROFILE_MAX_USD = 0.5
MAX_FOLLOWUP_TRACKS = 15
FOLLOWUP_MAX_USD = 0.75
# Failures that stop a research midway; the interim notes written until then are kept.
NOTED_FAILURES = frozenset({"timeout", "budget_exceeded", "limit_reached"})
# Replaces V1's "Interim notes directory" line: the notes are a tool call here.
NOTES_REMINDER = (
    "\nInterim notes: call record_candidate with each candidate as soon as you have scored it."
)

type Effort = Literal["low", "medium", "high", "xhigh", "max", "ultra"]


@dataclass(frozen=True)
class AgentCall:
    """How one agent call runs: its label, limits, model and instructions."""

    label: str
    seconds: float
    instructions: str
    model: str | None = None
    reasoning_effort: Effort | None = None
    max_usd: float | None = None
    # Interim candidates are recorded through record_candidate and survive a cut-off run.
    notes: bool = False
    # A helper call: an unset model means the configured helper model.
    helper: bool = False
    requests: int = HELPER_REQUESTS
    # The most candidates the call may note.
    candidates: int = 0


class ResearchError(MixhoundError):
    """An agent call failed; `code` names the reason."""

    def __init__(self, code: str, detail: str = "", notes: list[str] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail
        # Interim candidate notes (JSON) the research wrote before it was stopped.
        self.notes = notes or []


class ResearchAgents(Protocol):
    """Runs the research agents.

    Every call adds its usage and answer to the run state, failed or not, and raises
    ResearchError; a research cut off by a noted failure carries its notes.
    """

    async def research(self, state: RunState, call: AgentCall, prompt: str) -> ResearchOutput:
        """Research candidates."""
        ...

    async def profile(self, state: RunState, call: AgentCall, prompt: str) -> SonicProfile:
        """Write the references' sonic profile."""
        ...

    async def find_links(self, state: RunState, call: AgentCall, prompt: str) -> LinkSearchOutput:
        """Find track pages of known recordings."""
        ...

    async def orchestrate(
        self, state: RunState, call: AgentCall, prompt: str, workspace: AgenticWorkspace
    ) -> ResearchOutput:
        """Research candidates in agentic mode, seeding and verifying through the workspace."""
        ...


def _partial(state: RunState, error: ResearchError) -> ResearchOutput | None:
    """Candidates from a cut-off research's notes, unless the run was cancelled."""
    if error.code not in NOTED_FAILURES or not error.notes or state.cancelled():
        return None
    partial = notes_output(error.notes)
    return partial if partial.candidates else None


class ResearchSteps:
    """Agent calls of one run: the sonic profile, the research itself and link follow-up."""

    def __init__(
        self, agents: ResearchAgents, verifier: ResultVerifier, skills: SkillCatalog
    ) -> None:
        self.agents = agents
        self.verifier = verifier
        self.skills = skills

    def _research_call(self, state: RunState, label: str, *, strategy: bool) -> AgentCall:
        settings = state.run.input.settings
        budget = strategy_budget(settings.depth) if strategy else DEPTH_BUDGETS[settings.depth]
        return AgentCall(
            label,
            settings.minutes * 60,
            self.skills.research_instructions(strategy=strategy),
            settings.model,
            settings.reasoning_effort,
            strategy_max_usd(settings.max_usd) if strategy else settings.max_usd,
            notes=True,
            requests=budget.requests,
            candidates=budget.candidates,
        )

    async def research(self, state: RunState, prompt: str) -> ResearchOutput:
        """Run the main research; a cut-off answer falls back to its interim notes."""
        call = self._research_call(state, "research", strategy=False)
        try:
            return await self.agents.research(state, call, prompt + NOTES_REMINDER)
        except ResearchError as error:
            return self._noted(state, error)

    async def orchestrate(
        self, state: RunState, prompt: str, workspace: AgenticWorkspace
    ) -> ResearchOutput:
        """Run the orchestrator; a cut-off answer falls back to its interim notes."""
        call = replace(
            self._research_call(state, "research", strategy=False),
            instructions=self.skills.get(ORCHESTRATION_SKILL).instructions()
            + "\n"
            + self.skills.research_instructions(),
        )
        try:
            return await self.agents.orchestrate(state, call, prompt + NOTES_REMINDER, workspace)
        except ResearchError as error:
            return self._noted(state, error)

    @staticmethod
    def _noted(state: RunState, error: ResearchError) -> ResearchOutput:
        """The interim notes of a cut-off research, else the error raised again."""
        partial = _partial(state, error)
        if partial is None:
            raise error
        log.warning(
            "discovery_partial_research run=%s code=%s candidates=%d",
            state.run.id,
            error.code,
            len(partial.candidates),
        )
        state.run.partial = True
        state.detail = error.code
        return partial

    async def profile(
        self,
        state: RunState,
        session: DiscoverySession,
        preparation: Preparation | None,
        history: RunHistory,
    ) -> SonicProfile | None:
        """The session's previous profile, else one short profile-only pass."""
        profile = previous_profile(session, history)
        if profile is not None:
            return profile
        settings = state.run.input.settings
        call = AgentCall(
            "profile",
            HELPER_SECONDS,
            self.skills.get(PROFILE_SKILL).instructions(),
            settings.model,
            settings.reasoning_effort,
            PROFILE_MAX_USD,
        )
        try:
            return await self.agents.profile(state, call, profile_context(state.run, preparation))
        except ResearchError as error:
            state.check()
            log.warning("discovery_profile_failed run=%s code=%s", state.run.id, error.code)
            return None

    async def _strategy(
        self, state: RunState, name: str, prompt: str, call: AgentCall | None = None
    ) -> ResearchOutput:
        """One strategy's research; interim notes stand in for an answer cut off by a limit."""
        metrics = state.metrics.strategies[name] = StrategyMetrics()
        started = time.monotonic()
        call = call or self._research_call(state, name, strategy=True)
        try:
            output = await self.agents.research(state, call, prompt + NOTES_REMINDER)
        except ResearchError as error:
            metrics.failed = error.code
            partial = _partial(state, error)
            if partial is None:
                raise
            log.warning(
                "discovery_partial_strategy run=%s strategy=%s code=%s candidates=%d",
                state.run.id,
                name,
                error.code,
                len(partial.candidates),
            )
            state.run.partial = True
            state.detail = error.code
            output = partial
        finally:
            metrics.seconds = round(time.monotonic() - started, 2)
            usage = state.usage.get(name)
            if usage is not None:
                metrics.cost_usd = usage.cost_usd
                metrics.web_searches = usage.web_searches
        metrics.candidates = len(output.candidates)
        return output

    async def delegate(
        self, state: RunState, name: str, prompt: str, seconds: float, max_usd: float
    ) -> ResearchOutput:
        """One strategy's research delegated by the orchestrator, within the time and money left."""
        call = replace(
            self._research_call(state, name, strategy=True), seconds=seconds, max_usd=max_usd
        )
        return await self._strategy(state, name, prompt, call)

    async def _research_parallel(
        self,
        state: RunState,
        prompts: dict[str, str],
        profile: SonicProfile | None,
    ) -> ResearchOutput:
        """Run every strategy at once; the run fails only when all of them failed."""
        answers = await asyncio.gather(
            *(self._strategy(state, name, prompt) for name, prompt in prompts.items()),
            return_exceptions=True,
        )
        state.check()
        outputs: dict[str, ResearchOutput] = {}
        errors: list[tuple[str, BaseException]] = []
        for name, answer in zip(prompts, answers, strict=True):
            if isinstance(answer, ResearchOutput):
                outputs[name] = answer
            else:
                errors.append((name, answer))
        for name, error in errors:
            if not isinstance(error, ResearchError):
                log.error(
                    "discovery_strategy_crashed run=%s strategy=%s",
                    state.run.id,
                    name,
                    exc_info=error,
                )
                state.metrics.strategies[name].failed = "process_failure"
        if not outputs:
            raise errors[0][1]
        merged, state.metrics.overlap = merge_outputs(outputs, profile)
        return merged

    async def research_strategies(
        self,
        state: RunState,
        session: DiscoverySession,
        preparation: Preparation | None,
        history: RunHistory,
    ) -> ResearchOutput:
        """Experimental parallel mode: one profile, then one research per strategy."""
        run = state.run
        with state.enter("profile"):
            profile = await self.profile(state, session, preparation, history)
        with state.enter("context"):
            budget = strategy_budget(run.input.settings.depth)
            pools = split_pool(preparation.pool if preparation else [])
            prompts = {
                name: build_context(
                    run,
                    history,
                    preparation,
                    StrategyShare(name, description, profile, pools[name], budget, len(STRATEGIES)),
                )[0]
                for name, description in STRATEGIES.items()
            }
        with state.enter(RESEARCH_STAGE):
            return await self._research_parallel(state, prompts, profile)

    async def followup(self, state: RunState, result: DiscoveryResearchResult) -> None:
        """Ask a cheaper model for track pages the catalogs could not supply; verify each one."""
        missing = [track for track in missing_links(result) if likely_result(state, track)][
            :MAX_FOLLOWUP_TRACKS
        ]
        if not missing:
            return
        settings = state.run.input.settings
        call = AgentCall(
            "listening",
            HELPER_SECONDS,
            self.skills.get(LINK_SEARCH_SKILL).instructions(),
            settings.helper_model or settings.model,
            settings.helper_reasoning_effort,
            FOLLOWUP_MAX_USD,
            helper=True,
        )
        try:
            found = await self.agents.find_links(state, call, followup_prompt(missing))
        except ResearchError as error:
            state.check()
            log.warning(
                "discovery_listening_search_failed run=%s code=%s", state.run.id, error.code
            )
            return
        state.check()
        proposals: list[tuple[DiscoveryTrack, DiscoveryListeningLink]] = proposed_links(
            missing, found
        )
        await self.verifier.verify_proposed(state, proposals)
