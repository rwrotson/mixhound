import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Literal

from mixhound.memory.context import RunHistory
from mixhound.models.domain import (
    DiscoveryResearchResult,
    DiscoverySession,
    SonicProfile,
    now_utc,
)
from mixhound.models.research import ResearchOutput
from mixhound.services.agentic import ORCHESTRATION_NOTE, AgenticWorkspace, RunServices
from mixhound.services.audio import AudioMatcher
from mixhound.services.budgets import DEPTH_BUDGETS
from mixhound.services.context import build_context, exclusions_from_history, research_target
from mixhound.services.evidence import (
    ResearchEvidenceError,
    has_sonic_evidence,
    merge_outputs,
    publish_result,
    research_result,
    screen_research_result,
    trim_to_target,
)
from mixhound.services.exclusions import Exclusions
from mixhound.services.listening import missing_links
from mixhound.services.preparation import Preparation
from mixhound.services.research import NOTED_FAILURES, ResearchError, ResearchSteps
from mixhound.services.run_state import RESEARCH_STAGE, RunCancelledError, RunState
from mixhound.services.seed import Seeder
from mixhound.services.strategies import STRATEGIES
from mixhound.services.verification import ResultVerifier, promote_album_sources

log = logging.getLogger(__name__)


class ResearchPipeline:
    """One run's stages in pipeline mode: prepare, research, verify, publish."""

    def __init__(
        self,
        seeder: Seeder,
        verifier: ResultVerifier,
        audio_matcher: AudioMatcher,
        steps: ResearchSteps,
    ) -> None:
        self.seeder = seeder
        self.verifier = verifier
        self.audio_matcher = audio_matcher
        self.steps = steps

    async def execute(
        self, state: RunState, session: DiscoverySession, history: RunHistory
    ) -> dict[str, object]:
        """Run every stage, record on the run how it ended and return its diagnostic.

        Storing the run and the diagnostic is left to the caller.
        """
        state.run.status = "running"
        try:
            await self._stages(state, session, history)
        except (
            RunCancelledError,
            asyncio.CancelledError,
            ResearchError,
            ResearchEvidenceError,
        ) as error:
            _record_failure(state, error)
        except Exception as error:
            log.exception("discovery_run_failed run=%s", state.run.id)
            _record_failure(state, error)
        return _finish(state)

    async def _stages(
        self, state: RunState, session: DiscoverySession, history: RunHistory
    ) -> None:
        """Prepare the run, research it, verify the results and publish them."""
        run = state.run
        if run.input.settings.mode == "agentic":
            await self._agentic(state, history)
            return
        with state.enter("prepare"):
            exclusions = exclusions_from_history(history)
            preparation = await self._prepare(state, exclusions)
        output: ResearchOutput
        if run.input.settings.parallel_strategies:
            output = await self.steps.research_strategies(state, session, preparation, history)
        else:
            with state.enter("context"):
                prompt, exclusions = build_context(run, history, preparation)
            with state.enter(RESEARCH_STAGE):
                output = await self.steps.research(state, prompt)
        await self.finalize(state, output, preparation, exclusions)

    async def _agentic(self, state: RunState, history: RunHistory) -> None:
        """Let the orchestrator seed, research and check, then verify and publish as usual."""
        run = state.run
        with state.enter("prepare"):
            workspace = AgenticWorkspace(
                state,
                history,
                exclusions_from_history(history),
                RunServices(self.seeder, self.verifier, self.audio_matcher, self.steps),
            )
        with state.enter("context"):
            prompt, exclusions = build_context(run, history)
        with state.enter(RESEARCH_STAGE):
            output = await self._orchestrate(state, prompt + ORCHESTRATION_NOTE, workspace)
        await self.finalize(
            state, output, workspace.preparation, exclusions, complete=workspace.complete
        )

    async def _orchestrate(
        self, state: RunState, prompt: str, workspace: AgenticWorkspace
    ) -> ResearchOutput:
        """The orchestrator's answer merged with its delegated researches.

        Delegated answers outlive an orchestrator cut off by a limit.
        """
        try:
            own = await self.steps.orchestrate(state, prompt, workspace)
        except ResearchError as error:
            if error.code not in NOTED_FAILURES or not workspace.delegated or state.cancelled():
                raise
            log.warning("discovery_orchestrator_cut_off run=%s code=%s", state.run.id, error.code)
            state.run.partial = True
            state.detail = error.code
            own = ResearchOutput(sonic_profile=SonicProfile(), candidates=[])
        if not workspace.delegated:
            return own
        merged, state.metrics.overlap = merge_outputs({RESEARCH_STAGE: own, **workspace.delegated})
        return merged

    async def finalize(
        self,
        state: RunState,
        output: ResearchOutput,
        preparation: Preparation,
        exclusions: Exclusions,
        *,
        complete: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Screen, verify and publish a research answer; the same in both modes.

        `complete` fills reference facts the research skipped before the audio comparison.
        """
        run = state.run
        result = research_result(output, run.input)
        state.metrics.candidates = len(result.results) + len(result.unverified_leads)
        state.metrics.sonic_evidence = sum(
            has_sonic_evidence(track) for track in result.results + result.unverified_leads
        )
        await self._verify(state, result, preparation, exclusions, complete)
        state.check()
        run.results = result.results
        run.unverified_leads = result.unverified_leads
        run.sonic_profile = result.sonic_profile
        run.status = "completed"

    async def _prepare(self, state: RunState, exclusions: Exclusions) -> Preparation:
        """Gather catalog facts, reference audio and a candidate pool."""
        run_input = state.run.input
        preparation = await self.seeder.prepare(
            run_input,
            exclusions,
            DEPTH_BUDGETS[run_input.settings.depth].pool,
            state.check,
        )
        state.metrics.pool_size = len(preparation.pool)
        return preparation

    async def _verify(
        self,
        state: RunState,
        result: DiscoveryResearchResult,
        preparation: Preparation,
        exclusions: Exclusions,
        complete: Callable[[], Awaitable[None]] | None,
    ) -> None:
        """Screen, confirm, link, listen to and compare the results, then publish them."""
        run_input = state.run.input
        with state.enter("evidence"):
            state.rejections += screen_research_result(result, run_input, exclusions)
            budget = DEPTH_BUDGETS[run_input.settings.depth]
            state.rejections += trim_to_target(
                result,
                research_target(run_input.settings.desired_count, budget.candidates),
            )
        with state.enter("catalog"):
            # The catalog request also supplies the recording's service links (V1 expand_links
            # and search_youtube), so the link search below only promotes album pages.
            await self.verifier.confirm_catalog(state, result)
            state.rejections += screen_research_result(result, run_input, exclusions)
        with state.enter("independent_verification"):
            await self.verifier.confirm_sources(state, result)
        with state.enter("link_search"):
            promote_album_sources(result)
        if missing_links(result):
            with state.enter("listening"):
                await self.steps.followup(state, result)
        with state.enter("listening_verification"):
            await self.verifier.confirm_links(state, result)
        with state.enter("audio"):
            if complete is not None:
                await complete()
            await self.audio_matcher.compare(state, result, preparation)
        with state.enter("publish"):
            state.rejections += publish_result(result, run_input, state.confirmed)


def _finish(state: RunState) -> dict[str, object]:
    """Complete the run's metrics and return its diagnostic."""
    run, metrics = state.run, state.metrics
    run.finished_at = now_utc()
    metrics.verified = len(run.results)
    metrics.leads = len(run.unverified_leads)
    metrics.unresearched = sum(not track.researched for track in run.unverified_leads)
    metrics.dropped = sum(item.action == "dropped" for item in state.rejections)
    metrics.playable = sum(
        any(link.playback_available is True for link in track.listening_links)
        for track in run.results
    )
    run.metrics = metrics
    diagnostic: dict[str, object] = {
        "stage": state.stage if run.status == "failed" else run.status,
        "answer": state.answers.get("research", ""),
        "followup_answer": state.answers.get("listening", ""),
    }
    strategy_answers = {name: state.answers[name] for name in STRATEGIES if name in state.answers}
    if strategy_answers:
        diagnostic["strategy_answers"] = strategy_answers
    if run.status == "failed":
        run.failure_message = (
            state.detail[:500] if state.detail else (run.failure_code or "Research failed")
        )
        diagnostic |= {"failure_code": run.failure_code, "detail": state.detail}
    settings = run.input.settings
    return {
        "run_id": str(run.id),
        "session_id": str(run.session_id),
        "mode": settings.mode,
        "model": settings.model,
        "finished_at": run.finished_at.isoformat(),
        **diagnostic,
        "partial": run.partial,
        "metrics": metrics.model_dump(mode="json"),
        "rejections": [item.as_dict() for item in state.rejections],
    }


def _record_failure(state: RunState, error: BaseException) -> None:
    """Record on the run how it ended early: cancelled, or failed with a code and detail."""
    status: Literal["failed", "cancelled"]
    code: str | None
    if isinstance(error, RunCancelledError | asyncio.CancelledError) or (
        isinstance(error, ResearchError) and state.cancelled()
    ):
        status, code = "cancelled", None
    elif isinstance(error, ResearchError):
        status, code, state.detail = "failed", error.code, error.detail
    elif isinstance(error, ResearchEvidenceError):
        status, code, state.detail = "failed", "invalid_result", str(error)
    else:
        status, code, state.detail = "failed", "process_failure", f"{type(error).__name__}: {error}"
    state.run.status = status
    if code is not None:
        state.run.failure_code = code
