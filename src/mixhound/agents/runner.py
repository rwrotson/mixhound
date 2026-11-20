from typing import Protocol

from pydantic import BaseModel
from pydantic_ai import Agent, UsageLimits, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from mixhound.agents.failures import failure_code
from mixhound.agents.link_finder import build_link_finder
from mixhound.agents.orchestrator import build_orchestrator
from mixhound.agents.profiler import build_profiler
from mixhound.agents.researcher import build_researcher
from mixhound.models.domain import SonicProfile
from mixhound.models.research import LinkSearchOutput, ResearchOutput
from mixhound.services.agentic import AgenticWorkspace
from mixhound.services.research import AgentCall, Effort, ResearchError
from mixhound.services.run_state import LlmUsage, RunState
from mixhound.tools.deps import Toolbox, ToolDeps

MAX_DETAIL = 500


class Models(Protocol):
    """Resolves the models and request settings a call names."""

    def model(self, name: str | None = None, *, helper: bool = False) -> Model:
        """Return the named model, else the configured default."""
        ...

    def settings(self, effort: Effort | None) -> ModelSettings:
        """Return the request settings of a reasoning effort."""
        ...


def _answer(messages: list[ModelMessage]) -> str:
    """The last response's text or tool-call arguments, for the diagnostic."""
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            parts = [
                part.content if isinstance(part, TextPart) else part.args_as_json_str()
                for part in message.parts
                if isinstance(part, TextPart | ToolCallPart)
            ]
            return "\n".join(parts)
    return ""


def _usage(usage: RunUsage, deps: ToolDeps) -> LlmUsage:
    return LlmUsage(
        cost_usd=float(usage.cost) if usage.cost is not None else None,
        # pydantic-ai counts cache reads as input; V1 metrics keep them apart.
        input_tokens=max(usage.input_tokens - usage.cache_read_tokens, 0),
        cached_input_tokens=usage.cache_read_tokens,
        output_tokens=usage.output_tokens,
        web_searches=deps.searches,
        turns=usage.requests,
        failed_fetches=deps.failed_fetches,
    )


class PydanticResearchAgents:
    """Runs the research agents on pydantic-ai, bounded by each call's time and limits."""

    def __init__(self, models: Models, toolbox: Toolbox) -> None:
        self.models = models
        self.toolbox = toolbox
        self.researcher = build_researcher()
        self.profiler = build_profiler()
        self.link_finder = build_link_finder()
        self.orchestrator = build_orchestrator()

    async def research(self, state: RunState, call: AgentCall, prompt: str) -> ResearchOutput:
        """Research candidates; the research cannot work without web search."""
        if self.toolbox.search is None:
            raise ResearchError("search_unavailable", "No web search is configured")
        return await self._run(self.researcher, state, call, prompt)

    async def orchestrate(
        self, state: RunState, call: AgentCall, prompt: str, workspace: AgenticWorkspace
    ) -> ResearchOutput:
        """Research in agentic mode; the orchestrator cannot work without web search either."""
        if self.toolbox.search is None:
            raise ResearchError("search_unavailable", "No web search is configured")
        return await self._run(self.orchestrator, state, call, prompt, workspace)

    async def profile(self, state: RunState, call: AgentCall, prompt: str) -> SonicProfile:
        """Write the references' sonic profile."""
        return await self._run(self.profiler, state, call, prompt)

    async def find_links(self, state: RunState, call: AgentCall, prompt: str) -> LinkSearchOutput:
        """Find track pages of known recordings."""
        return await self._run(self.link_finder, state, call, prompt)

    async def _run[Output: BaseModel](
        self,
        agent: Agent[ToolDeps, Output],
        state: RunState,
        call: AgentCall,
        prompt: str,
        workspace: AgenticWorkspace | None = None,
    ) -> Output:
        """Run one call; record its usage and answer, and raise failures as ResearchError."""
        deps = ToolDeps(state, call, self.toolbox, workspace=workspace)
        usage = RunUsage()
        with capture_run_messages() as messages:
            try:
                result = await agent.run(
                    prompt,
                    deps=deps,
                    model=self.models.model(call.model, helper=call.helper),
                    model_settings=self.models.settings(call.reasoning_effort),
                    instructions=call.instructions,
                    usage_limits=UsageLimits(request_limit=call.requests),
                    usage=usage,
                    timeout=call.seconds,
                )
            except Exception as error:
                state.answers[call.label] = _answer(messages)
                raise ResearchError(
                    failure_code(error), str(error)[-MAX_DETAIL:], notes=list(deps.notes)
                ) from error
            finally:
                state.add_usage(call.label, _usage(usage, deps))
        state.answers[call.label] = result.output.model_dump_json()
        return result.output
