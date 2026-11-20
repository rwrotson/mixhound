import asyncio
import json
from collections.abc import Awaitable, Callable

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from mixhound.agents.runner import PydanticResearchAgents
from mixhound.models.domain import SonicProfile
from mixhound.models.research import LinkSearchOutput, ResearchOutput
from mixhound.services.ports import SearchHit
from mixhound.services.research import AgentCall, Effort, ResearchError
from mixhound.tools.deps import Toolbox, ToolDeps
from tests.fakes import FakeCatalog, FakeFetch
from tests.support.agents import FakeSearch, call, tool_deps
from tests.support.factories import candidate

type Script = Callable[[list[ModelMessage], AgentInfo], Awaitable[ModelResponse]]
OUTPUT = ResearchOutput(sonic_profile=SonicProfile(summary="Hushed"), candidates=[candidate()])


class FakeModels:
    def __init__(self, model: Model) -> None:
        self.chosen = model
        self.asked: list[tuple[str | None, bool, Effort | None]] = []

    def model(self, name: str | None = None, *, helper: bool = False) -> Model:
        self.asked.append((name, helper, None))
        return self.chosen

    def settings(self, effort: Effort | None) -> ModelSettings:
        self.asked.append((None, False, effort))
        return ModelSettings()


class PricedModel(FunctionModel):
    """A function model priced like a real Anthropic model."""

    @property
    def system(self) -> str:
        return "anthropic"


def agents(
    model: Model, *, search: FakeSearch | None = None
) -> tuple[PydanticResearchAgents, FakeModels]:
    models = FakeModels(model)
    toolbox = Toolbox(FakeFetch({}), FakeCatalog(), search or FakeSearch([SearchHit("T", "u")]))
    return PydanticResearchAgents(models, toolbox), models


def run_state() -> ToolDeps:
    return tool_deps()


def texts(messages: list[ModelMessage]) -> str:
    return json.dumps(
        [
            str(getattr(part, "content", ""))
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ]
    )


def respond(*steps: Callable[[AgentInfo], ModelResponse]) -> tuple[Script, list[str]]:
    """A model that plays `steps` in order and records what each request showed."""
    seen: list[str] = []

    async def play(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(texts(messages))
        return steps[len(seen) - 1](info)

    return play, seen


def tools(*calls: tuple[str, dict[str, object]]) -> Callable[[AgentInfo], ModelResponse]:
    return lambda _: ModelResponse(parts=[ToolCallPart(name, args) for name, args in calls])


def final(answer: ResearchOutput = OUTPUT) -> Callable[[AgentInfo], ModelResponse]:
    return lambda info: ModelResponse(
        parts=[ToolCallPart(info.output_tools[0].name, answer.model_dump(mode="json"))]
    )


NOTE: tuple[str, dict[str, object]] = (
    "record_candidate",
    {"candidate": candidate(title="Noted").model_dump(mode="json")},
)
SEARCH: tuple[str, dict[str, object]] = ("search_web", {"query": "low words"})


async def test_research_runs_tools_shows_the_clock_and_records_usage() -> None:
    script, seen = respond(tools(SEARCH, NOTE), final())
    runner, models = agents(FunctionModel(script))
    state = run_state().state

    answer = await runner.research(state, call(), "Find recordings.")

    assert answer == OUTPUT
    assert "Clock:" not in seen[0]
    assert "Clock: 0:00 of 1:00 used" in seen[1]
    assert "Notes written: 1." in seen[1]
    assert "<search_results" in seen[1]
    assert models.asked == [("test", False, None), (None, False, None)]
    assert state.answers["research"] == OUTPUT.model_dump_json()
    usage = state.usage["research"]
    assert (usage.turns, usage.web_searches) == (2, 1)
    assert state.metrics.llm_calls == 1
    assert state.counts["research"] == (1, 1)


async def test_timeout_keeps_the_notes_written_so_far() -> None:
    async def stall(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            await asyncio.sleep(5)
        return tools(SEARCH, NOTE)(info)

    runner, _ = agents(FunctionModel(stall))
    state = run_state().state

    with pytest.raises(ResearchError) as caught:
        await runner.research(state, call(seconds=0.2), "Find recordings.")

    assert caught.value.code == "timeout"
    assert [json.loads(note)["title"] for note in caught.value.notes] == ["Noted"]
    assert '"Noted"' in state.answers["research"]
    assert state.usage["research"].turns == 1


async def test_request_limit_stops_the_call() -> None:
    async def endless(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return tools(SEARCH)(info)

    runner, _ = agents(FunctionModel(endless))
    state = run_state().state

    with pytest.raises(ResearchError) as caught:
        await runner.research(state, call(), "Find recordings.")

    assert caught.value.code == "limit_reached"
    assert state.usage["research"].turns == 5


async def test_spending_cap_stops_the_call() -> None:
    async def costly(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        response = tools(NOTE)(info)
        response.usage = RequestUsage(input_tokens=1_000_000)
        return response

    runner, _ = agents(PricedModel(costly, model_name="claude-sonnet-4-5"))
    state = run_state().state

    with pytest.raises(ResearchError) as caught:
        await runner.research(state, call(max_usd=0.5), "Find recordings.")

    assert caught.value.code == "budget_exceeded"
    assert len(caught.value.notes) == 1
    cost = state.usage["research"].cost_usd
    assert cost is not None
    assert cost >= 0.5


async def test_model_errors_become_research_errors() -> None:
    async def limited(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(429, "test", body="slow down")

    runner, _ = agents(FunctionModel(limited))
    state = run_state().state

    with pytest.raises(ResearchError) as caught:
        await runner.profile(state, call(label="profile", notes=False), "Profile.")

    assert caught.value.code == "limit_reached"
    assert caught.value.notes == []
    assert state.answers["profile"] == ""
    assert state.metrics.llm_calls == 1


async def test_research_needs_web_search() -> None:
    runner, models = agents(TestModel())
    runner.toolbox = Toolbox(FakeFetch({}), FakeCatalog())
    state = run_state().state

    with pytest.raises(ResearchError, match="search_unavailable"):
        await runner.research(state, call(), "Find recordings.")

    assert models.asked == []
    assert state.metrics.llm_calls == 0


async def test_helpers_answer_in_their_own_schema() -> None:
    script, _ = respond(
        lambda info: ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"links": []})])
    )
    runner, models = agents(FunctionModel(script))
    state = run_state().state
    helper = AgentCall("listening", 60, "Find links.", helper=True)

    found = await runner.find_links(state, helper, "Find pages.")

    assert found == LinkSearchOutput(links=[])
    assert models.asked[0] == (None, True, None)
    assert state.answers["listening"] == '{"links":[]}'


async def test_profiler_runs_with_its_tools() -> None:
    runner, _ = agents(TestModel())
    state = run_state().state

    profile = await runner.profile(state, call(label="profile", notes=False), "Profile.")

    assert isinstance(profile, SonicProfile)
    assert state.usage["profile"].web_searches >= 1


async def test_text_answer_is_kept_when_the_output_is_invalid() -> None:
    script, _ = respond(*([lambda _: ModelResponse(parts=[TextPart("no tool")])] * 5))
    runner, _ = agents(FunctionModel(script))
    state = run_state().state

    with pytest.raises(ResearchError) as caught:
        await runner.profile(state, call(label="profile", notes=False), "Profile.")

    assert caught.value.code == "invalid_result"
    assert state.answers["profile"] == "no tool"
