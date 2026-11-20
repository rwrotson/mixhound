import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from mixhound.models.domain import SonicProfile
from mixhound.models.research import ResearchOutput
from mixhound.services.research import ResearchError
from mixhound.services.run_state import LlmUsage
from mixhound.tools.deps import Toolbox
from tests.fakes import FakeCatalog, FakeFetch
from tests.support.agentic import LOW, workspace
from tests.support.agents import call
from tests.support.factories import candidate
from tests.unit.agents.test_runner import (
    NOTE,
    OUTPUT,
    PricedModel,
    agents,
    final,
    respond,
    tools,
)

PROFILE = SonicProfile(summary="Hushed").model_dump(mode="json")
DELEGATED = ResearchOutput(sonic_profile=SonicProfile(), candidates=[candidate(title="Far")])


async def test_orchestrator_seeds_delegates_and_verifies_through_its_tools() -> None:
    script, seen = respond(
        tools(
            ("lookup_reference", {"reference_id": str(LOW.id)}),
            ("delegate_research", {"strategy": "traits", "profile": PROFILE}),
        ),
        tools(
            ("verify_candidates", {"candidates": [candidate().model_dump(mode="json")]}),
            NOTE,
        ),
        final(),
    )
    runner, _ = agents(FunctionModel(script))
    space, _, fake = workspace(outcomes={"traits": DELEGATED})
    state = space.state

    answer = await runner.orchestrate(state, call(seconds=600), "Orchestrate.", space)

    assert answer == OUTPUT
    assert "<catalog_facts" in seen[1]
    assert "Strategy traits kept 1 candidates" in seen[1]
    assert "Anticyclone: not confirmed by the catalogs." in seen[2]
    assert "Clock: 0:00 of 10:00 used" in seen[2]
    assert space.delegated == {"traits": DELEGATED}
    assert [agent_call.label for agent_call, _ in fake.calls] == ["traits"]
    assert state.answers["research"] == OUTPUT.model_dump_json()
    assert state.counts["research"] == (0, 1)


async def test_orchestrator_needs_web_search() -> None:
    runner, _ = agents(FunctionModel(respond(final())[0]))
    runner.toolbox = Toolbox(FakeFetch({}), FakeCatalog())
    space, _, _ = workspace()

    with pytest.raises(ResearchError) as caught:
        await runner.orchestrate(space.state, call(), "Orchestrate.", space)

    assert caught.value.code == "search_unavailable"


async def costly(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    response = tools(NOTE)(info)
    response.usage = RequestUsage(input_tokens=100_000)
    return response


async def test_spending_cap_covers_finished_delegated_researches() -> None:
    runner, _ = agents(PricedModel(costly, model_name="claude-sonnet-4-5"))
    space, _, _ = workspace()
    space.state.usage["traits"] = LlmUsage(cost_usd=0.3)

    with pytest.raises(ResearchError) as caught:
        await runner.orchestrate(space.state, call(max_usd=0.5), "Orchestrate.", space)

    assert caught.value.code == "budget_exceeded"
    assert space.state.usage["research"].turns == 1
