from collections.abc import Iterator

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset

from evals import tracing
from evals.cases import CaseMetadata, DiscoveryCase, RunOutcome
from evals.evaluators import default_evaluators
from evals.task import DiscoveryTask, apply_overrides
from mixhound.agents.runner import PydanticResearchAgents
from mixhound.config import PACKAGED_SKILLS_DIR
from mixhound.infrastructure.cache import MemoryCache
from mixhound.infrastructure.skills import load_catalog
from mixhound.models.domain import DiscoverySettings, SonicProfile
from mixhound.models.research import ResearchOutput, ResearchSource
from mixhound.services.audio import AudioMatcher
from mixhound.services.pages import PageInspector
from mixhound.services.pipeline import ResearchPipeline
from mixhound.services.ports import CatalogPage, CatalogRecording, SearchHit
from mixhound.services.research import Effort, ResearchSteps
from mixhound.services.seed import Seeder
from mixhound.services.verification import ResultVerifier
from mixhound.tools.deps import Toolbox
from tests.fakes import FakeCatalog, FakeFetch
from tests.support.agents import FakeSearch
from tests.support.evals import case
from tests.support.factories import candidate

SKILLS = load_catalog(PACKAGED_SKILLS_DIR)
REVIEW = "https://reviews.example/low-words"
GUESSED = "https://guessed.example/low-words"
WORDS = CatalogRecording(
    artist="Low",
    title="Words",
    confirmed_by=("musicbrainz",),
    mbid="mbid-words",
    year=1994,
    pages=(CatalogPage("musicbrainz", "https://musicbrainz.org/recording/mbid-words"),),
)
FOUND = candidate(
    artist="Low",
    title="Words",
    year=1994,
    sources=[ResearchSource(url=REVIEW, claim="Slow, brushed drums", target_type="review")],
    listening_urls=[GUESSED],
)
ANSWER = ResearchOutput(sonic_profile=SonicProfile(summary="Hushed"), candidates=[FOUND])


class FixedModels:
    def __init__(self, model: Model) -> None:
        self.chosen = model

    def model(self, name: str | None = None, *, helper: bool = False) -> Model:
        return self.chosen

    def settings(
        self, effort: Effort | None, name: str | None = None, *, helper: bool = False
    ) -> ModelSettings:
        return ModelSettings()


async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Search, then check the candidate (agentic) or read its review (pipeline), then answer."""
    output = info.output_tools[0]
    if "links" in output.parameters_json_schema["properties"]:
        return ModelResponse(parts=[ToolCallPart(output.name, {"links": []})])
    agentic = any(tool.name == "verify_candidates" for tool in info.function_tools)
    second = (
        ("verify_candidates", {"candidates": [FOUND.model_dump(mode="json")]})
        if agentic
        else ("fetch_page", {"url": REVIEW})
    )
    steps = [
        ("search_web", {"query": "low words"}),
        second,
        (output.name, ANSWER.model_dump(mode="json")),
    ]
    name, args = steps[sum(isinstance(message, ModelResponse) for message in messages)]
    return ModelResponse(parts=[ToolCallPart(name, args)])


def pipeline() -> ResearchPipeline:
    catalog = FakeCatalog({("Low", "Words"): WORDS})
    fetch = FakeFetch(
        {"reviews.example/low-words": b"<html><body>Low - Words: slow.</body></html>"}
    )
    verifier = ResultVerifier(catalog, PageInspector(fetch), MemoryCache())
    search = FakeSearch([SearchHit("Low - Words review", REVIEW, "Slow and hushed")])
    agents = PydanticResearchAgents(
        FixedModels(FunctionModel(script)), Toolbox(fetch, catalog, search)
    )
    return ResearchPipeline(
        Seeder(catalog, None), verifier, AudioMatcher(None), ResearchSteps(agents, verifier, SKILLS)
    )


@pytest.fixture
def traced() -> Iterator[None]:
    tracing.enable()
    yield
    tracing.disable()


def dataset(inputs: DiscoveryCase) -> Dataset[DiscoveryCase, RunOutcome, CaseMetadata]:
    return Dataset[DiscoveryCase, RunOutcome, CaseMetadata](
        name="smoke",
        cases=[Case(name="slowcore", inputs=inputs)],
        evaluators=default_evaluators(),
    )


@pytest.mark.usefixtures("traced")
@pytest.mark.parametrize("mode", ["pipeline", "agentic"])
async def test_task_runs_a_case_and_the_evaluators_score_it(mode: str) -> None:
    inputs = case(year_from=1990, exclude_reference_artists=True, desired_count=2)
    task = DiscoveryTask(pipeline(), {"mode": mode, "depth": None})

    report = await dataset(inputs).evaluate(task, progress=False)

    assert report.failures == []
    [result] = report.cases
    assert result.evaluator_failures == []
    assert result.output.run.input.settings.mode == mode
    assert result.attributes["status"] == "completed"
    assert result.attributes["tracks"] == ["✓ Low — Words (70/None)"]
    assert result.metrics["verified"] == 1
    assert result.metrics["verified_share"] == 0.5
    assert all(item.value for item in result.assertions.values())
    grounded = result.scores["grounded_url_share"]
    assert grounded.value == 0.5
    assert GUESSED in (grounded.reason or "")
    assert result.scores["calls_search_web"].value == 1
    if mode == "agentic":
        assert result.scores["checked_share"].value == 1.0
        assert result.assertions["catalog_checked"].value is True
    else:
        assert "checked_share" not in result.scores


def test_apply_overrides_keeps_settings_left_unset() -> None:
    settings = DiscoverySettings(depth="deep", minutes=20)
    changed = apply_overrides(settings, {"depth": "quick", "minutes": None, "mode": "agentic"})
    assert (changed.depth, changed.minutes, changed.mode) == ("quick", 20, "agentic")
