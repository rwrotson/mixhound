from uuid import uuid4

import pytest

from mixhound.models.domain import DiscoveryRun, DiscoveryRunInput
from mixhound.services.run_state import (
    STAGE_ORDER,
    LlmUsage,
    RunCancelledError,
    RunState,
)


def state() -> RunState:
    return RunState(DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput()))


def test_stages_accumulate_wall_time_even_when_they_fail() -> None:
    current = state()
    with current.enter("catalog"):
        pass
    with pytest.raises(ValueError, match="x"), current.enter("catalog"):
        raise ValueError("x")

    assert current.stage == "catalog"
    assert set(current.metrics.stages) == {"catalog"}
    assert current.research_started is None


def test_cancelled_run_stops_before_a_stage() -> None:
    current = RunState(
        DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput()), cancelled=lambda: True
    )
    with pytest.raises(RunCancelledError), current.enter("prepare"):
        pass
    assert current.metrics.stages == {}


def test_progress_reports_stage_research_time_and_live_counts() -> None:
    current = state()
    assert current.progress(600).step == 1
    assert current.progress(600).elapsed == 0.0

    with current.enter("research"):
        current.listener("collaborators")(4, 2)
        current.listener("traits")(1, 0)
        current.listener("traits")(3, 1)
        progress = current.progress(600)

    assert progress.stage == "research"
    assert progress.step == STAGE_ORDER.index("research") + 1
    assert progress.steps == len(STAGE_ORDER)
    assert progress.timeout == 600
    assert progress.elapsed >= 0
    assert (progress.searches, progress.candidates) == (7, 3)
    current.stage = "unknown"
    assert current.progress(1).step == 0


def test_usage_adds_up_and_helpers_are_not_research_searches() -> None:
    current = state()
    current.add_usage(
        "research",
        LlmUsage(
            cost_usd=0.5,
            input_tokens=10,
            cached_input_tokens=4,
            output_tokens=3,
            web_searches=6,
            turns=2,
            failed_fetches=1,
        ),
    )
    current.add_usage("listening", LlmUsage(web_searches=2, turns=1))
    current.add_usage("profile", LlmUsage(cost_usd=0.25))

    metrics = current.metrics
    assert metrics.llm_calls == 3
    assert metrics.cost_usd == 0.75
    assert (metrics.input_tokens, metrics.cached_input_tokens, metrics.output_tokens) == (10, 4, 3)
    assert (metrics.web_searches, metrics.research_searches) == (8, 6)
    assert (metrics.turns, metrics.failed_fetches) == (3, 1)
    assert current.usage["listening"].web_searches == 2
