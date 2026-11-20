import pytest
from pydantic_evals.evaluators import EvaluationReason
from pydantic_evals.otel import SpanTree

from evals.evaluators import (
    CatalogCheckedBeforeAnswer,
    ExclusionsRespected,
    GroundedUrls,
    KnownReferenceIds,
    Novelty,
    RunCompleted,
    ToolUsage,
    VerifiedCount,
    comparable_url,
    default_evaluators,
    research_answers,
)
from mixhound.models.domain import DiscoveryTrack, SonicProfile
from mixhound.models.research import ResearchCandidate, ResearchOutput, ResearchSource
from tests.support.evals import case, context, outcome, track
from tests.support.factories import candidate


def answer(*candidates: ResearchCandidate) -> str:
    return ResearchOutput(
        sonic_profile=SonicProfile(), candidates=list(candidates)
    ).model_dump_json()


def test_run_completed_passes_a_completed_run() -> None:
    inputs = case()
    result = RunCompleted().evaluate(context(inputs, outcome(inputs, status="completed")))
    assert result == EvaluationReason(value=True, reason=None)


def test_run_completed_names_the_failure() -> None:
    inputs = case()
    failed = outcome(
        inputs, status="failed", failure_code="timeout", partial=True, diagnostic={"detail": "slow"}
    )
    result = RunCompleted().evaluate(context(inputs, failed))
    assert result == EvaluationReason(value=False, reason="failure timeout, detail slow, partial")


def test_verified_count_scores_the_requested_share() -> None:
    inputs = case(desired_count=4)
    found = outcome(inputs, results=[track("Low", "Words")])
    assert VerifiedCount().evaluate(context(inputs, found)) == {
        "verified_share": 0.25,
        "any_verified": True,
    }


def test_exclusions_pass_when_every_rule_holds() -> None:
    inputs = case(year_from=1990, exclude_reference_artists=True)
    found = outcome(inputs, results=[track("Low", "Words", year=1994)])
    result = ExclusionsRespected().evaluate(context(inputs, found))
    assert isinstance(result, dict)
    assert all(isinstance(item, EvaluationReason) and item.value for item in result.values())


@pytest.mark.parametrize(
    ("settings", "shown", "rule", "reason"),
    [
        ({"excluded_artists": ["Low"]}, [track("Low", "Words")], "no_excluded_artist", "Low"),
        (
            {"exclude_reference_artists": True},
            [track("Codeine", "Loss Leader")],
            "no_excluded_artist",
            "Codeine — Loss Leader",
        ),
        ({}, [track("Codeine", "D")], "no_reference_recording", "Codeine — D"),
        ({"year_to": 2000}, [track("Low", "Words", year=2005)], "within_years", "(2005)"),
        (
            {"max_per_artist": 1},
            [track("Low", "Words"), track("Low", "Laser Beam")],
            "per_artist_cap",
            "low: 2",
        ),
    ],
)
def test_exclusions_report_the_offending_tracks(
    settings: dict[str, object], shown: list[DiscoveryTrack], rule: str, reason: str
) -> None:
    inputs = case(**settings)
    found = outcome(inputs, leads=shown)
    result = ExclusionsRespected().evaluate(context(inputs, found))
    assert isinstance(result, dict)
    item = result[rule]
    assert isinstance(item, EvaluationReason)
    assert item.value is False
    assert reason in (item.reason or "")


def test_known_reference_ids_flags_ids_in_answers_and_tracks() -> None:
    inputs = case()
    known = str(inputs.references[0].id)
    diagnostic = {
        "answer": answer(candidate(closest_reference_ids=[known])),
        "strategy_answers": {"traits": answer(candidate(title="B", closest_reference_ids=["x"]))},
    }
    found = outcome(inputs, diagnostic=diagnostic)
    result = KnownReferenceIds().evaluate(context(inputs, found))
    assert result.value is False
    assert result.reason == "Unknown reference IDs: Fogh Depot — B: x"


def test_known_reference_ids_passes_cited_references() -> None:
    inputs = case()
    shown = track("Low", "Words", closest_reference_ids=[inputs.references[0].id])
    result = KnownReferenceIds().evaluate(context(inputs, outcome(inputs, results=[shown])))
    assert result.value is True


def test_research_answers_skips_text_that_does_not_parse() -> None:
    diagnostic = {"answer": "not json", "strategy_answers": {"traits": answer(candidate())}}
    assert [len(item.candidates) for item in research_answers(diagnostic)] == [1]


def test_novelty_scores_distinct_and_new_artists() -> None:
    inputs = case()
    found = outcome(
        inputs,
        results=[track("Low", "Words"), track("Low", "Laser Beam"), track("Codeine", "Cave-In")],
    )
    assert Novelty().evaluate(context(inputs, found)) == {
        "distinct_artist_share": 0.67,
        "new_artist_share": 0.67,
    }
    assert Novelty().evaluate(context(inputs, outcome(inputs))) == {}


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.Example.com/a/#top", "example.com/a"),
        ("http://example.com/a?b=1", "example.com/a?b=1"),
    ],
)
def test_comparable_url_drops_scheme_www_fragment_and_slash(url: str, expected: str) -> None:
    assert comparable_url(url) == expected


def test_span_evaluators_skip_runs_without_spans() -> None:
    inputs = case(mode="agentic")
    diagnostic = {"answer": answer(candidate(listening_urls=["https://x.example/t"]))}
    ctx = context(inputs, outcome(inputs, diagnostic=diagnostic))
    assert GroundedUrls().evaluate(ctx) == {}
    assert CatalogCheckedBeforeAnswer().evaluate(ctx) == {}
    assert ToolUsage().evaluate(ctx) == {}


def test_grounded_urls_lists_urls_no_tool_returned() -> None:
    inputs = case()
    source = ResearchSource(url="https://reviews.example/a", claim="Slow", target_type="review")
    diagnostic = {"answer": answer(candidate(sources=[source]))}
    result = GroundedUrls().evaluate(
        context(inputs, outcome(inputs, diagnostic=diagnostic), spans=SpanTree())
    )
    assert result == {
        "grounded_url_share": EvaluationReason(
            value=0.0, reason="Not returned by any tool: https://reviews.example/a"
        )
    }


def test_catalog_check_is_not_scored_in_pipeline_mode() -> None:
    inputs = case(mode="pipeline")
    ctx = context(inputs, outcome(inputs), spans=SpanTree())
    assert CatalogCheckedBeforeAnswer().evaluate(ctx) == {}


def test_catalog_check_fails_an_agentic_answer_without_a_check() -> None:
    inputs = case(mode="agentic")
    diagnostic = {"answer": answer(candidate())}
    ctx = context(inputs, outcome(inputs, diagnostic=diagnostic), spans=SpanTree())
    assert CatalogCheckedBeforeAnswer().evaluate(ctx) == {
        "catalog_checked": False,
        "checked_share": EvaluationReason(
            value=0.0, reason="Answered unchecked: Fogh Depot — Anticyclone"
        ),
    }


def test_default_evaluators_have_distinct_types() -> None:
    names = [type(item).__name__ for item in default_evaluators()]
    assert len(names) == len(set(names)) == 8
