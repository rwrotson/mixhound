from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import EvaluationReason

from evals.judge import InstructionFit, case_text, tracks_text
from tests.support.evals import case, context, outcome, track

SEEN: list[str] = []


async def grade(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    SEEN.append(str(messages[-1]))
    verdict = {"reason": "Hushed and slow", "pass": True, "score": 0.8}
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, verdict)])


async def test_instruction_fit_grades_the_published_tracks() -> None:
    inputs = case()
    found = outcome(inputs, results=[track("Low", "Words", year=1994, genres=["slowcore"])])
    judge = InstructionFit(FunctionModel(grade))

    result = await judge.evaluate(context(inputs, found))

    assert result == {
        "instruction_fit": EvaluationReason(value=0.8, reason="Hushed and slow"),
        "instruction_fit_pass": True,
    }
    assert "Low — Words (1994, slowcore)" in SEEN[-1]
    assert judge.build_serialization_arguments() == {"model": "function:grade:"}


async def test_instruction_fit_skips_runs_without_results() -> None:
    inputs = case()
    assert (
        await InstructionFit(FunctionModel(grade)).evaluate(context(inputs, outcome(inputs))) == {}
    )


def test_texts_render_the_case_and_tracks() -> None:
    inputs = case()
    assert case_text(inputs) == (
        "Instructions: Slowcore\nReferences:\n- Codeine — D (positive, priority 3)"
    )
    assert tracks_text(outcome(inputs)) == "(no results)"
