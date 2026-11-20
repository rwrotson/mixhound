from dataclasses import dataclass, field

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import EvaluationReason, Evaluator
from pydantic_evals.evaluators.evaluator import EvaluatorOutput
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output

from evals.cases import CaseMetadata, DiscoveryCase, RunOutcome
from evals.evaluators import Context
from mixhound.models.domain import DiscoveryTrack

RUBRIC = (
    "The output lists recordings proposed for the input's references and instructions. "
    "Pass when the recordings, as a set, fit the instructions and the positive references' "
    "style and break none of the explicit constraints (era, genre, mood, what to avoid). "
    "Score from 0 to 1 how well they fit; judge by your knowledge of the recordings."
)


def case_text(case: DiscoveryCase) -> str:
    """The case as the judge reads it: instructions and weighted references."""
    references = "\n".join(
        f"- {ref.artist} — {ref.title} ({ref.polarity}, priority {ref.priority})"
        for ref in case.references
    )
    return f"Instructions: {case.instructions or '(none)'}\nReferences:\n{references}"


def _track(track: DiscoveryTrack) -> str:
    details = ", ".join(
        part for part in (str(track.year) if track.year else "", ", ".join(track.genres)) if part
    )
    similar = "; ".join(track.similarities[:3])
    return (
        f"- {track.artist} — {track.title}"
        + (f" ({details})" if details else "")
        + (f": {similar}" if similar else "")
    )


def tracks_text(outcome: RunOutcome) -> str:
    """The published results as the judge reads them."""
    return "\n".join(_track(track) for track in outcome.run.results) or "(no results)"


@dataclass
class InstructionFit(Evaluator[DiscoveryCase, RunOutcome, CaseMetadata]):
    """An LLM's judgement of how well the published results fit the case."""

    model: Model
    model_settings: ModelSettings | None = field(default=None)

    def build_serialization_arguments(self) -> dict[str, object]:
        """Name the model in the report instead of serialising the model object."""
        return {"model": self.model.model_name}

    async def evaluate(self, ctx: Context) -> EvaluatorOutput:
        """Score the fit and assert it passes; skipped when nothing was published."""
        if not ctx.output.run.results:
            return {}
        grade = await judge_input_output(
            case_text(ctx.inputs),
            tracks_text(ctx.output),
            RUBRIC,
            self.model,
            self.model_settings,
        )
        return {
            "instruction_fit": EvaluationReason(value=grade.score, reason=grade.reason),
            "instruction_fit_pass": grade.pass_,
        }
