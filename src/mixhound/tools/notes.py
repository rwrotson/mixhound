from pydantic_ai import RunContext

from mixhound.models.research import ResearchCandidate
from mixhound.tools.deps import ToolDeps


def record_candidate(ctx: RunContext[ToolDeps], candidate: ResearchCandidate) -> str:
    """Record one scored candidate as an interim note, kept if the run is cut off.

    Args:
        candidate: The candidate, with the same fields as in the final answer.
    """
    deps = ctx.deps
    if not deps.take("record_candidate"):
        return "Candidate limit reached; stop researching and answer."
    deps.notes.append(candidate.model_dump_json())
    deps.report()
    return f"Noted ({len(deps.notes)})."
