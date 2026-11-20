from pydantic_ai import Agent

from mixhound.agents.clock import clock
from mixhound.models.research import ResearchOutput
from mixhound.tools.catalog import describe_recording
from mixhound.tools.deps import ToolDeps
from mixhound.tools.notes import record_candidate
from mixhound.tools.orchestration import delegate_research, lookup_reference, verify_candidates
from mixhound.tools.web import fetch_page, search_web


def build_orchestrator() -> Agent[ToolDeps, ResearchOutput]:
    """The agentic-mode agent; model, instructions and limits come with each run."""
    return Agent(
        output_type=ResearchOutput,
        deps_type=ToolDeps,
        tools=[
            lookup_reference,
            describe_recording,
            search_web,
            fetch_page,
            verify_candidates,
            delegate_research,
            record_candidate,
        ],
        capabilities=[clock()],
        name="orchestrator",
    )
