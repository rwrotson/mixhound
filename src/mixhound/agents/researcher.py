from pydantic_ai import Agent

from mixhound.agents.clock import clock
from mixhound.models.research import ResearchOutput
from mixhound.tools.catalog import describe_recording
from mixhound.tools.deps import ToolDeps
from mixhound.tools.notes import record_candidate
from mixhound.tools.web import fetch_page, search_web


def build_researcher() -> Agent[ToolDeps, ResearchOutput]:
    """The research agent; model, instructions and limits come with each run."""
    return Agent(
        output_type=ResearchOutput,
        deps_type=ToolDeps,
        tools=[search_web, fetch_page, describe_recording, record_candidate],
        capabilities=[clock()],
        name="researcher",
    )
