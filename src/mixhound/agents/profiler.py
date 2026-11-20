from pydantic_ai import Agent

from mixhound.agents.clock import clock
from mixhound.models.domain import SonicProfile
from mixhound.tools.catalog import describe_recording
from mixhound.tools.deps import ToolDeps
from mixhound.tools.web import fetch_page, search_web


def build_profiler() -> Agent[ToolDeps, SonicProfile]:
    """The sonic-profile agent of parallel strategy runs."""
    return Agent(
        output_type=SonicProfile,
        deps_type=ToolDeps,
        tools=[search_web, fetch_page, describe_recording],
        capabilities=[clock()],
        name="profiler",
    )
