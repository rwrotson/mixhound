from pydantic_ai import Agent

from mixhound.agents.clock import clock
from mixhound.models.research import LinkSearchOutput
from mixhound.tools.deps import ToolDeps
from mixhound.tools.web import fetch_page, search_web


def build_link_finder() -> Agent[ToolDeps, LinkSearchOutput]:
    """The listening-link follow-up agent."""
    return Agent(
        output_type=LinkSearchOutput,
        deps_type=ToolDeps,
        tools=[search_web, fetch_page],
        capabilities=[clock()],
        name="link_finder",
    )
