from uuid import uuid4

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from mixhound.models.domain import DiscoveryRun, DiscoveryRunInput
from mixhound.services.ports import SearchHit
from mixhound.services.research import AgentCall
from mixhound.services.run_state import RunState
from mixhound.tools.deps import Toolbox, ToolDeps
from tests.fakes import FakeCatalog, FakeFetch, Reply

REQUESTS = 5


class FakeSearch:
    """Returns fixed hits, or raises, and records the queries."""

    def __init__(self, hits: list[SearchHit] | Exception | None = None) -> None:
        self.hits = hits if hits is not None else []
        self.queries: list[str] = []

    async def search(self, query: str, *, count: int | None = None) -> list[SearchHit]:
        self.queries.append(query)
        if isinstance(self.hits, Exception):
            raise self.hits
        return self.hits


def call(
    *,
    label: str = "research",
    seconds: float = 60.0,
    notes: bool = True,
    candidates: int = 2,
    max_usd: float | None = None,
) -> AgentCall:
    return AgentCall(
        label,
        seconds,
        "Research.",
        model="test",
        max_usd=max_usd,
        notes=notes,
        requests=REQUESTS,
        candidates=candidates,
    )


def tool_deps(
    *,
    search: FakeSearch | None = None,
    pages: dict[str, Reply] | None = None,
    catalog: FakeCatalog | None = None,
    agent_call: AgentCall | None = None,
) -> ToolDeps:
    state = RunState(DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput()))
    toolbox = Toolbox(FakeFetch(pages or {}), catalog or FakeCatalog(), search)
    return ToolDeps(state, agent_call or call(), toolbox)


def context(deps: ToolDeps) -> RunContext[ToolDeps]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())
