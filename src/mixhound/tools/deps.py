import time
from collections import Counter
from dataclasses import dataclass, field

from mixhound.services.ports import PageFetch, TrackCatalog, WebSearch
from mixhound.services.research import AgentCall
from mixhound.services.run_state import RunState

SEARCH_TOOLS = ("search_web", "fetch_page")
# Per-call caps of the tools that are not bounded by the request limit alone.
MAX_DESCRIBED = 20


@dataclass(frozen=True)
class Toolbox:
    """The clients the agents' tools use; shared by every run."""

    fetch: PageFetch
    catalog: TrackCatalog
    search: WebSearch | None = None


@dataclass
class ToolDeps:
    """What one agent call's tools see: the run, the call's limits and its own counters."""

    state: RunState
    call: AgentCall
    toolbox: Toolbox
    started: float = field(default_factory=time.monotonic)
    # Interim candidates as JSON, in the order the research recorded them.
    notes: list[str] = field(default_factory=list)
    calls: Counter[str] = field(default_factory=Counter)
    failed_fetches: int = 0

    def cap(self, tool: str) -> int:
        """Return how often a tool may run in this call."""
        if tool in SEARCH_TOOLS:
            return 2 * self.call.requests
        if tool == "describe_recording":
            return MAX_DESCRIBED
        return self.call.candidates

    def take(self, tool: str) -> bool:
        """Count one use of a tool; False once its cap is reached."""
        if self.calls[tool] >= self.cap(tool):
            return False
        self.calls[tool] += 1
        return True

    def elapsed(self) -> float:
        """Seconds since the call started."""
        return time.monotonic() - self.started

    @property
    def searches(self) -> int:
        """Web searches and page fetches made so far."""
        return sum(self.calls[tool] for tool in SEARCH_TOOLS)

    def report(self) -> None:
        """Show the call's searches and noted candidates in the run's live progress."""
        self.state.listener(self.call.label)(self.searches, len(self.notes))
