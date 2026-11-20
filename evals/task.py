from collections.abc import Mapping
from typing import Any

from evals import summary
from evals.cases import DiscoveryCase, RunOutcome
from mixhound.memory.context import RunHistory
from mixhound.models.domain import (
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySession,
    DiscoverySettings,
)
from mixhound.services.pipeline import ResearchPipeline
from mixhound.services.run_state import RunState


def apply_overrides(settings: DiscoverySettings, overrides: Mapping[str, Any]) -> DiscoverySettings:
    """The case's settings with the given ones replaced; None leaves a setting as it is."""
    changes = {name: value for name, value in overrides.items() if value is not None}
    return DiscoverySettings.model_validate(settings.model_dump() | changes)


class DiscoveryTask:
    """Runs one case through the research pipeline, as the run queue would."""

    def __init__(
        self, pipeline: ResearchPipeline, overrides: Mapping[str, Any] | None = None
    ) -> None:
        self.pipeline = pipeline
        self.overrides = dict(overrides or {})

    async def __call__(self, case: DiscoveryCase) -> RunOutcome:
        """Research the case and return the finished run and its diagnostic."""
        settings = apply_overrides(case.settings, self.overrides)
        session = DiscoverySession(
            name="eval",
            references=case.references,
            settings=settings,
            instructions=case.instructions,
        )
        run = DiscoveryRun(
            session_id=session.id,
            input=DiscoveryRunInput(
                references=case.references,
                instructions=case.instructions,
                settings=settings,
            ),
        )
        # The run executes in the case's own context, so its agent spans join the case's span
        # tree, and with no history, so one case's tracks never exclude another's.
        diagnostic = await self.pipeline.execute(
            RunState(run), session, RunHistory(runs=[], tracks=[])
        )
        summary.record(run, diagnostic)
        return RunOutcome(run=run, diagnostic=diagnostic)
