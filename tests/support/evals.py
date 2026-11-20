from typing import Any
from uuid import uuid4

from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.otel import SpanTree, SpanTreeRecordingError

from evals.cases import CaseMetadata, DiscoveryCase, RunOutcome
from mixhound.models.domain import (
    DiscoveryReference,
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySettings,
    DiscoveryTrack,
)


def reference(artist: str = "Codeine", title: str = "D") -> DiscoveryReference:
    return DiscoveryReference(kind="manual", artist=artist, title=title)


def case(**settings: object) -> DiscoveryCase:
    return DiscoveryCase(
        instructions="Slowcore",
        settings=DiscoverySettings.model_validate(settings),
        references=[reference()],
    )


def track(artist: str, title: str, **changes: object) -> DiscoveryTrack:
    return DiscoveryTrack.model_validate({"artist": artist, "title": title, **changes})


def outcome(
    inputs: DiscoveryCase,
    *,
    results: list[DiscoveryTrack] | None = None,
    leads: list[DiscoveryTrack] | None = None,
    diagnostic: dict[str, Any] | None = None,
    **run: object,
) -> RunOutcome:
    found = DiscoveryRun(
        session_id=uuid4(),
        input=DiscoveryRunInput(
            references=inputs.references, instructions=inputs.instructions, settings=inputs.settings
        ),
        results=[t.model_copy(update={"verification": "verified"}) for t in results or []],
        unverified_leads=leads or [],
    )
    return RunOutcome(run=found.model_copy(update=run), diagnostic=diagnostic or {})


def context(
    inputs: DiscoveryCase, output: RunOutcome, *, spans: SpanTree | None = None
) -> EvaluatorContext[DiscoveryCase, RunOutcome, CaseMetadata]:
    return EvaluatorContext(
        name="case",
        inputs=inputs,
        metadata=None,
        expected_output=None,
        output=output,
        duration=1.0,
        _span_tree=spans if spans is not None else SpanTreeRecordingError("not recorded"),
        attributes={},
        metrics={},
    )
