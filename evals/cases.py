from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_evals import Dataset

from mixhound.models.domain import DiscoveryReference, DiscoveryRun, DiscoverySettings

DATASET = Path(__file__).with_name("datasets") / "discovery.yaml"


class DiscoveryCase(BaseModel):
    """One session to research: references, instructions and settings."""

    instructions: str = ""
    settings: DiscoverySettings = Field(default_factory=DiscoverySettings)
    references: list[DiscoveryReference]


class CaseMetadata(BaseModel):
    """Where a case comes from."""

    source: str = ""


class RunOutcome(BaseModel):
    """A finished run and its diagnostic."""

    run: DiscoveryRun
    diagnostic: dict[str, Any] = Field(default_factory=dict)


type DiscoveryDataset = Dataset[DiscoveryCase, RunOutcome, CaseMetadata]


def load_dataset(path: Path = DATASET, names: list[str] | None = None) -> DiscoveryDataset:
    """Load the cases, keeping only the named ones when names are given."""
    dataset = Dataset[DiscoveryCase, RunOutcome, CaseMetadata].from_file(path)
    if names:
        missing = set(names) - {case.name for case in dataset.cases}
        if missing:
            raise ValueError(f"Unknown cases: {', '.join(sorted(missing))}")
        dataset.cases = [case for case in dataset.cases if case.name in names]
    return dataset
