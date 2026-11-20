from pathlib import Path

import pytest
from pydantic_evals import Case, Dataset

from evals.__main__ import build_parser, command_compare, load, overrides, save
from evals.cases import CaseMetadata, DiscoveryCase, RunOutcome
from evals.evaluators import VerifiedCount
from tests.support.evals import case, outcome, track


async def report(verified: int, tmp_path: Path, name: str) -> Path:
    inputs = case(desired_count=2)

    async def task(given: DiscoveryCase) -> RunOutcome:
        return outcome(given, results=[track("Low", f"T{n}") for n in range(verified)])

    dataset = Dataset[DiscoveryCase, RunOutcome, CaseMetadata](
        name="cli", cases=[Case(name="slowcore", inputs=inputs)], evaluators=[VerifiedCount()]
    )
    path = tmp_path / f"{name}.json"
    save(await dataset.evaluate(task, name=name, progress=False), path)
    return path


async def test_saved_reports_load_and_compare(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = await report(1, tmp_path, "before")
    after = await report(2, tmp_path, "after")

    loaded = load(after)
    command_compare(build_parser().parse_args(["compare", str(before), str(after)]))

    assert loaded.cases[0].scores["verified_share"].value == 1.0
    printed = capsys.readouterr().out
    assert "slowcore" in printed
    assert "verified_share" in printed


def test_run_arguments_become_setting_overrides() -> None:
    args = build_parser().parse_args(
        ["run", "--depth", "quick", "--effort", "low", "--count", "5", "--parallel"]
    )
    assert args.mode == "both"
    assert {name: value for name, value in overrides(args).items() if value is not None} == {
        "depth": "quick",
        "reasoning_effort": "low",
        "desired_count": 5,
        "parallel_strategies": True,
    }


def test_run_rejects_unknown_choices() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--mode", "fast"])
