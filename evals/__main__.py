import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, get_args

from pydantic_evals.reporting import EvaluationReport, EvaluationReportAdapter

from evals import tracing
from evals.cases import DATASET, DiscoveryDataset, load_dataset
from evals.evaluators import default_evaluators
from evals.judge import InstructionFit
from evals.task import DiscoveryTask
from mixhound.config import Settings, StorageSettings
from mixhound.container import open_container
from mixhound.models.domain import DiscoverySettings

MODES = ("pipeline", "agentic")
type Report = EvaluationReport[Any, Any, Any]


def _choices(name: str) -> tuple[str, ...]:
    """The literal values a DiscoverySettings field accepts."""
    annotation = DiscoverySettings.model_fields[name].annotation
    # `Literal[...] | None` nests the literal one level deeper than a plain `Literal[...]`.
    values = get_args(annotation)
    literal = next((get_args(value) for value in values if get_args(value)), values)
    return tuple(value for value in literal if isinstance(value, str))


def overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Run settings given on the command line, by DiscoverySettings field."""
    return {
        "model": args.model,
        "helper_model": args.helper_model,
        "depth": args.depth,
        "minutes": args.minutes,
        "max_usd": args.max_usd,
        "reasoning_effort": args.effort,
        "desired_count": args.count,
        "parallel_strategies": True if args.parallel else None,
    }


def save(report: Report, path: Path) -> None:
    """Write a report, outputs included, so `compare` can read it later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(EvaluationReportAdapter.dump_json(report, indent=2))


def load(path: Path) -> Report:
    """Read a report written by `save`."""
    return EvaluationReportAdapter.validate_json(path.read_bytes())


async def evaluate_modes(dataset: DiscoveryDataset, args: argparse.Namespace) -> dict[str, Report]:
    """Evaluate the dataset once per requested mode with real models."""
    settings = Settings().model_copy(update={"storage": StorageSettings(backend="memory")})
    tracing.enable()
    reports: dict[str, Report] = {}
    async with open_container(settings) as container:
        for evaluator in default_evaluators():
            dataset.add_evaluator(evaluator)
        if args.judge:
            dataset.add_evaluator(InstructionFit(container.models.model(helper=True)))
        for mode in MODES if args.mode == "both" else (args.mode,):
            task = DiscoveryTask(container.pipeline, overrides(args) | {"mode": mode})
            reports[mode] = await dataset.evaluate(
                task,
                name=mode,
                task_name="discovery",
                max_concurrency=args.concurrency,
                repeat=args.repeat,
                metadata={"mode": mode, **{k: v for k, v in task.overrides.items() if v}},
            )
    return reports


def command_run(args: argparse.Namespace) -> None:
    """Evaluate cases in each mode, print the reports and compare the modes."""
    dataset = load_dataset(Path(args.dataset), args.case)
    reports = asyncio.run(evaluate_modes(dataset, args))
    for mode, report in reports.items():
        report.print(include_averages=True)
        if args.out:
            save(report, Path(args.out) / f"{mode}.json")
    if len(reports) == len(MODES):
        reports["agentic"].print(baseline=reports["pipeline"], include_averages=True)
    if args.out:
        sys.stdout.write(f"Reports written to {args.out}\n")


def command_compare(args: argparse.Namespace) -> None:
    """Print the change from one saved report to another."""
    load(Path(args.after)).print(baseline=load(Path(args.before)), include_averages=True)


def build_parser() -> argparse.ArgumentParser:
    """The command line: `run` evaluates cases, `compare` diffs two saved reports."""
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Evaluate discovery runs on fixed reference sets.",
        epilog="See evals/README.md for the commands and the scores.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="research the cases with real models (spends credits)")
    run.add_argument("--dataset", default=str(DATASET))
    run.add_argument("--case", action="append", default=[], help="case name; repeatable")
    run.add_argument("--mode", choices=(*MODES, "both"), default="both")
    run.add_argument("--model")
    run.add_argument("--helper-model")
    run.add_argument("--depth", choices=_choices("depth"))
    run.add_argument("--minutes", type=int, help="research time limit (default 10)")
    run.add_argument("--max-usd", type=float, help="research spending cap (default 3)")
    run.add_argument("--effort", choices=_choices("reasoning_effort"))
    run.add_argument("--count", type=int, help="desired verified results")
    run.add_argument("--parallel", action="store_true", help="parallel strategies (pipeline)")
    run.add_argument("--repeat", type=int, default=1, help="runs per case")
    run.add_argument("--concurrency", type=int, default=1, help="cases researched at once")
    run.add_argument("--judge", action="store_true", help="add the LLM instruction-fit judge")
    run.add_argument("--out", help="folder for <mode>.json reports")
    run.set_defaults(handler=command_run)
    compare = commands.add_parser("compare", help="compare two saved reports")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.set_defaults(handler=command_compare)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run a command."""
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main(sys.argv[1:])
