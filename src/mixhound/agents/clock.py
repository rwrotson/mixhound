from dataclasses import dataclass
from decimal import Decimal

from pydantic_ai import (
    InstructionCapabilityPart,
    ModelRequestContext,
    OperatorSource,
    OperatorText,
    RunContext,
)
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelResponse

from mixhound.services.errors import MixhoundError
from mixhound.tools.deps import ToolDeps

# The last tenth of the time is left for the final answer.
WRAP_SHARE = 0.9


class BudgetExceededError(MixhoundError):
    """The estimated cost of an agent call reached its spending cap."""


@dataclass(kw_only=True)
class ClockPart(InstructionCapabilityPart, namespace="mixhound"):
    """The real time an agent call has used, shown with one model request only."""


def _minutes(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def clock_line(elapsed: float, limit: float, notes: int) -> str:
    """The clock the model sees: it cannot tell time and overestimates it several times over."""
    return (
        f"Clock: {_minutes(elapsed)} of {_minutes(limit)} used (real time, not your estimate); "
        f"keep researching until about {_minutes(limit * WRAP_SHARE)}. Notes written: {notes}."
    )


def check_budget(cost: Decimal | None, max_usd: float | None) -> None:
    """Stop a call once its estimated cost reached the cap; unpriced models are not capped."""
    if cost is not None and max_usd is not None and cost >= Decimal(str(max_usd)):
        raise BudgetExceededError(f"Estimated cost {cost:.4f} USD reached the cap {max_usd} USD")


async def _before_request(
    ctx: RunContext[ToolDeps], request_context: ModelRequestContext
) -> ModelRequestContext:
    deps = ctx.deps
    check_budget(ctx.usage.cost, deps.call.max_usd)
    # Like V1's PostToolUse hook: researching calls see the clock once they have acted.
    if deps.call.notes and any(isinstance(message, ModelResponse) for message in ctx.messages):
        line = clock_line(deps.elapsed(), deps.call.seconds, len(deps.notes))
        ctx.enqueue(ClockPart(content=OperatorText(line), source=OperatorSource(), scope="request"))
    return request_context


def clock() -> Hooks:
    """Hooks that cap a call's spending and show researching calls the real time used."""
    return Hooks(before_model_request=_before_request)
