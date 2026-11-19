from dataclasses import dataclass


@dataclass(frozen=True)
class DepthBudget:
    """What depth sets; the research time and spending cap are separate run settings."""

    candidates: int
    pool: int  # Server-found leads offered to the research.


DEPTH_BUDGETS = {
    "quick": DepthBudget(candidates=20, pool=15),
    "standard": DepthBudget(candidates=60, pool=40),
    "deep": DepthBudget(candidates=120, pool=80),
}
