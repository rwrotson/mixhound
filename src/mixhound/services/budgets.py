from dataclasses import dataclass


@dataclass(frozen=True)
class DepthBudget:
    """What depth sets; the research time and spending cap are separate run settings."""

    candidates: int
    pool: int  # Server-found leads offered to the research.
    # Model requests one research agent may make; V1 had only time and cost limits.
    requests: int


DEPTH_BUDGETS = {
    "quick": DepthBudget(candidates=20, pool=15, requests=60),
    "standard": DepthBudget(candidates=60, pool=40, requests=120),
    "deep": DepthBudget(candidates=120, pool=80, requests=200),
}
# Model requests of a helper call: the profile pass and the listening-link search.
HELPER_REQUESTS = 25
