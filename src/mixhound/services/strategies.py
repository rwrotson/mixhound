import math
from dataclasses import replace

from mixhound.services.budgets import DEPTH_BUDGETS, DepthBudget

STRATEGIES: dict[str, str] = {
    "collaborators": (
        "Credited collaborators of the references (performers, producers, remixers), their "
        "other projects, bands, and aliases. The candidate pool holds MusicBrainz relations."
    ),
    "label_scene": (
        "Nearby releases on the references' labels and sublabels, and curated compilations, "
        "radio shows, and mixes that contain a reference."
    ),
    "traits": (
        "Web searches for combinations of the profile's traits, including outside the "
        "references' genre, and documented covers, remixes, and samples of the references."
    ),
}


def split_pool(pool: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    """Give relations to collaborators and alternate similar-artist leads among the rest."""
    shares: dict[str, list[dict[str, object]]] = {name: [] for name in STRATEGIES}
    similar = 0
    for item in pool:
        if str(item.get("via") or "").startswith("MusicBrainz"):
            shares["collaborators"].append(item)
            continue
        shares["label_scene" if similar % 2 == 0 else "traits"].append(item)
        similar += 1
    return shares


def strategy_budget(depth: str) -> DepthBudget:
    """One strategy's share of the candidates; time is not divided."""
    budget = DEPTH_BUDGETS[depth]
    return replace(budget, candidates=math.ceil(budget.candidates / len(STRATEGIES)))


def strategy_max_usd(max_usd: float) -> float:
    """One strategy's share of the spending cap."""
    return round(max_usd / len(STRATEGIES), 2)
