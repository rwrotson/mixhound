import json
import math
from dataclasses import dataclass
from uuid import UUID

from mixhound.memory.context import RunHistory
from mixhound.models.domain import DiscoveryRun, DiscoverySession, DiscoveryTrack, SonicProfile
from mixhound.services.budgets import DEPTH_BUDGETS, DepthBudget
from mixhound.services.exclusions import Exclusions, metadata_key
from mixhound.services.identity import canonical_name, normalize
from mixhound.services.preparation import Preparation

MAX_FEEDBACK_ITEMS = 40
MAX_PROMPT_HISTORY_KEYS = 300
MAX_EXCLUDED_ARTISTS = 300
PRIORITY_SCALE = "priority 1 (minor) to 5 (decisive); it is weight, never a rating"
# Researched candidates asked for per published result, so the server has a choice; the
# server trims the research answer to this many before its own checks.
TARGET_PER_RESULT = 3


def research_target(desired_count: int, max_candidates: int, shares: int = 1) -> int:
    """How many researched candidates one research call should return."""
    return max(1, min(max_candidates, math.ceil(TARGET_PER_RESULT * desired_count / shares)))


def excluded_artists(run: DiscoveryRun, history: list[DiscoveryTrack]) -> list[str]:
    """Every artist name the server will drop, settled up front so the research needn't weigh it.

    Exact names only, as the server checks them (`discovery_evidence._exclusion`).
    """
    settings = run.input.settings
    names = list(settings.excluded_artists)
    if settings.exclude_reference_artists:
        names += [reference.artist for reference in run.input.references]
    if settings.exclude_previously_proposed_artists:
        names += [track.artist for track in history]
    unique: dict[str, str] = {}
    for name in names:
        unique.setdefault(normalize(canonical_name(name)), _clip(name, 120))
    return [name for key, name in unique.items() if key][:MAX_EXCLUDED_ARTISTS]


def _clip(value: str | None, limit: int = 500) -> str:
    return (value or "").strip()[:limit]


def _feedback_weight(item: dict[str, object]) -> int:
    weight = 0
    if item.get("excluded") or item.get("avoid_similar") or item.get("reason"):
        weight += 2
    marked = item.get("weight")
    if item.get("polarity") in ("positive", "negative") and isinstance(marked, int):
        weight += marked
    return weight


def _bounded_feedback(feedback: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep the most informative, most recent feedback with clipped free text."""
    ranked = sorted(
        enumerate(feedback),
        key=lambda pair: (_feedback_weight(pair[1]), pair[0]),
        reverse=True,
    )[:MAX_FEEDBACK_ITEMS]
    bounded = []
    for _, item in sorted(ranked, key=lambda pair: pair[0]):
        copy = {key: value for key, value in item.items() if key != "audio"}
        for name in ("artist", "title"):
            copy[name] = _clip(str(copy.get(name) or ""), 120)
        if isinstance(copy.get("reason"), str):
            copy["reason"] = _clip(str(copy["reason"]), 250)
        tags = copy.get("tags")
        if isinstance(tags, list):
            copy["tags"] = [_clip(str(tag), 40) for tag in tags[:10]]
        bounded.append(copy)
    return bounded


def exclusions_from_history(history: RunHistory) -> Exclusions:
    """Recordings and artists proposed by any earlier run."""
    return Exclusions(
        prior_keys=frozenset(metadata_key(track) for track in history.tracks),
        prior_artists=frozenset(
            normalize(canonical_name(track.artist)) for track in history.tracks
        ),
    )


def _previous_profile(previous: list[DiscoveryRun], session_id: UUID) -> SonicProfile | None:
    runs = [
        old for old in previous if old.session_id == session_id and old.sonic_profile is not None
    ]
    return max(runs, key=lambda old: old.created_at).sonic_profile if runs else None


def _references(run: DiscoveryRun, preparation: Preparation | None) -> list[dict[str, object]]:
    facts = preparation.facts if preparation else {}
    references = []
    for r in run.input.references:
        if r.polarity == "neutral":
            continue
        item: dict[str, object] = {
            "id": str(r.id),
            "artist": _clip(r.artist, 120),
            "title": _clip(r.title, 120),
            "polarity": r.polarity,
            "priority": r.priority,
            "note": _clip(r.note, 250),
            "album": r.album,
            "year": r.year,
            "url": r.url,
            "version": r.version_qualifier,
        }
        known = facts.get(r.id)
        if known and known.tags:
            item["catalog_tags"] = known.tags
        if known and known.audio:
            item["audio"] = known.audio.describe()
        references.append(item)
    return references


def profile_context(run: DiscoveryRun, preparation: Preparation | None = None) -> str:
    """The prompt of the profile-only pass that precedes parallel strategy runs."""
    payload = {
        "instructions": _clip(run.input.instructions, 2000),
        "references": _references(run, preparation),
        "feedback": _bounded_feedback(run.input.feedback),
    }
    return (
        f"Write the sonic profile of these references. Reference priority runs from "
        f"{PRIORITY_SCALE}.\n" + json.dumps(payload, ensure_ascii=False)
    )


def previous_profile(session: DiscoverySession, history: RunHistory) -> SonicProfile | None:
    """The sonic profile of this session's latest earlier run, if any."""
    return _previous_profile(history.runs, session.id)


@dataclass(frozen=True)
class StrategyShare:
    """One parallel strategy's part of a run: its search path, profile, pool and budget."""

    name: str
    description: str
    profile: SonicProfile | None
    pool: list[dict[str, object]]
    budget: DepthBudget
    # How many parallel searches split the run's target.
    shares: int


def build_context(
    run: DiscoveryRun,
    history: RunHistory,
    preparation: Preparation | None = None,
    share: StrategyShare | None = None,
) -> tuple[str, Exclusions]:
    """Return a compact prompt and validator context derived from persisted history.

    A parallel strategy run passes its `share`; without it the prompt is the single-pass one.
    """
    tracks = history.tracks
    ordered_keys = list(dict.fromkeys(metadata_key(track) for track in tracks))
    exclusions = Exclusions(
        prior_keys=frozenset(ordered_keys),
        prior_artists=frozenset(normalize(canonical_name(track.artist)) for track in tracks),
    )
    settings = run.input.settings
    budget = share.budget if share else DEPTH_BUDGETS[settings.depth]
    target = research_target(
        settings.desired_count,
        budget.candidates,
        share.shares if share else 1,
    )
    references = _references(run, preparation)
    profile = share.profile if share else _previous_profile(history.runs, run.session_id)
    pool = share.pool if share else preparation.pool if preparation else []
    payload: dict[str, object] = {
        "instructions": _clip(run.input.instructions, 2000),
        "references": references,
        "settings": {
            "desired_count": settings.desired_count,
            "max_per_artist": settings.max_per_artist,
            "year_from": settings.year_from,
            "year_to": settings.year_to,
            "excluded_artists": excluded_artists(run, tracks),
        },
        "budget": {"target": target, "candidates": budget.candidates, "minutes": settings.minutes},
        "feedback": _bounded_feedback(run.input.feedback),
        "previous_profile": profile.model_dump(mode="json", exclude={"version"})
        if profile
        else None,
        "candidate_pool": pool,
        "previous_recording_keys": ordered_keys[-MAX_PROMPT_HISTORY_KEYS:],
    }
    if share is not None:
        del payload["previous_profile"]
        payload["profile"] = profile.model_dump(mode="json") if profile else None
        payload["strategy"] = {"name": share.name, "description": share.description}
    prompt = (
        f"Research recordings that sound like these references and return up to {target} "
        "scored candidates, best first. The server verifies them and publishes the best "
        f"{settings.desired_count}, so don't stop once you have {settings.desired_count}. "
        "Also include recordings you saw but did not research, marked researched: false. "
        f"Reference priority runs from {PRIORITY_SCALE}. In closest_reference_ids use only "
        "the IDs supplied in references; do not invent or replace reference IDs.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    return prompt, exclusions
