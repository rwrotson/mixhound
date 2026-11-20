import statistics
from typing import Any

from pydantic_evals.dataset import increment_eval_metric, set_eval_attribute

from mixhound.models.domain import DiscoveryRun
from mixhound.services.identity import canonical_name, normalize
from mixhound.services.run_state import RESEARCH_STAGE


def _share(part: float, whole: float) -> float:
    return round(part / whole, 2) if whole else 0.0


def _mean(values: list[int]) -> float | None:
    return round(statistics.fmean(values), 1) if values else None


def metrics(run: DiscoveryRun) -> dict[str, float]:
    """Quality, cost and speed figures of one finished run (V1 `summarize`)."""
    results, leads = run.results, run.unverified_leads
    artists = [normalize(canonical_name(track.artist)) for track in results]
    figures: dict[str, float | None] = {
        "verified": len(results),
        "leads": len(leads),
        "verified_share": _share(len(results), run.input.settings.desired_count),
        "distinct_artist_share": _share(len(set(artists)), len(artists)),
        "mean_similarity": _mean(
            [track.similarity_score for track in results if track.similarity_score is not None]
        ),
        "mean_audio_similarity": _mean(
            [
                track.audio_similarity
                for track in results + leads
                if track.audio_similarity is not None
            ]
        ),
    }
    stats = run.metrics
    if stats is not None:
        figures |= {
            "seconds": round(sum(stats.stages.values()), 1),
            "research_seconds": stats.stages.get(RESEARCH_STAGE),
            "profile_seconds": stats.stages.get("profile"),
            "cost_usd": stats.cost_usd,
            "input_tokens": stats.input_tokens,
            "cached_input_tokens": stats.cached_input_tokens,
            "output_tokens": stats.output_tokens,
            "web_searches": stats.web_searches,
            "research_searches": stats.research_searches,
            "candidates": stats.candidates,
            "sonic_evidence_share": _share(stats.sonic_evidence, stats.candidates),
            "overlap": stats.overlap,
        }
    return {name: value for name, value in figures.items() if value is not None}


def attributes(run: DiscoveryRun, diagnostic: dict[str, Any]) -> dict[str, object]:
    """How the run ended, what it showed and what it rejected."""
    stats = run.metrics
    return {
        "status": run.status,
        "failure": run.failure_code,
        "partial": run.partial,
        "tracks": [
            f"{'✓' if track.verification == 'verified' else '·'} {track.artist} — {track.title}"
            f" ({track.similarity_score}/{track.audio_similarity})"
            for track in run.results + run.unverified_leads
        ],
        "strategies": (
            {name: item.model_dump(mode="json") for name, item in stats.strategies.items()}
            if stats
            else {}
        ),
        "rejections": [
            f"{item['action']}: {item['artist']} — {item['title']} ({item['reason']})"
            for item in diagnostic.get("rejections", [])
        ],
    }


def record(run: DiscoveryRun, diagnostic: dict[str, Any]) -> None:
    """Attach the run's figures and outline to the current eval case."""
    for name, figure in metrics(run).items():
        increment_eval_metric(name, figure)
    for name, value in attributes(run, diagnostic).items():
        set_eval_attribute(name, value)
