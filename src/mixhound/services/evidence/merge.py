import json

from mixhound.models.domain import SonicProfile
from mixhound.models.research import ResearchCandidate, ResearchOutput
from mixhound.services.evidence.links import _split_alternate
from mixhound.services.identity import recording_key

_CANDIDATE_DEFAULTS: dict[str, object] = {
    "artist_alt": None,
    "title_alt": None,
    "album": None,
    "year": None,
    "version_qualifier": None,
    "original_artist": None,
    "original_title": None,
    "genres": [],
    "similarity_score": None,
    "confidence": "low",
    "researched": True,
    "similarities": [],
    "differences": [],
    "closest_reference_ids": [],
    "discovery_paths": [],
    "sources": [],
    "listening_urls": [],
}


def notes_output(lines: list[str]) -> ResearchOutput:
    """Candidates from interim notes written before the research was stopped."""
    candidates: list[ResearchCandidate] = []
    for line in lines:
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                candidates.append(ResearchCandidate.model_validate({**_CANDIDATE_DEFAULTS, **data}))
        except ValueError:
            continue
    return ResearchOutput(sonic_profile=SonicProfile(), candidates=candidates)


def _candidate_key(candidate: ResearchCandidate) -> str:
    artist, _ = _split_alternate(candidate.artist, candidate.artist_alt)
    title, _ = _split_alternate(candidate.title, candidate.title_alt)
    return recording_key(artist, title, qualifier=candidate.version_qualifier)


def _absorb(kept: ResearchCandidate, other: ResearchCandidate) -> ResearchCandidate:
    """Keep the higher-scored candidate and add the other's paths, sources, and links."""
    if (other.similarity_score or -1) > (kept.similarity_score or -1):
        kept, other = other.model_copy(deep=True), kept
    paths = {(path.strategy, path.explanation) for path in kept.discovery_paths}
    kept.discovery_paths += [
        path for path in other.discovery_paths if (path.strategy, path.explanation) not in paths
    ]
    urls = {source.url for source in kept.sources}
    kept.sources += [source for source in other.sources if source.url not in urls]
    kept.listening_urls = list(dict.fromkeys(kept.listening_urls + other.listening_urls))
    kept.researched = kept.researched or other.researched
    return kept


def merge_outputs(
    outputs: dict[str, ResearchOutput],
    profile: SonicProfile | None = None,
) -> tuple[ResearchOutput, int]:
    """Combine parallel strategy answers; return the merged answer and the overlap.

    Candidates are interleaved best-first across strategies; one recording found by several
    strategies becomes one candidate. The overlap counts recordings found by two or more.
    """
    merged: dict[str, ResearchCandidate] = {}
    finders: dict[str, set[str]] = {}
    queues = [(name, list(output.candidates)) for name, output in outputs.items()]
    for rank in range(max((len(queue) for _, queue in queues), default=0)):
        for name, queue in queues:
            if rank >= len(queue):
                continue
            candidate = queue[rank]
            key = _candidate_key(candidate)
            finders.setdefault(key, set()).add(name)
            merged[key] = (
                _absorb(merged[key], candidate)
                if key in merged
                else candidate.model_copy(deep=True)
            )
    if profile is None:
        profile = next(
            (
                output.sonic_profile
                for output in outputs.values()
                if output.sonic_profile.summary or output.sonic_profile.shared_traits
            ),
            SonicProfile(),
        )
    overlap = sum(len(names) > 1 for names in finders.values())
    return ResearchOutput(sonic_profile=profile, candidates=list(merged.values())), overlap
