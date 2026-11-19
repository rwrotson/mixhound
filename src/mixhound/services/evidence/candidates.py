from collections.abc import Collection
from uuid import UUID

from mixhound.models.domain import (
    DiscoveryListeningLink,
    DiscoveryPath,
    DiscoveryResearchResult,
    DiscoveryRunInput,
    DiscoverySource,
    DiscoveryTrack,
)
from mixhound.models.research import ResearchCandidate, ResearchOutput
from mixhound.services.evidence.links import _link_target, _service, _split_alternate


def candidate_track(
    candidate: ResearchCandidate,
    reference_ids: Collection[UUID],
) -> DiscoveryTrack:
    """Build an unconfirmed track from one research candidate."""
    artist, artist_alt = _split_alternate(candidate.artist, candidate.artist_alt)
    title, title_alt = _split_alternate(candidate.title, candidate.title_alt)
    closest: list[UUID] = []
    for raw in candidate.closest_reference_ids:
        try:
            value = UUID(raw)
        except ValueError:
            continue
        if value in reference_ids and value not in closest:
            closest.append(value)
    links: list[DiscoveryListeningLink] = []
    for url in dict.fromkeys(candidate.listening_urls):
        service = _service(url)
        links.append(
            DiscoveryListeningLink(
                url=url,
                service=service,
                target_type=_link_target(url, service),  # type: ignore[arg-type]
            ),
        )
    return DiscoveryTrack(
        artist=artist,
        title=title,
        artist_alt=artist_alt,
        title_alt=title_alt,
        album=candidate.album,
        year=candidate.year,
        genres=candidate.genres[:8],
        version_qualifier=candidate.version_qualifier,
        original_artist=candidate.original_artist,
        original_title=candidate.original_title,
        similarity_score=candidate.similarity_score,
        confidence=candidate.confidence,
        researched=candidate.researched,
        similarities=candidate.similarities,
        differences=candidate.differences,
        closest_reference_ids=closest,
        sources=[
            DiscoverySource(
                url=source.url,
                source_type=_service(source.url),
                claim=source.claim,
                target_type=source.target_type,
            )
            for source in candidate.sources
        ],
        listening_links=links,
        discovery_paths=[
            DiscoveryPath(
                strategy=path.strategy,
                explanation=path.explanation,
                source_url=path.source_url,
            )
            for path in candidate.discovery_paths
        ],
    )


def research_result(
    output: ResearchOutput | DiscoveryResearchResult,
    run_input: DiscoveryRunInput,
) -> DiscoveryResearchResult:
    """All candidates of a research answer, pending server confirmation.

    A pre-built result (a replayed or test answer) passes through unchanged.
    """
    if isinstance(output, DiscoveryResearchResult):
        return output
    reference_ids = {reference.id for reference in run_input.references}
    return DiscoveryResearchResult(
        results=[candidate_track(candidate, reference_ids) for candidate in output.candidates],
        sonic_profile=output.sonic_profile,
    )
