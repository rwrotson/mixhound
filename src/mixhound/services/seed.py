import logging
from collections.abc import Callable

from mixhound.models.domain import DiscoveryReference, DiscoveryRunInput
from mixhound.services.audio import AudioAnalyzer
from mixhound.services.audio_features import AudioFeatures
from mixhound.services.exclusions import Exclusions
from mixhound.services.identity import canonical_name, normalize, recording_key
from mixhound.services.ports import (
    CatalogLead,
    CatalogRecording,
    CatalogUnavailableError,
    LeadGroup,
    TrackCatalog,
)
from mixhound.services.preparation import Preparation, ReferenceFacts

log = logging.getLogger(__name__)

MAX_SEED_REFERENCES = 6
RELATIONS_PER_ARTIST = 6
MIN_REFERENCE_TAGS = 3
MAX_REFERENCE_TAGS = 8
MAX_SIMILAR_PER_ARTIST = 2
SIMILAR_TRACKS = 15
SIMILAR_ARTISTS = 8
RELATION_TYPES = {
    "member of band",
    "is person",
    "collaboration",
    "supporting musician",
    "subgroup",
    "founder",
    "artist rename",
    "vocal supporting musician",
    "instrumental supporting musician",
    "tribute",
    "producer",
}


def _never_cancelled() -> None:
    return None


def _group(recording: CatalogRecording, group: LeadGroup) -> list[CatalogLead]:
    return [lead for lead in recording.leads if lead.group == group]


def _leads(recording: CatalogRecording | None) -> list[list[CatalogLead]]:
    """Lead lists for one reference, one list per source and path, in V1 order."""
    if recording is None:
        return []
    groups = [_group(recording, "lastfm_similar_tracks")[:SIMILAR_TRACKS]]
    if recording.artist_mbids:
        groups.append(_group(recording, "lb_radio"))
        groups.append(_group(recording, "lb_similar_artists")[:SIMILAR_ARTISTS])
    else:
        groups.append(_group(recording, "lastfm_similar_artists")[:SIMILAR_ARTISTS])
    groups.append(
        [
            lead
            for lead in _group(recording, "musicbrainz_relations")
            if lead.relation in RELATION_TYPES
        ][:RELATIONS_PER_ARTIST],
    )
    return groups


def _tags(recording: CatalogRecording) -> list[str]:
    """MusicBrainz tags, topped up from Last.fm when there are too few."""
    tags = list(recording.tags.get("musicbrainz", ()))
    if len(tags) < MIN_REFERENCE_TAGS:
        tags += [tag for tag in recording.tags.get("lastfm", ()) if tag not in tags][
            : MAX_REFERENCE_TAGS - len(tags)
        ]
    return tags


class Seeder:
    """Server-side preparation before research: catalog facts, audio and a candidate pool."""

    def __init__(self, catalog: TrackCatalog, audio: AudioAnalyzer | None) -> None:
        self.catalog = catalog
        self.audio = audio

    async def _reference_audio(
        self, recording: CatalogRecording | None, facts: ReferenceFacts
    ) -> AudioFeatures | None:
        """Measure a reference from the first catalog preview that decodes."""
        if self.audio is None or recording is None:
            return None
        for preview in recording.previews:
            found = await self.audio.url(preview.url, key=f"{preview.source}:{preview.page_url}")
            if found:
                facts.preview_url = preview.page_url
                return found
        return None

    async def prepare(
        self,
        run_input: DiscoveryRunInput,
        exclusions: Exclusions,
        pool_size: int,
        check: Callable[[], None] = _never_cancelled,
    ) -> Preparation:
        """Gather reference facts and up to `pool_size` leads; `check` may abort between steps."""
        preparation = Preparation()
        references = sorted(
            (item for item in run_input.references if item.polarity != "neutral"),
            key=lambda item: -item.priority,
        )
        catalog: TrackCatalog | None = self.catalog
        seeds: list[tuple[DiscoveryReference, list[list[CatalogLead]]]] = []
        for reference in references:
            check()
            facts = ReferenceFacts()
            preparation.facts[reference.id] = facts
            seeded = reference.polarity == "positive" and len(seeds) < MAX_SEED_REFERENCES
            found: CatalogRecording | None = None
            if catalog is not None:
                try:
                    found = await catalog.reference(
                        reference.artist, reference.title, version=reference.version_qualifier
                    )
                except CatalogUnavailableError as error:
                    log.warning("seed_catalog_unavailable error=%s", error)
                    catalog = None
            if found:
                facts.mbid, facts.artist_mbids, facts.tags = (
                    found.mbid,
                    found.artist_mbids,
                    _tags(found),
                )
            facts.audio = await self._reference_audio(found, facts)
            if seeded:
                seeds.append((reference, _leads(found)))
        preparation.pool = _interleave(seeds, run_input, exclusions, pool_size)
        return preparation


def _interleave(
    seeds: list[tuple[DiscoveryReference, list[list[CatalogLead]]]],
    run_input: DiscoveryRunInput,
    exclusions: Exclusions,
    limit: int,
) -> list[dict[str, object]]:
    """Merge the leads round-robin across references and sources, skipping excluded ones."""
    pool = _Pool(run_input, exclusions)
    queues = [(reference, list(group)) for reference, groups in seeds for group in groups if group]

    # Round-robin: each queue contributes one accepted lead per round.
    while queues and len(pool.leads) < limit:
        for reference, queue in list(queues):
            while queue and not pool.accept(reference, queue.pop(0)):
                pass
            if not queue:
                queues.remove((reference, queue))
            if len(pool.leads) >= limit:
                break
    return pool.leads


class _Pool:
    """Leads accepted so far, and the rules that exclude a lead."""

    def __init__(self, run_input: DiscoveryRunInput, exclusions: Exclusions) -> None:
        settings = run_input.settings
        self.exclusions = exclusions
        self.reference_artists = {
            normalize(canonical_name(item.artist)) for item in run_input.references
        }
        self.reference_keys = {
            recording_key(item.artist, item.title) for item in run_input.references
        }
        self.excluded = {normalize(value) for value in settings.excluded_artists}
        if settings.exclude_reference_artists:
            self.excluded |= self.reference_artists
        if settings.exclude_previously_proposed_artists:
            self.excluded |= exclusions.prior_artists
        self.seen: set[str] = set()
        self.per_artist: dict[str, int] = {}
        self.leads: list[dict[str, object]] = []

    def accept(self, reference: DiscoveryReference, item: CatalogLead) -> bool:
        """Add a lead unless it is excluded, repeated or over its artist's share."""
        artist = normalize(canonical_name(item.artist))
        key = recording_key(item.artist, item.title) if item.title else f"artist:{artist}"
        if (
            not artist
            or artist in self.excluded
            or key in self.seen
            or key in self.reference_keys
            or key in self.exclusions.prior_keys
            or self.per_artist.get(artist, 0) >= MAX_SIMILAR_PER_ARTIST
            or (item.title is None and artist in self.reference_artists)
        ):
            return False
        self.seen.add(key)
        self.per_artist[artist] = self.per_artist.get(artist, 0) + 1
        self.leads.append(
            {
                "artist": item.artist,
                "title": item.title,
                "via": f"{item.source} {item.relation} of {reference.artist} — {reference.title}",
                "reference_id": str(reference.id),
                "url": item.url,
            },
        )
        return True
