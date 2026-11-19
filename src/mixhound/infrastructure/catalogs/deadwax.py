from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from deadwax import (
    Capability,
    Deadwax,
    DeadwaxError,
    Dossier,
    EntityKind,
    EntityType,
    ErrorKind,
    IdKind,
    LookupConfig,
    Provenance,
    Query,
    Relation,
    RelationKind,
    RelationLimit,
    Request,
)

from mixhound.config import DeadwaxSettings
from mixhound.services.ports import (
    CATALOG_SOURCES,
    CatalogLead,
    CatalogLink,
    CatalogPage,
    CatalogRecording,
    CatalogUnavailableError,
    DossierFact,
    LeadGroup,
    Preview,
)

log = structlog.get_logger(__name__)

# Sources of preview clips, in the order audio analysis tries them.
PREVIEW_ORDER = ("itunes", "spotify", "bandcamp")
TAGS_PER_SOURCE = 8
TEXT_EXCERPT = 600
# The identity pass failing for these reasons means the catalog is down, not the recording unknown.
UNAVAILABLE = frozenset(
    {ErrorKind.FAILED, ErrorKind.TIMEOUT, ErrorKind.BLOCKED, ErrorKind.RATE_LIMITED}
)

REFERENCE_FACETS = frozenset(
    {Capability.IDENTITY, Capability.TAGS, Capability.RELATIONS, Capability.PREVIEWS}
)
REFERENCE_RELATIONS = (
    RelationLimit(kind=RelationKind.SIMILAR, target=EntityType.RECORDING, per_source=15),
    RelationLimit(kind=RelationKind.SIMILAR, target=EntityType.ARTIST, per_source=8),
    RelationLimit(kind=RelationKind.SIMILAR_ARTIST_RECORDING, per_source=16),
    RelationLimit(kind=RelationKind.MEMBER, per_source=6),
    RelationLimit(kind=RelationKind.PROJECT, per_source=6),
    RelationLimit(kind=RelationKind.ASSOCIATED, per_source=6),
)
CANDIDATE_FACETS = frozenset(
    {
        Capability.IDENTITY,
        Capability.LINKS,
        Capability.PREVIEWS,
        Capability.TAGS,
        Capability.CREDITS,
        Capability.TEXTS,
        Capability.RELATIONS,
    }
)
# Relations that describe a candidate; similarity relations would only add lookups.
CANDIDATE_RELATIONS = tuple(
    RelationLimit(kind=kind, per_source=limit)
    for kind, limit in (
        (RelationKind.LABEL, 3),
        (RelationKind.APPEARS_ON, 3),
        (RelationKind.MEMBER, 6),
        (RelationKind.PROJECT, 6),
        (RelationKind.REMIX_OF, 3),
        (RelationKind.COVER_OF, 3),
        (RelationKind.SAMPLES, 5),
    )
)

SOURCE_LABELS = {"lastfm": "Last.fm", "listenbrainz": "ListenBrainz", "musicbrainz": "MusicBrainz"}
MUSICBRAINZ_RELATIONS = frozenset(
    {RelationKind.MEMBER, RelationKind.PROJECT, RelationKind.ASSOCIATED}
)
MEMBER_RELATION = "member of band"


def lookup_config(settings: DeadwaxSettings, data_dir: Path) -> LookupConfig:
    """Map mixhound settings to the deadwax configuration."""
    secrets = {
        "lastfm": {"api_key": settings.lastfm_api_key},
        "discogs": {"token": settings.discogs_token},
        "youtube": {"api_key": settings.youtube_api_key},
        "spotify": {
            "client_id": settings.spotify_client_id,
            "client_secret": settings.spotify_client_secret,
        },
    }
    values: dict[str, object] = {
        "http": {"contact": settings.contact},
        "cache": {"enabled": settings.cache_enabled, "path": data_dir / "deadwax-cache"},
        "adapter_timeout": settings.adapter_timeout,
        "sources": {
            name: {key: value for key, value in fields.items() if value is not None}
            for name, fields in secrets.items()
        },
    }
    if settings.sources is not None:
        values["default_sources"] = tuple(settings.sources)
    return LookupConfig.model_validate(values)


@asynccontextmanager
async def open_catalog(settings: DeadwaxSettings, data_dir: Path) -> AsyncIterator[DeadwaxCatalog]:
    """Open the deadwax client shared by every lookup of the process."""
    async with Deadwax(lookup_config(settings, data_dir)) as client:
        yield DeadwaxCatalog(client)


def _query(artist: str, title: str, version: str | None, isrc: str | None) -> Query:
    # deadwax reads the version from the title, as catalogs print it.
    if version and version.casefold() not in title.casefold():
        title = f"{title} ({version})"
    ids = {IdKind.ISRC: isrc} if isrc else {}
    return Query(kind=EntityKind.TRACK, artist=artist, title=title, ids=ids)


class DeadwaxCatalog:
    """The track catalog backed by one long-lived deadwax client."""

    def __init__(self, client: Deadwax) -> None:
        self.client = client

    async def reference(
        self, artist: str, title: str, *, version: str | None = None, isrc: str | None = None
    ) -> CatalogRecording | None:
        """Return identity, tags, leads and previews of a reference, or None when unknown."""
        request = Request(
            query=_query(artist, title, version, isrc),
            facets=REFERENCE_FACETS,
            tags_per_source=TAGS_PER_SOURCE,
            relations=REFERENCE_RELATIONS,
        )
        return await self._lookup(request)

    async def candidate(
        self, artist: str, title: str, *, version: str | None = None, isrc: str | None = None
    ) -> CatalogRecording | None:
        """Return identity, links, previews and descriptive facts of a candidate."""
        request = Request(
            query=_query(artist, title, version, isrc),
            facets=CANDIDATE_FACETS,
            tags_per_source=TAGS_PER_SOURCE,
            relations=CANDIDATE_RELATIONS,
        )
        return await self._lookup(request)

    async def _lookup(self, request: Request) -> CatalogRecording | None:
        try:
            dossier = await self.client.lookup(request)
        except DeadwaxError as failure:
            raise CatalogUnavailableError(str(failure)) from failure
        for problem in dossier.errors:
            log.debug(
                "catalog_source_error",
                source=problem.source,
                kind=problem.kind,
                reason=problem.reason,
            )
        if not dossier.found:
            if any(
                problem.source == "musicbrainz" and problem.kind in UNAVAILABLE
                for problem in dossier.errors
            ):
                raise CatalogUnavailableError("MusicBrainz is unavailable")
            return None
        return recording(dossier)


def recording(dossier: Dossier) -> CatalogRecording | None:
    """Map a dossier to what mixhound reads; None without an artist and a title."""
    identity = dossier.identity
    if identity.artist is None or identity.title is None:
        return None
    artist_sources = {item.source for item in identity.artist.sources}
    title_sources = [item for item in identity.title.sources if item.source in artist_sources]
    confirmed = tuple(dict.fromkeys(i.source for i in title_sources if i.source in CATALOG_SOURCES))
    return CatalogRecording(
        artist=identity.artist.value,
        title=identity.title.value,
        confirmed_by=confirmed,
        mbid=_first_id(dossier, IdKind.MUSICBRAINZ_RECORDING),
        artist_mbids=tuple(
            item.value for item in identity.external_ids if item.kind is IdKind.MUSICBRAINZ_ARTIST
        ),
        isrcs=tuple(dict.fromkeys(item.value for item in identity.isrcs)),
        year=_year(identity.date.value if identity.date else None),
        tags=_tags(dossier),
        pages=tuple(
            CatalogPage(item.source, item.url)
            for item in dict.fromkeys(title_sources)
            if item.source in confirmed and item.url
        ),
        links=tuple(
            CatalogLink(
                link.sources[0].source if link.sources else link.service,
                link.url,
                link.target.value,
            )
            for link in dossier.links
        ),
        previews=_previews(dossier),
        leads=tuple(lead for relation in dossier.relations if (lead := _lead(relation))),
        facts=_facts(dossier),
    )


def _first_id(dossier: Dossier, kind: IdKind) -> str | None:
    return next((item.value for item in dossier.identity.external_ids if item.kind is kind), None)


def _year(date: str | None) -> int | None:
    return int(date[:4]) if date and date[:4].isdigit() else None


def _tags(dossier: Dossier) -> dict[str, tuple[str, ...]]:
    by_source: dict[str, list[str]] = {}
    for tag in dossier.tags:
        for provenance in tag.sources:
            names = by_source.setdefault(provenance.source, [])
            if tag.name not in names:
                names.append(tag.name)
    return {source: tuple(names) for source, names in by_source.items()}


def _previews(dossier: Dossier) -> tuple[Preview, ...]:
    found = [
        Preview(
            link.sources[0].source if link.sources else link.service, link.url, link.preview_url
        )
        for link in dossier.links
        if link.preview_url
    ]

    def rank(preview: Preview) -> int:
        return (
            PREVIEW_ORDER.index(preview.source)
            if preview.source in PREVIEW_ORDER
            else len(PREVIEW_ORDER)
        )

    return tuple(sorted(found, key=rank))


def _lead(relation: Relation) -> CatalogLead | None:
    """A relation that names a lead, with V1's source and relation wording."""
    source = relation.sources[0].source if relation.sources else ""
    target = relation.target
    group: LeadGroup
    title: str | None = None
    if relation.kind is RelationKind.SIMILAR and target.type is EntityType.RECORDING:
        if source != "lastfm" or not target.artist:
            return None
        group, wording, title, artist = (
            "lastfm_similar_tracks",
            "similar track",
            target.name,
            target.artist,
        )
    elif relation.kind is RelationKind.SIMILAR_ARTIST_RECORDING:
        if source != "listenbrainz" or not target.artist:
            return None
        group, wording, title, artist = (
            "lb_radio",
            "recording by a similar artist",
            target.name,
            target.artist,
        )
    elif relation.kind is RelationKind.SIMILAR and target.type is EntityType.ARTIST:
        if source not in {"lastfm", "listenbrainz"}:
            return None
        group = "lastfm_similar_artists" if source == "lastfm" else "lb_similar_artists"
        wording, artist = "similar artist", target.name
    elif relation.kind in MUSICBRAINZ_RELATIONS and source == "musicbrainz":
        group, artist = "musicbrainz_relations", target.name
        wording = (
            (relation.note or "").split(":", 1)[0].strip()
            if relation.kind is RelationKind.ASSOCIATED
            else MEMBER_RELATION
        )
    else:
        return None
    return CatalogLead(group, artist, title, SOURCE_LABELS[source], wording, target.url)


def _facts(dossier: Dossier) -> tuple[DossierFact, ...]:
    facts: list[DossierFact] = []
    for credit in dossier.credits:
        detail = f" ({credit.detail})" if credit.detail else ""
        facts.append(_fact("credit", f"{credit.role}: {credit.name}{detail}", credit.sources))
    for text in dossier.texts:
        heading = " — ".join(part for part in (text.publication, text.title) if part)
        body = text.body[:TEXT_EXCERPT]
        facts.append(
            _fact(
                f"text:{text.kind.value}", f"{heading}: {body}" if heading else body, text.sources
            )
        )
    for relation in dossier.relations:
        who = (
            f"{relation.target.artist} — {relation.target.name}"
            if relation.target.artist
            else relation.target.name
        )
        note = f" ({relation.note})" if relation.note else ""
        facts.append(_fact(f"relation:{relation.kind.value}", f"{who}{note}", relation.sources))
    return tuple(facts)


def _fact(kind: str, text: str, sources: list[Provenance]) -> DossierFact:
    first = sources[0] if sources else Provenance(source="")
    return DossierFact(kind, text, first.source, first.url)
