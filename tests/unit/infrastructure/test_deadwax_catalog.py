from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from deadwax import (
    AdapterError,
    Capability,
    Credit,
    Deadwax,
    DeadwaxError,
    Dossier,
    Entity,
    EntityKind,
    EntityType,
    ErrorKind,
    ExternalId,
    Identity,
    IdKind,
    Link,
    LinkTarget,
    LookupConfig,
    Provenance,
    Query,
    Relation,
    RelationKind,
    Request,
    Scope,
    Sourced,
    Tag,
    Text,
    TextKind,
)
from pydantic import SecretStr

from mixhound.config import DeadwaxSettings
from mixhound.infrastructure.catalogs.deadwax import (
    CANDIDATE_FACETS,
    DeadwaxCatalog,
    lookup_config,
    open_catalog,
    recording,
)
from mixhound.services.ports import CatalogLead, CatalogUnavailableError, DossierFact, Preview

MB = "https://musicbrainz.org"


def by(*sources: str) -> list[Provenance]:
    return [Provenance(source=source, url=f"https://{source}.example/track") for source in sources]


def dossier(**facts: object) -> Dossier:
    identity = Identity(
        artist=Sourced(value="Radiohead", sources=by("musicbrainz", "itunes", "lastfm")),
        title=Sourced(value="Reckoner", sources=by("musicbrainz", "itunes", "odesli")),
    )
    return Dossier.model_validate(
        {
            "query": Query(kind=EntityKind.TRACK, artist="Radiohead", title="Reckoner"),
            "identity": identity,
            "generated_at": datetime(2026, 11, 20, tzinfo=UTC),
            **facts,
        }
    )


def relation(
    kind: RelationKind,
    source: str,
    name: str,
    *,
    track_by: str | None = None,
    note: str | None = None,
) -> Relation:
    """A relation to an artist, or to a recording when `track_by` names its artist ("" for none)."""
    target = EntityType.ARTIST if track_by is None else EntityType.RECORDING
    return Relation(
        kind=kind,
        target=Entity(type=target, name=name, artist=track_by or None, url=f"{MB}/{name}"),
        note=note,
        scope=Scope.ARTIST,
        sources=[Provenance(source=source)],
    )


class FakeClient:
    def __init__(self, result: Dossier | Exception) -> None:
        self.result = result
        self.requests: list[Request] = []

    async def lookup(self, request: Request) -> Dossier:
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def catalog(result: Dossier | Exception) -> tuple[DeadwaxCatalog, FakeClient]:
    client = FakeClient(result)
    return DeadwaxCatalog(client), client  # type: ignore[arg-type]  # duck-typed Deadwax


def test_identity_is_confirmed_by_catalogs_that_matched_artist_and_title() -> None:
    found = recording(
        dossier(
            identity=Identity(
                artist=Sourced(value="Radiohead", sources=by("musicbrainz", "itunes", "lastfm")),
                title=Sourced(value="Reckoner", sources=by("musicbrainz", "itunes", "odesli")),
                date=Sourced(value="2007-10-10", sources=by("musicbrainz")),
                isrcs=[Sourced(value="GBSTK0700005", sources=by("musicbrainz"))] * 2,
                external_ids=[
                    ExternalId(
                        kind=IdKind.MUSICBRAINZ_RECORDING, value="rec", sources=by("musicbrainz")
                    ),
                    ExternalId(
                        kind=IdKind.MUSICBRAINZ_ARTIST, value="a1", sources=by("musicbrainz")
                    ),
                    ExternalId(
                        kind=IdKind.MUSICBRAINZ_ARTIST, value="a2", sources=by("musicbrainz")
                    ),
                ],
            )
        )
    )

    assert found is not None
    assert found.confirmed
    assert found.confirmed_by == ("musicbrainz", "itunes")
    assert (found.mbid, found.artist_mbids, found.isrcs, found.year) == (
        "rec",
        ("a1", "a2"),
        ("GBSTK0700005",),
        2007,
    )
    assert [(page.source, page.url) for page in found.pages] == [
        ("musicbrainz", "https://musicbrainz.example/track"),
        ("itunes", "https://itunes.example/track"),
    ]


def test_a_recording_only_aggregators_know_is_not_confirmed() -> None:
    found = recording(
        dossier(
            identity=Identity(
                artist=Sourced(value="Radiohead", sources=by("lastfm", "odesli")),
                title=Sourced(value="Reckoner", sources=by("lastfm", "odesli")),
                date=Sourced(value="unknown", sources=by("lastfm")),
            )
        )
    )

    assert found is not None
    assert not found.confirmed
    assert (found.pages, found.mbid, found.year) == ((), None, None)


def test_a_dossier_without_an_artist_is_not_a_recording() -> None:
    assert (
        recording(dossier(identity=Identity(title=Sourced(value="X", sources=by("itunes")))))
        is None
    )


def test_tags_are_grouped_by_source_in_order() -> None:
    found = recording(
        dossier(
            tags=[
                Tag(scope=Scope.RECORDING, name="art rock", sources=by("musicbrainz", "lastfm")),
                Tag(scope=Scope.RECORDING, name="alternative", sources=by("lastfm")),
                Tag(scope=Scope.RECORDING, name="art rock", sources=by("lastfm")),
            ]
        )
    )

    assert found is not None
    assert found.tags == {"musicbrainz": ("art rock",), "lastfm": ("art rock", "alternative")}


def test_links_and_previews_keep_their_source_and_preview_order() -> None:
    found = recording(
        dossier(
            links=[
                Link(
                    service="bandcamp",
                    url="https://a.bandcamp.com/track/r",
                    target=LinkTarget.TRACK,
                    preview_url="https://bc/stream",
                    sources=by("bandcamp"),
                ),
                Link(
                    service="youtube",
                    url="https://www.youtube.com/watch?v=x",
                    target=LinkTarget.TRACK,
                    sources=by("youtube"),
                ),
                Link(
                    service="deezer",
                    url="https://www.deezer.com/track/1",
                    target=LinkTarget.TRACK,
                    preview_url="https://dz/clip",
                    sources=by("deezer"),
                ),
                Link(
                    service="itunes",
                    url="https://music.apple.com/x",
                    target=LinkTarget.TRACK,
                    preview_url="https://it/clip",
                    sources=by("itunes"),
                ),
                Link(
                    service="spotify",
                    url="https://open.spotify.com/album/1",
                    target=LinkTarget.ALBUM,
                    preview_url="https://sp/clip",
                    sources=[],
                ),
            ]
        )
    )

    assert found is not None
    assert [(link.source, link.target) for link in found.links] == [
        ("bandcamp", "track"),
        ("youtube", "track"),
        ("deezer", "track"),
        ("itunes", "track"),
        ("spotify", "album"),
    ]
    assert found.previews == (
        Preview("itunes", "https://music.apple.com/x", "https://it/clip"),
        Preview("spotify", "https://open.spotify.com/album/1", "https://sp/clip"),
        Preview("bandcamp", "https://a.bandcamp.com/track/r", "https://bc/stream"),
        Preview("deezer", "https://www.deezer.com/track/1", "https://dz/clip"),
    )


def test_relations_become_leads_with_v1_wording() -> None:
    found = recording(
        dossier(
            relations=[
                relation(
                    RelationKind.SIMILAR,
                    "lastfm",
                    "Nude",
                    track_by="Radiohead",
                    note="match 0.9",
                ),
                relation(
                    RelationKind.SIMILAR,
                    "listenbrainz",
                    "Videotape",
                    track_by="Radiohead",
                ),
                relation(RelationKind.SIMILAR, "lastfm", "Untitled", track_by=""),
                relation(
                    RelationKind.SIMILAR_ARTIST_RECORDING,
                    "listenbrainz",
                    "Black Swan",
                    track_by="Thom Yorke",
                ),
                relation(
                    RelationKind.SIMILAR_ARTIST_RECORDING, "listenbrainz", "Anonymous", track_by=""
                ),
                relation(RelationKind.SIMILAR, "listenbrainz", "Atoms for Peace"),
                relation(RelationKind.SIMILAR, "lastfm", "Portishead"),
                relation(RelationKind.SIMILAR, "discogs", "Elsewhere"),
                relation(RelationKind.MEMBER, "musicbrainz", "Jonny Greenwood", note="guitar"),
                relation(RelationKind.PROJECT, "musicbrainz", "The Smile"),
                relation(
                    RelationKind.ASSOCIATED,
                    "musicbrainz",
                    "Nigel Godrich",
                    note="producer: original",
                ),
                relation(RelationKind.MEMBER, "discogs", "Ed O'Brien"),
                relation(RelationKind.LABEL, "musicbrainz", "XL"),
            ]
        )
    )

    assert found is not None
    assert found.leads == (
        CatalogLead(
            "lastfm_similar_tracks", "Radiohead", "Nude", "Last.fm", "similar track", f"{MB}/Nude"
        ),
        CatalogLead(
            "lb_radio",
            "Thom Yorke",
            "Black Swan",
            "ListenBrainz",
            "recording by a similar artist",
            f"{MB}/Black Swan",
        ),
        CatalogLead(
            "lb_similar_artists",
            "Atoms for Peace",
            None,
            "ListenBrainz",
            "similar artist",
            f"{MB}/Atoms for Peace",
        ),
        CatalogLead(
            "lastfm_similar_artists",
            "Portishead",
            None,
            "Last.fm",
            "similar artist",
            f"{MB}/Portishead",
        ),
        CatalogLead(
            "musicbrainz_relations",
            "Jonny Greenwood",
            None,
            "MusicBrainz",
            "member of band",
            f"{MB}/Jonny Greenwood",
        ),
        CatalogLead(
            "musicbrainz_relations",
            "The Smile",
            None,
            "MusicBrainz",
            "member of band",
            f"{MB}/The Smile",
        ),
        CatalogLead(
            "musicbrainz_relations",
            "Nigel Godrich",
            None,
            "MusicBrainz",
            "producer",
            f"{MB}/Nigel Godrich",
        ),
    )


def test_credits_texts_and_relations_become_facts_with_sources() -> None:
    found = recording(
        dossier(
            credits=[
                Credit(
                    scope=Scope.RECORDING,
                    role="producer",
                    name="Nigel Godrich",
                    sources=by("discogs"),
                ),
                Credit(
                    scope=Scope.RECORDING,
                    role="vocals",
                    name="Thom Yorke",
                    detail="lead",
                    sources=[],
                ),
            ],
            texts=[
                Text(
                    scope=Scope.RECORDING,
                    kind=TextKind.REVIEW,
                    body="x" * 700,
                    publication="Pitchfork",
                    title="In Rainbows",
                    sources=by("pitchfork"),
                ),
                Text(
                    scope=Scope.RECORDING,
                    kind=TextKind.WIKI,
                    body="A song.",
                    sources=by("wikipedia"),
                ),
            ],
            relations=[
                relation(RelationKind.LABEL, "musicbrainz", "XL"),
                relation(
                    RelationKind.REMIX_OF,
                    "discogs",
                    "Reckoner",
                    track_by="Radiohead",
                    note="remix",
                ),
            ],
        )
    )

    assert found is not None
    assert found.facts == (
        DossierFact(
            "credit", "producer: Nigel Godrich", "discogs", "https://discogs.example/track"
        ),
        DossierFact("credit", "vocals: Thom Yorke (lead)", "", None),
        DossierFact(
            "text:review",
            "Pitchfork — In Rainbows: " + "x" * 600,
            "pitchfork",
            "https://pitchfork.example/track",
        ),
        DossierFact("text:wiki", "A song.", "wikipedia", "https://wikipedia.example/track"),
        DossierFact("relation:label", "XL", "musicbrainz", None),
        DossierFact("relation:remix_of", "Radiohead — Reckoner (remix)", "discogs", None),
    )


async def test_reference_asks_for_identity_tags_leads_and_previews() -> None:
    adapter, client = catalog(dossier())

    found = await adapter.reference("Radiohead", "Reckoner", version="Remix", isrc="gbstk0700005")

    assert found is not None
    request = client.requests[0]
    assert request.facets == frozenset(
        {Capability.IDENTITY, Capability.TAGS, Capability.RELATIONS, Capability.PREVIEWS}
    )
    assert request.tags_per_source == 8
    assert request.query.title == "Reckoner (Remix)"
    assert request.query.ids == {IdKind.ISRC: "GBSTK0700005"}
    assert request.relations is not None
    assert {(limit.kind, limit.target, limit.per_source) for limit in request.relations} == {
        (RelationKind.SIMILAR, EntityType.RECORDING, 15),
        (RelationKind.SIMILAR, EntityType.ARTIST, 8),
        (RelationKind.SIMILAR_ARTIST_RECORDING, None, 16),
        (RelationKind.MEMBER, None, 6),
        (RelationKind.PROJECT, None, 6),
        (RelationKind.ASSOCIATED, None, 6),
    }


async def test_candidate_asks_for_links_and_descriptive_facets() -> None:
    adapter, client = catalog(dossier())

    await adapter.candidate("Radiohead", "Reckoner (Remix)", version="remix")

    request = client.requests[0]
    assert request.facets == CANDIDATE_FACETS
    assert request.query.title == "Reckoner (Remix)"
    assert request.query.ids == {}
    assert request.relations is not None
    assert all(limit.kind is not RelationKind.SIMILAR for limit in request.relations)


async def test_an_unknown_recording_is_none() -> None:
    missing = dossier(
        identity=Identity(),
        errors=[AdapterError(source="musicbrainz", kind=ErrorKind.NOT_FOUND, reason="no match")],
    )

    assert await catalog(missing)[0].candidate("Nobody", "Nothing") is None


@pytest.mark.parametrize(
    "kind", [ErrorKind.FAILED, ErrorKind.TIMEOUT, ErrorKind.BLOCKED, ErrorKind.RATE_LIMITED]
)
async def test_a_failing_identity_source_makes_the_catalog_unavailable(kind: ErrorKind) -> None:
    failed = dossier(
        identity=Identity(), errors=[AdapterError(source="musicbrainz", kind=kind, reason="down")]
    )

    with pytest.raises(CatalogUnavailableError, match="MusicBrainz"):
        await catalog(failed)[0].candidate("Radiohead", "Reckoner")


async def test_a_deadwax_error_makes_the_catalog_unavailable() -> None:
    with pytest.raises(CatalogUnavailableError, match="broken"):
        await catalog(DeadwaxError("broken"))[0].reference("Radiohead", "Reckoner")


def test_lookup_config_maps_settings_and_secrets(tmp_path: Path) -> None:
    settings = DeadwaxSettings(
        contact="ops@example.org",
        adapter_timeout=12,
        sources=["musicbrainz", "itunes"],
        cache_enabled=False,
        lastfm_api_key=SecretStr("lf"),
        spotify_client_id=SecretStr("id"),
    )

    config = lookup_config(settings, tmp_path)

    assert config.http.contact == "ops@example.org"
    assert (config.adapter_timeout, config.default_sources) == (12, ("musicbrainz", "itunes"))
    assert (config.cache.enabled, config.cache.path) == (False, tmp_path / "deadwax-cache")
    assert config.sources.lastfm.api_key == SecretStr("lf")
    assert config.sources.spotify.client_id == SecretStr("id")
    assert config.sources.discogs.token is None
    assert (
        lookup_config(DeadwaxSettings(), tmp_path).default_sources == LookupConfig().default_sources
    )


async def test_real_client_runs_offline_over_a_mock_transport(tmp_path: Path) -> None:
    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Not Found"})

    config = lookup_config(DeadwaxSettings(sources=["musicbrainz"], cache_enabled=False), tmp_path)
    async with Deadwax(config, transport=httpx.MockTransport(not_found)) as client:
        assert await DeadwaxCatalog(client).candidate("Nobody", "Nothing") is None


async def test_open_catalog_opens_one_client(tmp_path: Path) -> None:
    async with open_catalog(DeadwaxSettings(cache_enabled=False), tmp_path) as opened:
        assert isinstance(opened.client, Deadwax)
