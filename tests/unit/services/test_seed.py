import pytest

from mixhound.infrastructure.cache import MemoryCache
from mixhound.models.domain import DiscoveryReference, DiscoveryRunInput, DiscoverySettings
from mixhound.services.audio import AudioAnalyzer
from mixhound.services.exclusions import Exclusions
from mixhound.services.identity import recording_key
from mixhound.services.ports import CatalogLead, CatalogRecording, LeadGroup, Preview
from mixhound.services.run_state import RunCancelledError
from mixhound.services.seed import Seeder, _interleave
from tests.fakes import FakeCatalog, FakeDecoder, FakeFetch
from tests.support.factories import clicks


def lead(
    group: LeadGroup, artist: str, title: str | None = None, relation: str = "x"
) -> CatalogLead:
    return CatalogLead(group, artist, title, "Src", relation)


def reference(artist: str, title: str, **changes: object) -> DiscoveryReference:
    return DiscoveryReference.model_validate(
        {"kind": "manual", "artist": artist, "title": title, **changes}
    )


def test_pool_interleaves_references_and_respects_exclusions() -> None:
    first = reference("Fogh Depot", "Anticyclone")
    second = reference("Bohren", "Prowler")
    run_input = DiscoveryRunInput(
        references=[first, second],
        settings=DiscoverySettings(excluded_artists=["Blocked"]),
    )
    seeds = [
        (
            first,
            [
                [
                    CatalogLead(
                        "lb_radio", "Povarovo", f"Song {index}", "ListenBrainz", "recording"
                    )
                    for index in range(5)
                ],
                [
                    CatalogLead(
                        "lastfm_similar_tracks", "Blocked", "X", "Last.fm", "similar track"
                    ),
                    CatalogLead("lastfm_similar_artists", "Fogh Depot", None, "Last.fm", "similar"),
                ],
            ],
        ),
        (
            second,
            [
                [
                    CatalogLead("lastfm_similar_tracks", "Dale Cooper", "Prior", "Last.fm", "s"),
                    CatalogLead("lastfm_similar_tracks", "Dale Cooper", "Fresh", "Last.fm", "s"),
                ],
            ],
        ),
    ]
    pool = _interleave(
        seeds,
        run_input,
        Exclusions(prior_keys=frozenset({recording_key("Dale Cooper", "Prior")})),
        limit=10,
    )
    assert [(item["artist"], item["title"]) for item in pool] == [
        ("Povarovo", "Song 0"),
        ("Dale Cooper", "Fresh"),
        ("Povarovo", "Song 1"),
    ]
    assert pool[0]["reference_id"] == str(first.id)
    assert pool[0]["via"] == "ListenBrainz recording of Fogh Depot — Anticyclone"
    assert len(_interleave(seeds, run_input, Exclusions(), limit=2)) == 2


def test_pool_excludes_reference_and_prior_artists_when_asked() -> None:
    first = reference("Low", "Words")
    run_input = DiscoveryRunInput(
        references=[first],
        settings=DiscoverySettings(
            exclude_reference_artists=True, exclude_previously_proposed_artists=True
        ),
    )
    seeds = [
        (
            first,
            [
                [
                    lead("lb_radio", "Low", "Other"),
                    lead("lb_radio", "Codeine", "D"),
                    lead("lb_radio", "Low", "Words"),
                    lead("lb_radio", "", "Nameless"),
                    lead("lb_radio", "Duster", "Inside Out"),
                ]
            ],
        )
    ]

    pool = _interleave(seeds, run_input, Exclusions(prior_artists=frozenset({"codeine"})), 10)

    assert [item["artist"] for item in pool] == ["Duster"]


async def test_seeder_collects_facts_and_leads_without_network() -> None:
    words = reference("Low", "Words", priority=5)
    catalog = FakeCatalog(
        {
            ("Low", "Words"): CatalogRecording(
                "Low",
                "Words",
                confirmed_by=("musicbrainz",),
                mbid="r1",
                artist_mbids=("m1",),
                tags={"musicbrainz": ("slowcore",), "lastfm": ("slowcore", "sad", "dream")},
                leads=(
                    lead("lastfm_similar_artists", "Ignored with an MBID"),
                    lead("lb_similar_artists", "Codeine"),
                    lead("musicbrainz_relations", "Alan Sparhawk", relation="member of band"),
                    lead("musicbrainz_relations", "Not a lead", relation="teacher"),
                ),
            )
        }
    )

    prepared = await Seeder(catalog, None).prepare(
        DiscoveryRunInput(references=[words]), Exclusions(), pool_size=10
    )

    facts = prepared.facts[words.id]
    assert (facts.mbid, facts.artist_mbids) == ("r1", ("m1",))
    assert facts.tags == ["slowcore", "sad", "dream"]
    assert [item["artist"] for item in prepared.pool] == ["Codeine", "Alan Sparhawk"]
    assert catalog.calls == [("reference", "Low", "Words", None, None)]


async def test_lead_groups_follow_v1_order_and_caps() -> None:
    leads = (
        *(lead("musicbrainz_relations", f"Rel {i}", relation="collaboration") for i in range(8)),
        *(lead("lastfm_similar_artists", f"Fm {i}") for i in range(10)),
        *(lead("lastfm_similar_tracks", f"Track {i}", "T") for i in range(20)),
        lead("lb_radio", "Radio", "R"),
    )
    words = reference("Low", "Words")
    catalog = FakeCatalog({("Low", "Words"): CatalogRecording("Low", "Words", leads=leads)})

    prepared = await Seeder(catalog, None).prepare(
        DiscoveryRunInput(references=[words]), Exclusions(), pool_size=100
    )

    artists = [str(item["artist"]) for item in prepared.pool]
    assert artists[:3] == ["Track 0", "Fm 0", "Rel 0"]
    assert sum(name.startswith("Track") for name in artists) == 15
    assert sum(name.startswith("Fm") for name in artists) == 8
    assert sum(name.startswith("Rel") for name in artists) == 6
    assert "Radio" not in artists


async def test_neutral_references_are_skipped_and_negative_ones_give_no_leads() -> None:
    neutral = reference("A", "1", polarity="neutral")
    negative = reference("B", "2", polarity="negative", priority=5)
    positive = reference("C", "3", priority=1)
    recording = CatalogRecording("X", "Y", leads=(lead("lastfm_similar_tracks", "Lead", "L"),))
    catalog = FakeCatalog({("B", "2"): recording, ("C", "3"): recording})

    prepared = await Seeder(catalog, None).prepare(
        DiscoveryRunInput(references=[neutral, negative, positive]), Exclusions(), pool_size=5
    )

    assert [call[1] for call in catalog.calls] == ["B", "C"]
    assert set(prepared.facts) == {negative.id, positive.id}
    assert [item["reference_id"] for item in prepared.pool] == [str(positive.id)]


async def test_catalog_outage_stops_further_lookups() -> None:
    catalog = FakeCatalog(down=True)
    references = [reference("A", "1"), reference("B", "2")]

    prepared = await Seeder(catalog, None).prepare(
        DiscoveryRunInput(references=references), Exclusions(), pool_size=5
    )

    assert len(catalog.calls) == 1
    assert prepared.pool == []
    assert all(facts.tags == [] for facts in prepared.facts.values())


async def test_check_aborts_preparation() -> None:
    def cancelled() -> None:
        raise RunCancelledError

    with pytest.raises(RunCancelledError):
        await Seeder(FakeCatalog(), None).prepare(
            DiscoveryRunInput(references=[reference("A", "1")]), Exclusions(), 5, cancelled
        )


async def test_reference_audio_comes_from_the_first_preview_that_decodes() -> None:
    previews = (
        Preview("itunes", "https://music.apple.com/x", "https://audio.example/broken.m4a"),
        Preview("spotify", "https://open.spotify.com/track/1", "https://audio.example/ok.mp3"),
    )
    catalog = FakeCatalog({("Low", "Words"): CatalogRecording("Low", "Words", previews=previews)})
    fetch = FakeFetch({"audio.example/broken.m4a": b"broken", "audio.example/ok.mp3": b"ok"})
    decoder = FakeDecoder({b"broken": ValueError("bad"), b"ok": clicks(100)})
    words = reference("Low", "Words")

    prepared = await Seeder(catalog, AudioAnalyzer(fetch, decoder, MemoryCache())).prepare(
        DiscoveryRunInput(references=[words]), Exclusions(), pool_size=5
    )

    facts = prepared.facts[words.id]
    assert facts.audio is not None
    assert facts.preview_url == "https://open.spotify.com/track/1"


async def test_reference_without_previews_has_no_audio() -> None:
    catalog = FakeCatalog({("Low", "Words"): CatalogRecording("Low", "Words")})
    analyzer = AudioAnalyzer(FakeFetch({}), FakeDecoder(), MemoryCache())
    words = reference("Low", "Words")

    prepared = await Seeder(catalog, analyzer).prepare(
        DiscoveryRunInput(references=[words]), Exclusions(), pool_size=5
    )

    assert prepared.facts[words.id].audio is None
    assert prepared.facts[words.id].preview_url is None
