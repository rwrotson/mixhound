from uuid import uuid4

from mixhound.models.domain import (
    DiscoveryReference,
    DiscoveryResearchResult,
    DiscoveryRunInput,
    DiscoveryTrack,
    SonicProfile,
)
from mixhound.models.research import ResearchOutput, ResearchPath, ResearchSource
from mixhound.services.evidence import candidate_track, research_result
from tests.support.factories import candidate


def test_candidate_becomes_an_unconfirmed_track_with_classified_links() -> None:
    reference = DiscoveryReference(kind="manual", artist="Low", title="Words")
    found = candidate(
        artist="Самцы Дронта (Samtsi Dronta)",
        title="Song",
        genres=[str(index) for index in range(10)],
        closest_reference_ids=[str(reference.id), str(reference.id), "not-a-uuid", str(uuid4())],
        sources=[
            ResearchSource(
                url="https://label.bandcamp.com/album/x", claim="Tags", target_type="album"
            )
        ],
        discovery_paths=[ResearchPath(strategy="label", explanation="Same label", source_url=None)],
        listening_urls=[
            "https://open.spotify.com/track/1",
            "https://open.spotify.com/track/1",
            "https://label.bandcamp.com/album/x",
            "https://music.apple.com/us/album/x?i=1",
            "https://www.deezer.com/track/1",
            "https://www.example.org/page",
        ],
    )

    track = candidate_track(found, {reference.id})

    assert (track.artist, track.artist_alt) == ("Самцы Дронта", "Samtsi Dronta")
    assert track.closest_reference_ids == [reference.id]
    assert len(track.genres) == 8
    assert track.verification == "unverified"
    assert track.sources[0].source_type == "Bandcamp"
    assert track.discovery_paths[0].strategy == "label"
    assert [(link.service, link.target_type) for link in track.listening_links] == [
        ("Spotify", "track"),
        ("Bandcamp", "album"),
        ("Apple Music", "track"),
        ("Deezer", "track"),
        ("example.org", "other"),
    ]


def test_explicit_alternate_name_is_kept() -> None:
    track = candidate_track(candidate(artist=" Low ", artist_alt=" Лоу "), set())

    assert (track.artist, track.artist_alt) == ("Low", "Лоу")
    assert candidate_track(candidate(artist_alt="  "), set()).artist_alt is None


def test_research_result_converts_output_and_passes_built_results_through() -> None:
    reference = DiscoveryReference(kind="manual", artist="Low", title="Words")
    run_input = DiscoveryRunInput(references=[reference])
    output = ResearchOutput(
        sonic_profile=SonicProfile(summary="Slow"),
        candidates=[candidate(closest_reference_ids=[str(reference.id)])],
    )
    built = DiscoveryResearchResult(results=[DiscoveryTrack(artist="A", title="B")])

    result = research_result(output, run_input)

    assert result.sonic_profile == SonicProfile(summary="Slow")
    assert result.results[0].closest_reference_ids == [reference.id]
    assert research_result(built, run_input) is built
