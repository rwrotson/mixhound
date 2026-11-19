import math
from typing import Literal

from mixhound.models.domain import (
    DiscoveryResearchResult,
    DiscoveryRunInput,
    DiscoverySource,
    DiscoveryTrack,
    SonicProfile,
)
from mixhound.models.research import ResearchCandidate, ResearchOutput, ResearchPath, ResearchSource
from mixhound.services.budgets import DEPTH_BUDGETS
from mixhound.services.evidence import has_sonic_evidence, merge_outputs, publish_result
from mixhound.services.strategies import (
    STRATEGIES,
    split_pool,
    strategy_budget,
    strategy_max_usd,
)


def candidate(
    artist: str,
    title: str,
    score: int,
    strategy: str = "path",
    links: tuple[str, str | None] = ("https://example.com/a", None),
    **changes: object,
) -> ResearchCandidate:
    url, listen = links
    fields: dict[str, object] = {
        "artist": artist,
        "title": title,
        "artist_alt": None,
        "title_alt": None,
        "album": None,
        "year": None,
        "version_qualifier": None,
        "original_artist": None,
        "original_title": None,
        "genres": [],
        "similarity_score": score,
        "confidence": "low",
        "researched": True,
        "similarities": ["Hazy"],
        "differences": [],
        "closest_reference_ids": [],
        "discovery_paths": [
            ResearchPath(strategy=strategy, explanation=f"via {strategy}", source_url=None),
        ],
        "sources": [ResearchSource(url=url, claim="Release", target_type="album")],
        "listening_urls": [listen] if listen else [],
    }
    fields.update(changes)
    return ResearchCandidate.model_validate(fields)


def test_budget_is_divided_between_strategies() -> None:
    full = DEPTH_BUDGETS["deep"]

    assert strategy_budget("deep").candidates == math.ceil(full.candidates / len(STRATEGIES))
    assert strategy_budget("quick").pool == DEPTH_BUDGETS["quick"].pool
    assert strategy_max_usd(3.0) == 1.0


def test_pool_split_sends_relations_to_collaborators() -> None:
    pool: list[dict[str, object]] = [
        {"artist": "A", "via": "MusicBrainz member of band of Low — Words"},
        {"artist": "B", "via": "ListenBrainz similar artist of Low — Words"},
        {"artist": "C", "via": "Last.fm similar track of Low — Words"},
        {"artist": "D", "via": "ListenBrainz radio of Low — Words"},
    ]
    shares = split_pool(pool)
    assert [item["artist"] for item in shares["collaborators"]] == ["A"]
    assert [item["artist"] for item in shares["label_scene"]] == ["B", "D"]
    assert [item["artist"] for item in shares["traits"]] == ["C"]


def test_merge_collapses_duplicates_and_unites_paths() -> None:
    first = ResearchOutput(
        sonic_profile=SonicProfile(),
        candidates=[
            candidate(
                "Duster",
                "Inside Out",
                60,
                "collaborators",
                ("https://a.example/1", "https://youtu.be/one"),
            ),
            candidate("Codeine", "D", 50, "collaborators"),
        ],
    )
    second = ResearchOutput(
        sonic_profile=SonicProfile(),
        candidates=[
            candidate(
                "duster",
                "Inside out",
                75,
                "traits",
                ("https://b.example/2", "https://youtu.be/two"),
            ),
        ],
    )
    merged, overlap = merge_outputs(
        {"collaborators": first, "traits": second},
        SonicProfile(summary="Shared"),
    )
    assert overlap == 1
    assert merged.sonic_profile.summary == "Shared"
    duster = merged.candidates[0]
    assert len(merged.candidates) == 2
    assert duster.similarity_score == 75
    assert {path.strategy for path in duster.discovery_paths} == {"collaborators", "traits"}
    assert {source.url for source in duster.sources} == {
        "https://a.example/1",
        "https://b.example/2",
    }
    assert set(duster.listening_urls) == {"https://youtu.be/one", "https://youtu.be/two"}


def test_merge_keeps_a_recording_researched_if_any_strategy_studied_it() -> None:
    seen = ResearchOutput(
        sonic_profile=SonicProfile(),
        candidates=[candidate("Duster", "Inside Out", 30, "traits", researched=False)],
    )
    studied = ResearchOutput(
        sonic_profile=SonicProfile(summary="From a strategy"),
        candidates=[candidate("Duster", "Inside Out", 20, "label_scene")],
    )
    merged, _ = merge_outputs({"traits": seen, "label_scene": studied})
    assert [item.researched for item in merged.candidates] == [True]
    assert merged.sonic_profile.summary == "From a strategy"


def test_candidates_without_sonic_evidence_stay_leads() -> None:
    def track(*urls: tuple[str, Literal["artist", "track", "album"]]) -> DiscoveryTrack:
        return DiscoveryTrack(
            artist="Duster",
            title="Inside Out",
            similarity_score=70,
            similarities=["Hazy"],
            sources=[
                DiscoverySource(
                    url="https://musicbrainz.org/recording/1",
                    source_type="MusicBrainz",
                    claim="Recording",
                    target_type="track",
                    reachable=True,
                    matches_recording=True,
                ),
                *(
                    DiscoverySource(url=url, source_type="web", claim="Sound", target_type=kind)
                    for url, kind in urls
                ),
            ],
        )

    guessed = track(
        ("https://listenbrainz.org/artist/1", "artist"),
        ("https://open.spotify.com/track/1", "track"),
    )
    researched = track(("https://example.bandcamp.com/album/x", "album"))
    assert not has_sonic_evidence(guessed)
    assert has_sonic_evidence(researched)
    for candidate_track, verified in ((guessed, False), (researched, True)):
        candidate_track.verification = "verified"  # As claimed, so a demotion is recorded.
        result = DiscoveryResearchResult(results=[candidate_track])
        rejections = publish_result(result, DiscoveryRunInput(), {candidate_track.id})
        assert bool(result.results) is verified
        assert bool(result.unverified_leads) is not verified
        assert [item.reason for item in rejections] == ([] if verified else ["No sonic evidence"])
