from mixhound.models.domain import (
    DiscoveryResearchResult,
    DiscoveryRunInput,
    DiscoverySettings,
    DiscoverySource,
    DiscoveryTrack,
)
from mixhound.services.evidence import AUDIO_WEIGHT, MIN_VERIFIED_SCORE, publish_result, rank_score

EVIDENCE = DiscoverySource(
    url="https://label.bandcamp.com/track/x",
    source_type="Bandcamp",
    claim="Tags: slowcore",
    target_type="track",
    reachable=True,
    matches_recording=True,
)


def track(artist: str, score: int = 70, **changes: object) -> DiscoveryTrack:
    values: dict[str, object] = {
        "artist": artist,
        "title": f"{artist} song {score}",
        "similarity_score": score,
        "similarities": ["Hazy"],
        "sources": [EVIDENCE],
    }
    values.update(changes)
    return DiscoveryTrack.model_validate(values)


def publish(
    *tracks: DiscoveryTrack, settings: DiscoverySettings, confirmed: set[object] | None = None
) -> tuple[DiscoveryResearchResult, list[tuple[str, str, str]]]:
    result = DiscoveryResearchResult(results=list(tracks))
    ids = {item.id for item in tracks} if confirmed is None else confirmed
    rejections = publish_result(result, DiscoveryRunInput(settings=settings), ids)  # type: ignore[arg-type]
    return result, [(item.artist, item.reason, item.action) for item in rejections]


def test_rank_blends_measured_audio_similarity() -> None:
    assert rank_score(track("A", 80)) == 80
    blended = rank_score(track("A", 80, audio_similarity=40))
    assert blended == (1 - AUDIO_WEIGHT) * 80 + AUDIO_WEIGHT * 40


def test_per_artist_and_desired_count_limits_demote_and_drop() -> None:
    settings = DiscoverySettings(desired_count=2, max_per_artist=1)

    result, rejections = publish(
        track("A", 90),
        track("A", 80),
        track("B", 70),
        track("C", 60),
        settings=settings,
    )

    assert [(item.artist, item.similarity_score) for item in result.results] == [
        ("A", 90),
        ("B", 70),
    ]
    assert [(item.artist, item.similarity_score) for item in result.unverified_leads] == [("C", 60)]
    assert rejections == [
        ("A", "More recordings by this artist than allowed", "demoted"),
        ("C", "More verified results than requested", "demoted"),
        ("A", "More recordings by this artist than allowed", "dropped"),
    ]


def test_unconfirmed_weak_or_unexplained_candidates_stay_leads() -> None:
    unconfirmed = track("A")
    weak = track("B", MIN_VERIFIED_SCORE - 1)
    unexplained = track("C", similarities=[])
    unsourced = track("D", sources=[])
    claimed = track("E", verification="verified", sources=[])

    result, rejections = publish(
        unconfirmed,
        weak,
        unexplained,
        unsourced,
        claimed,
        settings=DiscoverySettings(),
        confirmed={weak.id, unexplained.id, unsourced.id, claimed.id},
    )

    assert result.results == []
    assert len(result.unverified_leads) == 5
    assert rejections == [("E", "Verified result lacks recording evidence", "demoted")]
    assert {item.verification for item in result.unverified_leads} == {"unverified"}
