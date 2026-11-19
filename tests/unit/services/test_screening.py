import pytest

from mixhound.models.domain import (
    DiscoveryListeningLink,
    DiscoveryPath,
    DiscoveryReference,
    DiscoveryResearchResult,
    DiscoveryRunInput,
    DiscoverySettings,
    DiscoverySource,
    DiscoveryTrack,
)
from mixhound.services.evidence import (
    ResearchEvidenceError,
    screen_research_result,
    validate_research_result,
)
from mixhound.services.exclusions import Exclusions


def track(**changes: object) -> DiscoveryTrack:
    values: dict[str, object] = {"artist": "Duster", "title": "Inside Out", "year": 1998}
    values.update(changes)
    return DiscoveryTrack.model_validate(values)


def screen(
    *tracks: DiscoveryTrack,
    settings: DiscoverySettings | None = None,
    references: list[DiscoveryReference] | None = None,
    exclusions: Exclusions | None = None,
) -> tuple[DiscoveryResearchResult, list[tuple[str, str]]]:
    result = DiscoveryResearchResult(results=list(tracks))
    run_input = DiscoveryRunInput(
        settings=settings or DiscoverySettings(), references=references or []
    )
    rejections = screen_research_result(result, run_input, exclusions or Exclusions())
    return result, [(item.reason, item.action) for item in rejections]


@pytest.mark.parametrize(
    ("candidate", "settings", "reason"),
    [
        (track(artist=" "), DiscoverySettings(), "Missing recording identity or duplicate"),
        (track(), DiscoverySettings(excluded_artists=["DUSTER"]), "Excluded artist"),
        (track(year=1990), DiscoverySettings(year_from=1995), "Outside requested year range"),
        (track(year=2001), DiscoverySettings(year_to=2000), "Outside requested year range"),
        (
            track(version_qualifier="cover"),
            DiscoverySettings(),
            "Cover lacks original-work attribution",
        ),
    ],
)
def test_excluded_candidates_are_dropped(
    candidate: DiscoveryTrack, settings: DiscoverySettings, reason: str
) -> None:
    result, rejections = screen(candidate, settings=settings)

    assert result.results == []
    assert rejections == [(reason, "dropped")]


def test_duplicates_references_and_history_are_dropped() -> None:
    reference = DiscoveryReference(kind="manual", artist="Low", title="Words")
    settings = DiscoverySettings(
        exclude_reference_artists=True, exclude_previously_proposed_artists=True
    )
    exclusions = Exclusions(prior_artists=frozenset({"codeine"}))

    result, rejections = screen(
        track(),
        track(),
        track(artist="Low", title="Words"),
        track(artist="Low", title="Other"),
        track(artist="Codeine", title="D"),
        settings=settings,
        references=[reference],
        exclusions=exclusions,
    )

    assert [item.title for item in result.results] == ["Inside Out"]
    assert rejections == [
        ("Missing recording identity or duplicate", "dropped"),
        ("Reference recording", "dropped"),
        ("Reference artist", "dropped"),
        ("Previously proposed artist", "dropped"),
    ]


def test_withdrawable_claims_are_corrected_not_dropped() -> None:
    unknown = DiscoveryReference(kind="manual", artist="X", title="Y").id
    candidate = track(
        audio_analyzed=True,
        closest_reference_ids=[unknown],
        preferred_listening_url="https://gone.example/1",
        discovery_paths=[
            DiscoveryPath(strategy=" ", explanation="x"),
            DiscoveryPath(strategy="label", explanation="x", source_url="ftp://x"),
        ],
        sources=[DiscoverySource(url="mailto:x", source_type="x", claim="x", target_type="other")],
        listening_links=[
            DiscoveryListeningLink(url="javascript:x", service="x", target_type="track"),
            DiscoveryListeningLink(
                url="https://open.spotify.com/track/1",
                service="Spotify",
                target_type="track",
                reachable=None,
                playback_available=True,
            ),
        ],
    )

    result, rejections = screen(candidate)

    [kept] = result.results
    assert not kept.audio_analyzed
    assert kept.closest_reference_ids == []
    assert kept.discovery_paths == []
    assert kept.sources == []
    assert [link.playback_available for link in kept.listening_links] == [None]
    assert kept.preferred_listening_url is None
    assert {reason for reason, _ in rejections} == {
        "Audio analysis claim lacks evidence",
        "Unknown closest reference",
        "Incomplete discovery path",
        "Invalid discovery URL",
        "Invalid source URL",
        "Invalid listening URL",
        "Unsupported playback claim",
        "Preferred listening URL is absent",
    }
    assert {action for _, action in rejections} == {"corrected"}


def test_leads_are_screened_too_and_rejections_serialize() -> None:
    result = DiscoveryResearchResult(unverified_leads=[track(artist="")])

    [rejection] = screen_research_result(result, DiscoveryRunInput())

    assert result.unverified_leads == []
    assert rejection.as_dict() == {
        "artist": "",
        "title": "Inside Out",
        "reason": "Missing recording identity or duplicate",
        "action": "dropped",
    }


def test_strict_validation_rejects_too_many_verified_results() -> None:
    def verified(title: str) -> DiscoveryTrack:
        return track(
            title=title,
            verification="verified",
            similarity_score=70,
            similarities=["Hazy"],
            sources=[
                DiscoverySource(
                    url="https://label.bandcamp.com/track/x",
                    source_type="Bandcamp",
                    claim="Tags",
                    target_type="track",
                    reachable=True,
                    matches_recording=True,
                )
            ],
        )

    with pytest.raises(ResearchEvidenceError, match="More verified results"):
        validate_research_result(
            DiscoveryResearchResult(results=[verified("A"), verified("B")]),
            DiscoveryRunInput(settings=DiscoverySettings(desired_count=1)),
        )
