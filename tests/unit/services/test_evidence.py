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
    notes_output,
    publish_result,
    trim_to_target,
    validate_research_result,
)
from mixhound.services.exclusions import Exclusions
from mixhound.services.identity import recording_key


def candidate(**changes: object) -> DiscoveryTrack:
    values: dict[str, object] = {
        "artist": "Fictional Ensemble",
        "title": "Specific Recording",
        "year": 2001,
        "verification": "verified",
        "similarity_score": 73,
        "similarities": ["Sparse arrangement and restrained vocals"],
        "sources": [
            DiscoverySource(
                url="https://example.org/track",
                source_type="release",
                claim="Release lists this recording",
                target_type="track",
                reachable=True,
                matches_recording=True,
            ),
        ],
    }
    values.update(changes)
    return DiscoveryTrack.model_validate(values)


def check(
    track: DiscoveryTrack,
    settings: DiscoverySettings | None = None,
    prior_keys: frozenset[str] = frozenset(),
) -> DiscoveryResearchResult:
    return validate_research_result(
        DiscoveryResearchResult(results=[track]),
        DiscoveryRunInput(settings=settings or DiscoverySettings()),
        Exclusions(prior_keys=prior_keys),
    )


def test_slowcore_fixture_requires_specific_evidence_and_respects_exclusions() -> None:
    track = candidate(genres=["slowcore"])
    assert check(track).results == [track]
    with pytest.raises(ResearchEvidenceError, match="recording evidence"):
        check(candidate(sources=[]))
    with pytest.raises(ResearchEvidenceError, match="Previously proposed"):
        check(track, prior_keys=frozenset({recording_key(track.artist, track.title)}))


def test_review_can_confirm_recording_without_being_a_track_page() -> None:
    review = DiscoverySource(
        url="https://example.org/review",
        source_type="review",
        claim="Names and describes this recording",
        target_type="review",
        reachable=True,
        matches_recording=True,
    )
    track = candidate(sources=[review])
    assert check(track).results == [track]


def test_electronic_cover_fixture_preserves_performer_and_original() -> None:
    cover = candidate(
        artist="Electronic Performer",
        title="A Song",
        version_qualifier="cover",
        original_artist="Original Writer",
        original_title="A Song",
    )
    assert check(cover).results == [cover]
    with pytest.raises(ResearchEvidenceError, match="original-work"):
        check(cover.model_copy(update={"original_artist": None}))
    assert recording_key(cover.artist, cover.title, qualifier="cover") != recording_key(
        "Original Writer",
        "A Song",
    )


def test_feedback_refinement_fixture_excludes_rejected_and_prior_recordings() -> None:
    track = candidate()
    key = recording_key(track.artist, track.title)
    with pytest.raises(ResearchEvidenceError, match="Previously proposed"):
        check(track, prior_keys=frozenset({key}))
    with pytest.raises(ResearchEvidenceError, match="Reference artist"):
        validate_research_result(
            DiscoveryResearchResult(results=[track]),
            DiscoveryRunInput(
                references=[DiscoveryReference(kind="manual", artist=track.artist, title="Other")],
                settings=DiscoverySettings(exclude_reference_artists=True),
            ),
        )


def test_unavailable_source_fixture_rejects_false_playback() -> None:
    with pytest.raises(ResearchEvidenceError, match="recording evidence"):
        check(
            candidate(
                sources=[
                    DiscoverySource(
                        url="https://example.org/track",
                        source_type="release",
                        claim="Listed",
                        target_type="track",
                        reachable=False,
                        matches_recording=True,
                    ),
                ],
            ),
        )
    album_listing = DiscoverySource(
        url="https://example.org/album",
        source_type="release",
        claim="Lists this recording",
        target_type="album",
        reachable=True,
        matches_recording=True,
    )
    assert check(candidate(sources=[album_listing])).results
    album_link = DiscoveryListeningLink(
        url="https://example.org/album",
        service="Catalog",
        target_type="album",
        reachable=True,
        playback_available=True,
    )
    assert check(candidate(sources=[album_listing], listening_links=[album_link])).results
    with pytest.raises(ResearchEvidenceError, match="playback"):
        check(candidate(listening_links=[album_link]))
    with pytest.raises(ResearchEvidenceError, match="playback"):
        check(
            candidate(
                sources=[album_listing],
                listening_links=[album_link.model_copy(update={"reachable": False})],
            ),
        )
    with pytest.raises(ResearchEvidenceError, match="playback"):
        check(
            candidate(
                sources=[album_listing.model_copy(update={"matches_recording": False})],
                listening_links=[album_link],
            ),
        )
    with pytest.raises(ResearchEvidenceError, match="Audio analysis"):
        check(candidate(audio_analyzed=True))
    lead = candidate(verification="unverified", sources=[])
    assert validate_research_result(
        DiscoveryResearchResult(unverified_leads=[lead]),
        DiscoveryRunInput(),
    ).unverified_leads == [lead]


def test_discovery_path_and_reference_consistency() -> None:
    track = candidate(
        discovery_paths=[
            DiscoveryPath(
                strategy="Label",
                explanation="Same label",
                source_url="https://example.org/label",
            ),
        ],
    )
    assert check(track).results[0].discovery_paths[0].strategy == "Label"
    with pytest.raises(ResearchEvidenceError, match="discovery URL"):
        check(
            candidate(
                discovery_paths=[
                    DiscoveryPath(
                        strategy="Label",
                        explanation="Same label",
                        source_url="file:///tmp/label",
                    ),
                ],
            ),
        )
    with pytest.raises(ResearchEvidenceError, match="Preferred listening"):
        check(candidate(preferred_listening_url="https://example.org/track"))


def test_unresearched_candidate_stays_a_lead_even_when_confirmed() -> None:
    studied, seen = (
        candidate(),
        candidate(artist="Other Ensemble", researched=False, verification="unverified"),
    )
    result = DiscoveryResearchResult(results=[studied, seen])
    rejections = publish_result(result, DiscoveryRunInput(), {studied.id, seen.id})
    assert result.results == [studied]
    assert result.unverified_leads == [seen]
    assert seen.verification == "unverified"
    assert rejections == []


def test_trim_keeps_researched_then_best_scored_candidates() -> None:
    low = candidate(artist="Low Score", similarity_score=40)
    high = candidate(artist="High Score", similarity_score=80)
    seen = candidate(artist="Seen Only", similarity_score=90, researched=False)
    result = DiscoveryResearchResult(results=[low, seen], unverified_leads=[high])
    assert trim_to_target(DiscoveryResearchResult(results=[low]), 1) == []
    rejections = trim_to_target(result, 2)
    assert result.results == [low]
    assert result.unverified_leads == [high]
    assert [(item.artist, item.reason, item.action) for item in rejections] == [
        ("Seen Only", "Below research target", "dropped"),
    ]


def test_notes_without_the_flag_count_as_researched() -> None:
    [line] = notes_output(['{"artist": "A", "title": "B", "similarity_score": 50}']).candidates
    assert line.researched is True


def test_undated_recording_is_a_lead_when_a_year_range_is_set() -> None:
    undated = candidate(year=None)
    run_input = DiscoveryRunInput(settings=DiscoverySettings(year_from=1980))
    result = DiscoveryResearchResult(results=[undated])
    rejections = publish_result(result, run_input, {undated.id})
    assert result.unverified_leads == [undated]
    assert [(item.reason, item.action) for item in rejections] == [
        ("Year not confirmed", "demoted"),
    ]
    open_range = DiscoveryResearchResult(results=[candidate(year=None)])
    publish_result(open_range, DiscoveryRunInput(), {open_range.results[0].id})
    assert len(open_range.results) == 1
