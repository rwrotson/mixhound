from mixhound.models.domain import DiscoveryListeningLink, DiscoveryResearchResult, DiscoveryTrack
from mixhound.models.research import FoundLink, LinkSearchOutput
from mixhound.services.listening import followup_prompt, missing_links, proposed_links

SPOTIFY = "https://open.spotify.com/track/3q7D4kfMcju2alVibiUAcz"


def track(
    artist: str = "The Coral",
    title: str = "In the Morning",
    qualifier: str | None = None,
) -> DiscoveryTrack:
    return DiscoveryTrack(
        artist=artist,
        title=title,
        version_qualifier=qualifier,
        verification="verified",
    )


def found(*links: tuple[str, str, str]) -> LinkSearchOutput:
    return LinkSearchOutput(
        links=[FoundLink(artist=artist, title=title, url=url) for artist, title, url in links],
    )


def test_followup_proposes_only_matching_recordings_on_playable_track_pages() -> None:
    original = DiscoveryResearchResult(results=[track()])
    missing = missing_links(original)
    assert missing == original.results
    assert "In the Morning" in followup_prompt(missing)
    assert not proposed_links(missing, found(("The Coral", "In the Morning (Live)", SPOTIFY)))
    assert not proposed_links(missing, found(("Other Band", "In the Morning", SPOTIFY)))
    assert not proposed_links(
        missing,
        found(
            (
                "The Coral",
                "In the Morning",
                "https://open.spotify.com/album/3q7D4kfMcju2alVibiUAcz",
            ),
        ),
    )
    assert not proposed_links(
        missing,
        found(("The Coral", "In the Morning", "https://example.org/track/1")),
    )
    proposals = proposed_links(missing, found(("The Coral", "In the Morning", SPOTIFY)))
    assert [(target, link.url, link.service) for target, link in proposals] == [
        (original.results[0], SPOTIFY, "Spotify"),
    ]
    assert proposals[0][1].reachable is None  # The coordinator verifies the page itself.


def test_followup_matches_transliterated_names() -> None:
    lead = track(artist="Самцы Дронта", title="Сумерки")
    lead.artist_alt, lead.title_alt = "Samtsi Dronta", "Twilight"
    url = "https://bferecords.bandcamp.com/track/03-twilight"
    proposals = proposed_links([lead], found(("Samtsi Dronta", "Twilight", url)))
    assert [link.url for _, link in proposals] == [url]


def test_unplayable_link_still_needs_a_followup() -> None:
    candidate = track()
    candidate.listening_links = [
        DiscoveryListeningLink(
            url="https://www.youtube.com/watch?v=abcdefghijk",
            service="YouTube",
            target_type="track",
            reachable=True,
            playback_available=False,
        ),
    ]
    assert missing_links(DiscoveryResearchResult(results=[candidate])) == [candidate]


def test_unverified_leads_also_get_a_listening_followup() -> None:
    lead = track(artist="Bryozone", title="Sub Nautica")
    lead.verification = "unverified"
    original = DiscoveryResearchResult(unverified_leads=[lead])
    assert missing_links(original) == [lead]
