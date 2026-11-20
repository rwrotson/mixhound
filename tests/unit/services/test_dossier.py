from mixhound.services.dossier import MAX_FACTS, describe_recording
from mixhound.services.ports import CatalogPage, CatalogRecording, DossierFact


def test_recording_is_described_with_sources() -> None:
    recording = CatalogRecording(
        artist="Low",
        title="Words",
        confirmed_by=("musicbrainz", "itunes"),
        mbid="m1",
        isrcs=("USX1", "USX2"),
        year=1994,
        tags={"musicbrainz": ("slowcore",) * 10, "lastfm": ()},
        pages=(CatalogPage("musicbrainz", "https://musicbrainz.org/recording/m1"),),
        facts=(
            DossierFact("credit", "drums: Mimi Parker", "musicbrainz", "https://mb/1"),
            DossierFact("text:review", "Hushed and slow", "wikipedia"),
            DossierFact("relation:similar", "Codeine", ""),
        ),
    )

    text = describe_recording(recording)

    assert text.splitlines() == [
        "Recording: Low — Words",
        "Identity: year 1994; MusicBrainz m1; ISRC USX1, USX2",
        "Confirmed by: musicbrainz, itunes",
        "Tags (musicbrainz): " + ", ".join(["slowcore"] * 8),
        "- credit: drums: Mimi Parker [musicbrainz https://mb/1]",
        "- text:review: Hushed and slow [wikipedia]",
        "- relation:similar: Codeine",
        "Page (musicbrainz): https://musicbrainz.org/recording/m1",
    ]


def test_unconfirmed_recording_says_so_and_caps_facts() -> None:
    facts = tuple(DossierFact("credit", f"role {n}", "discogs") for n in range(MAX_FACTS + 5))

    text = describe_recording(CatalogRecording(artist="A", title="B", facts=facts))

    assert "Confirmed by: no catalog (unconfirmed)" in text
    assert "Identity" not in text
    assert text.count("- credit:") == MAX_FACTS
