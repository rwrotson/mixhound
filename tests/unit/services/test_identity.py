import pytest

from mixhound.services.identity import (
    WantedRecording,
    canonical_name,
    identity_matches,
    name_variants,
    normalize,
    parse_manual_references,
    recording_key,
)


def test_identity_and_manual_input() -> None:
    assert normalize("ＣＡＦÉ — Song") == normalize("café - song")
    assert parse_manual_references(" A — Song\nB - Other\n") == [("A", "Song"), ("B", "Other")]
    with pytest.raises(ValueError, match="Line 1"):
        parse_manual_references("Song alone")
    assert recording_key("A", "Song (Live)") != recording_key("A", "Song")
    assert recording_key("A", "Song") != recording_key("B", "Song")
    assert recording_key("A", "Song", qualifier="remix") != recording_key("A", "Song")


def test_external_id_wins_over_metadata() -> None:
    assert recording_key("A", "Song", external_recording_id=" MBID:X ") == "external:mbid:x"


def test_transliterations_are_split_but_version_labels_stay() -> None:
    assert name_variants("Самцы Дронта (Samtsi Dronta)") == [
        "Самцы Дронта (Samtsi Dronta)",
        "Самцы Дронта",
        "Samtsi Dronta",
    ]
    assert name_variants("Дети Госпиталей / Children of Hospitals")[1:] == [
        "Дети Госпиталей",
        "Children of Hospitals",
    ]
    assert name_variants("Song (Live)") == ["Song (Live)"]
    assert canonical_name("Самцы Дронта (Samtsi Dronta)") == "Самцы Дронта"


@pytest.mark.parametrize(
    ("artist", "title", "wanted", "expected"),
    [
        (
            "The Coral",
            "In the Morning (Live)",
            WantedRecording("The Coral", "In the Morning", "Live"),
            True,
        ),
        (
            "The Coral",
            "In the Morning (Live)",
            WantedRecording("The Coral", "In the Morning"),
            False,
        ),
        ("Coral - Topic", "In The Morning", WantedRecording("The Coral", "In the Morning"), True),
        ("Someone Else", "In the Morning", WantedRecording("The Coral", "In the Morning"), False),
        (
            "Samtsi Dronta",
            "Song",
            WantedRecording("Самцы Дронта", "Song", alternates=("Samtsi Dronta", None)),
            True,
        ),
    ],
)
def test_identity_matches(
    artist: str | None, title: str, wanted: WantedRecording, expected: bool
) -> None:
    assert identity_matches(artist, title, wanted) is expected


def test_loose_title_match_accepts_surrounding_words() -> None:
    wanted = WantedRecording("Low", "Words")

    assert identity_matches("Low", "Words - 2009 Remaster Edition", wanted, exact_title=False)
    assert not identity_matches("Low", "Words - 2009 Remaster Edition", wanted)
    assert identity_matches(None, "Low - Words", wanted, exact_title=False)
    assert not identity_matches(None, "Low - Words", wanted)
