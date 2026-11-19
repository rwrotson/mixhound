import re
import unicodedata
from dataclasses import dataclass

_VERSION = re.compile(
    r"\b(live|demo|remix|remaster(?:ed)?|cover|alternate|alt\.? take|reissue)\b",
    re.IGNORECASE,
)
_SPACE = re.compile(r"\s+")


def normalize(value: str) -> str:
    """Fold case, Unicode forms, punctuation and spacing for comparison."""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(" " if unicodedata.category(char).startswith("P") else char for char in value)
    return _SPACE.sub(" ", value).strip()


_FEATURING = re.compile(r"\b(?:feat|ft|featuring|with)\b.*$")
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")
_NEUTRAL_VERSIONS = {"remaster", "remastered", "reissue"}
_TOPIC_SUFFIX = re.compile(r"\s+-\s+topic$", re.IGNORECASE)


def fold(value: str) -> str:
    """Normalize for fuzzy identity: diacritics, '&', punctuation, case, a leading article."""
    decomposed = unicodedata.normalize("NFKD", value.replace("&", " and "))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    folded = normalize(stripped)
    return folded.removeprefix("the ")


def version_tokens(*values: str | None) -> frozenset[str]:
    """Recording-version markers, ignoring remaster/reissue which keep the same performance."""
    found = {
        "alternate" if match.casefold().startswith("alt") else match.casefold()
        for value in values
        if value
        for match in _VERSION.findall(value)
    }
    return frozenset(token for token in found if token not in _NEUTRAL_VERSIONS)


def core_title(title: str) -> str:
    """Return a title without featured artists, version labels and years."""
    folded = _FEATURING.sub("", fold(title))
    folded = _VERSION.sub(" ", folded)
    folded = _YEAR.sub(" ", folded)
    return _SPACE.sub(" ", folded).strip()


def contains_words(haystack: str, needle: str) -> bool:
    """Whether folded `needle` occurs in folded `haystack` on word boundaries."""
    return bool(needle) and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


_PARENTHETICAL = re.compile(r"^(?P<outer>.+?)\s*[(\[](?P<inner>[^()\[\]]+)[)\]]\s*$")
_SLASH = re.compile(r"\s+/\s+")


def _scripts(value: str) -> frozenset[str]:
    """Writing systems of the letters in `value` (LATIN, CYRILLIC, CJK, ...)."""
    found = set()
    for char in value:
        if char.isalpha():
            name = unicodedata.name(char, "")
            found.add(name.split(" ", 1)[0] if name else "OTHER")
    return frozenset(found)


def _alternate_script(first: str, second: str) -> bool:
    """Whether two spellings are the same name in different writing systems."""
    one, two = _scripts(first), _scripts(second)
    return bool(one) and bool(two) and not (one & two)


def name_variants(value: str) -> list[str]:
    """The name itself plus the halves of a transliteration or translation written beside it.

    "Самцы Дронта (Samtsi Dronta)" and "Дети Госпиталей / Children of Hospitals" also yield
    each half; "Song (Part 2)" or "Song (Live)" stay whole because both parts share a script.
    """
    value = value.strip()
    variants = [value]
    found = _PARENTHETICAL.match(value)
    if (
        found
        and not _VERSION.search(found["inner"])
        and _alternate_script(found["outer"], found["inner"])
    ):
        variants += [found["outer"].strip(), found["inner"].strip()]
    halves = _SLASH.split(value, maxsplit=1)
    if len(halves) > 1 and _alternate_script(*halves):
        variants += [half.strip() for half in halves]
    return list(dict.fromkeys(variant for variant in variants if variant))


def canonical_name(value: str) -> str:
    """The original-script spelling of a name written with its transliteration beside it."""
    return name_variants(value)[1] if len(name_variants(value)) > 1 else value.strip()


@dataclass(frozen=True)
class WantedRecording:
    """The recording a match looks for: artist, title, version and transliterated names."""

    artist: str
    title: str
    version: str | None = None
    alternates: tuple[str | None, str | None] = (None, None)


def _identity_matches(
    found_artist: str | None,
    found_title: str,
    wanted: WantedRecording,
    *,
    exact_title: bool,
) -> bool:
    wanted_artist = fold(_TOPIC_SUFFIX.sub("", wanted.artist))
    wanted_title = core_title(wanted.title)
    if not wanted_artist or not wanted_title:
        return False
    titles = [core_title(variant) for variant in name_variants(found_title)]
    if exact_title:
        title_ok = wanted_title in titles
    else:
        title_ok = any(contains_words(observed, wanted_title) for observed in titles)
    context = fold(_TOPIC_SUFFIX.sub("", found_artist)) if found_artist else fold(found_title)
    artist_ok = contains_words(context, wanted_artist) or (
        found_artist is not None and contains_words(fold(found_title), wanted_artist)
    )
    return (
        title_ok
        and artist_ok
        and version_tokens(found_title) == version_tokens(wanted.title, wanted.version)
    )


def identity_matches(
    found_artist: str | None,
    found_title: str,
    wanted: WantedRecording,
    *,
    exact_title: bool = True,
) -> bool:
    """Compare an observed recording identity with the expected artist, title, and version.

    `found_artist` may be any text expected to name the artist (an uploader, a description);
    when it is None, the artist must appear in `found_title` itself. Each expected name also
    matches through its transliteration (`name_variants` or the explicit `alternates`).
    """
    alt_artist, alt_title = wanted.alternates
    artists = name_variants(wanted.artist) + ([alt_artist] if alt_artist else [])
    titles = name_variants(wanted.title) + ([alt_title] if alt_title else [])
    return any(
        _identity_matches(
            found_artist,
            found_title,
            WantedRecording(artist, title, wanted.version),
            exact_title=exact_title,
        )
        for artist in dict.fromkeys(artists)
        for title in dict.fromkeys(titles)
    )


def parse_manual_references(lines: str) -> list[tuple[str, str]]:
    """Parse `Artist — Track` lines; raise ValueError on a malformed line."""
    parsed: list[tuple[str, str]] = []
    for number, raw in enumerate(lines.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = re.split(r"\s+[—–-]\s+", line, maxsplit=1)  # noqa: RUF001 - em and en dashes separate artist and title
        if len(parts) == 1 or not all(parts):
            raise ValueError(f"Line {number} must be Artist — Track")
        parsed.append((parts[0].strip(), parts[1].strip()))
    return parsed


def recording_key(
    artist: str,
    title: str,
    *,
    qualifier: str | None = None,
    external_recording_id: str | None = None,
) -> str:
    """Return a key that identifies a recording across sources."""
    if external_recording_id:
        return f"external:{external_recording_id.strip().casefold()}"
    versions = sorted(set(_VERSION.findall(f"{title} {qualifier or ''}")))
    return "metadata:" + "|".join(
        (normalize(canonical_name(artist)), normalize(canonical_name(title)), ",".join(versions)),
    )
