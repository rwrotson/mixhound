import json
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlencode, urlsplit

from mixhound.services.identity import (
    WantedRecording,
    contains_words,
    fold,
    identity_matches,
    name_variants,
    version_tokens,
)
from mixhound.services.ports import FetchError, PageFetch
from mixhound.services.urls import host_matches, service_for

log = logging.getLogger(__name__)

VERIFY_PAGE_BYTES = 1_048_576
NEARBY_CHARACTERS = 300
GONE_STATUSES = frozenset({404, 410})
OEMBED_ENDPOINTS = {
    "youtube.com": "https://www.youtube.com/oembed",
    "youtu.be": "https://www.youtube.com/oembed",
    "soundcloud.com": "https://soundcloud.com/oembed",
    "spotify.com": "https://open.spotify.com/oembed",
}

type Candidate = tuple[str | None, str]


@dataclass(frozen=True)
class PageRecording:
    """The recording a public page describes, as far as its metadata says."""

    url: str
    service: str
    artist: str | None = None
    title: str | None = None


def _json_ld_types(item: dict[str, object]) -> set[str]:
    """A JSON-LD item's "@type", which may be one string or a list of them."""
    kinds = item.get("@type")
    if isinstance(kinds, str):
        return {kinds}
    return {kind for kind in kinds if isinstance(kind, str)} if isinstance(kinds, list) else set()


class _MetadataParser(HTMLParser):
    """Collect meta tags, JSON-LD, the page title, and visible text."""

    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.json_ld: list[str] = []
        self.page_title = ""
        self.text: list[str] = []
        self._json_ld = False
        self._title = False
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "meta":
            key = values.get("property") or values.get("name")
            if key and values.get("content"):
                self.meta.setdefault(key.lower(), values["content"] or "")
        elif tag == "script":
            self._hidden += 1
            self._json_ld = values.get("type") == "application/ld+json"
        elif tag == "style":
            self._hidden += 1
        elif tag == "title":
            self._title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._hidden:
            self._hidden -= 1
            self._json_ld = False
        elif tag == "title":
            self._title = False

    def handle_data(self, data: str) -> None:
        if self._json_ld:
            self.json_ld.append(data)
        if self._title:
            self.page_title += data
        if not self._hidden:
            self.text.append(data)

    def recordings(self) -> list[Candidate]:
        """(artist, title) pairs declared by JSON-LD MusicRecording items."""
        found: list[Candidate] = []
        for script in self.json_ld:
            try:
                _visit_recordings(json.loads(script), found)
            except ValueError:
                continue
        return found


def _visit_recordings(item: object, found: list[Candidate]) -> None:
    """Collect (artist, title) pairs of recordings in a JSON-LD item and its children."""
    if isinstance(item, list):
        for child in item:
            _visit_recordings(child, found)
        return
    if not isinstance(item, dict):
        return
    name = item.get("name")
    if _json_ld_types(item) & {"MusicRecording", "MusicComposition"} and isinstance(name, str):
        by = item.get("byArtist")
        artists = by if isinstance(by, list) else [by]
        names = [
            entry["name"]
            for entry in artists
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        ]
        found.append((" ".join(names) or None, name))
    for key in ("@graph", "track", "itemListElement", "item"):
        _visit_recordings(item.get(key), found)


def _parse(content: bytes) -> _MetadataParser:
    parser = _MetadataParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    return parser


class PageInspector:
    """Read what public pages say about recordings, independently of the researcher."""

    def __init__(self, fetch: PageFetch) -> None:
        self.fetch = fetch

    async def gone(self, url: str) -> bool:
        """Whether the page no longer exists, as opposed to being merely unverifiable."""
        try:
            await self.fetch.page(url, truncate=True)
        except FetchError as error:
            return error.status in GONE_STATUSES
        except OSError, ValueError:
            return False
        return False

    async def resolve(self, url: str) -> PageRecording:
        """Fetch a public page and read the recording it describes; raises FetchError."""
        final_url, content = await self.fetch.page(url, max_bytes=VERIFY_PAGE_BYTES, truncate=True)
        service = service_for(final_url)
        parser = _parse(content)
        title = (
            parser.meta.get("music:song")
            or parser.meta.get("og:title")
            or parser.page_title
            or None
        )
        artist = parser.meta.get("music:musician")
        if artist and artist.startswith("http"):
            artist = None
        recordings = parser.recordings()
        if recordings:
            artist, title = recordings[0][0] or artist, recordings[0][1]
        if service == "Bandcamp" and title and not artist and " by " in title:
            title, artist = title.rsplit(" by ", 1)
        return PageRecording(url=final_url, service=service, artist=artist, title=title)

    async def _oembed_identity(self, url: str) -> list[Candidate]:
        host = (urlsplit(url).hostname or "").lower()
        endpoint = next(
            (value for domain, value in OEMBED_ENDPOINTS.items() if host_matches(host, domain)),
            None,
        )
        if endpoint is None:
            return []
        data = await self.fetch.json(f"{endpoint}?{urlencode({'url': url, 'format': 'json'})}")
        if not isinstance(data, dict) or not isinstance(data.get("title"), str):
            return []
        author = data.get("author_name") if isinstance(data.get("author_name"), str) else None
        return [(author, data["title"])]

    async def identity(self, url: str) -> tuple[list[Candidate], str]:
        """Return structured (artist context, title) candidates and the visible text of a page.

        oEmbed is consulted first for services that render pages client-side; the page itself
        supplies Open Graph, JSON-LD, and title metadata.
        """
        candidates: list[Candidate] = []
        try:
            candidates += await self._oembed_identity(url)
        except FetchError as error:
            log.info("oembed_lookup_failed url=%s code=%s", url, error.code)
        try:
            _, content = await self.fetch.page(url, max_bytes=VERIFY_PAGE_BYTES, truncate=True)
        except FetchError:
            if candidates:
                return candidates, ""
            raise
        parser = _parse(content)
        candidates += parser.recordings()
        meta_context = " ".join(
            value
            for key, value in parser.meta.items()
            if key
            in {
                "og:description",
                "description",
                "music:musician",
                "og:site_name",
                "twitter:audio:artist_name",
            }
            and not value.startswith("http")
        )
        for key in ("music:song", "og:title", "twitter:title"):
            if parser.meta.get(key):
                candidates.append((meta_context or None, parser.meta[key]))
        if parser.page_title.strip():
            candidates.append((meta_context or None, parser.page_title.strip()))
        return candidates, " ".join(parser.text)

    async def verify_recording(
        self,
        url: str,
        artist: str,
        title: str,
        version: str | None = None,
        *,
        alternates: tuple[str | None, str | None] = (None, None),
    ) -> bool:
        """Check public page content independently of the researcher's assertions."""
        candidates, text = await self.identity(url)
        if any(
            identity_matches(
                found_artist,
                found_title,
                WantedRecording(artist, title, version, alternates),
                exact_title=False,
            )
            for found_artist, found_title in candidates
        ):
            return True
        page = fold(re.sub(r"\s+", " ", text))
        wanted_versions = version_tokens(title, version)
        artists = name_variants(artist) + ([alternates[0]] if alternates[0] else [])
        titles = name_variants(title) + ([alternates[1]] if alternates[1] else [])
        for artist_name in dict.fromkeys(artists):
            for title_name in dict.fromkeys(titles):
                artist_key, title_key = fold(artist_name), fold(title_name)
                if not artist_key or not title_key:
                    continue
                for match in re.finditer(rf"(?<!\w){re.escape(title_key)}(?!\w)", page):
                    nearby = page[
                        max(0, match.start() - NEARBY_CHARACTERS) : match.end() + NEARBY_CHARACTERS
                    ]
                    if contains_words(nearby, artist_key) and wanted_versions <= version_tokens(
                        nearby,
                    ):
                        return True
        return False
