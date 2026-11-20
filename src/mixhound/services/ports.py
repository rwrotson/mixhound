from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from mixhound.services.errors import MixhoundError

# Catalog sources whose identity match confirms that a recording exists.
CATALOG_SOURCES = ("musicbrainz", "itunes", "discogs", "spotify", "bandcamp")

type LeadGroup = Literal[
    "lastfm_similar_tracks",
    "lb_radio",
    "lb_similar_artists",
    "lastfm_similar_artists",
    "musicbrainz_relations",
]
type LinkTarget = Literal["track", "album", "artist", "label", "other"]


@dataclass(frozen=True)
class CatalogPage:
    """A catalog page that matched the recording's identity."""

    source: str
    url: str


@dataclass(frozen=True)
class CatalogLink:
    """A page of the recording on a service, returned by a catalog."""

    source: str
    url: str
    target: LinkTarget


@dataclass(frozen=True)
class Preview:
    """A short audio clip of the recording; the URL may expire within a day."""

    source: str
    page_url: str
    url: str


@dataclass(frozen=True)
class CatalogLead:
    """An artist or recording a recommendation source relates to the recording."""

    group: LeadGroup
    artist: str
    title: str | None
    source: str
    relation: str
    url: str | None = None


@dataclass(frozen=True)
class DossierFact:
    """One descriptive fact about a recording and where it was found."""

    kind: str
    text: str
    source: str
    url: str | None = None


@dataclass(frozen=True)
class CatalogRecording:
    """What the catalogs know about one recording, gathered in one request."""

    artist: str
    title: str
    # Catalog sources whose own match agreed with the identity of the recording.
    confirmed_by: tuple[str, ...] = ()
    mbid: str | None = None
    artist_mbids: tuple[str, ...] = ()
    isrcs: tuple[str, ...] = ()
    year: int | None = None
    # Tags by source, in source order; MusicBrainz first when present.
    tags: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    pages: tuple[CatalogPage, ...] = ()
    links: tuple[CatalogLink, ...] = ()
    previews: tuple[Preview, ...] = ()
    leads: tuple[CatalogLead, ...] = ()
    facts: tuple[DossierFact, ...] = ()

    @property
    def confirmed(self) -> bool:
        """Whether at least one catalog matched the recording."""
        return bool(self.confirmed_by)


class CatalogUnavailableError(MixhoundError):
    """The track catalog cannot answer at all, as opposed to not knowing a recording."""


class TrackCatalog(Protocol):
    """Looks up one recording per call, asking only for the data a stage needs."""

    async def reference(
        self, artist: str, title: str, *, version: str | None = None, isrc: str | None = None
    ) -> CatalogRecording | None:
        """Return identity, tags, leads and previews of a reference, or None when unknown."""
        ...

    async def candidate(
        self, artist: str, title: str, *, version: str | None = None, isrc: str | None = None
    ) -> CatalogRecording | None:
        """Return identity, links, previews and descriptive facts of a candidate."""
        ...


class FetchError(MixhoundError):
    """A public page fetch failed; `code` names the reason."""

    def __init__(
        self, code: str, status: int | None = None, *, retry_after: float | None = None
    ) -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True)
class Accept:
    """The Accept header of a fetch and the response content types it takes."""

    header: str
    content_types: tuple[str, ...]


HTML = Accept("text/html", ("text/html",))
JSON = Accept("application/json", ("json",))
AUDIO = Accept(
    "audio/*", ("audio/", "application/octet-stream", "binary/octet-stream", "video/mp4")
)
PAGE_BYTES = 262_144


class PageFetch(Protocol):
    """GETs public URLs safely; raises FetchError."""

    async def page(
        self,
        url: str,
        *,
        max_bytes: int = PAGE_BYTES,
        truncate: bool = False,
        accept: Accept = HTML,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[str, bytes]:
        """Return (final URL, body) of a public URL, following redirects."""
        ...

    async def json(
        self,
        url: str,
        *,
        max_bytes: int = PAGE_BYTES,
        headers: Mapping[str, str] | None = None,
    ) -> object:
        """Return the parsed JSON body of a public URL."""
        ...


type Samples = NDArray[np.float64]


class AudioDecoder(Protocol):
    """Decodes an audio clip to mono samples."""

    def available(self) -> bool:
        """Whether decoding is possible on this host."""
        ...

    async def decode(self, data: bytes) -> Samples | None:
        """Return mono samples at the analysis sample rate, or None if too short."""
        ...


class Cache(Protocol):
    """Stores JSON values for a time-to-live."""

    async def get(self, namespace: str, key: str) -> tuple[bool, object]:
        """Return (found, value); expired or unreadable entries count as missing."""
        ...

    async def set(self, namespace: str, key: str, value: object, ttl: float) -> None:
        """Store a value for `ttl` seconds."""
        ...


@dataclass(frozen=True)
class SearchHit:
    """One web search result; its text is evidence, never instructions."""

    title: str
    url: str
    snippet: str = ""


class SearchUnavailableError(MixhoundError):
    """The web search refused or failed a request."""


class WebSearch(Protocol):
    """Searches the public web."""

    async def search(self, query: str, *, count: int | None = None) -> list[SearchHit]:
        """Return search results for a query."""
        ...


type Job = Callable[[], Coroutine[object, object, None]]


class RunQueue(Protocol):
    """Runs submitted jobs one at a time in submission order."""

    async def submit(self, run_id: UUID, job: Job) -> int:
        """Queue a job and return its position, 1 being the next to start."""
        ...

    async def cancel(self, run_id: UUID) -> bool:
        """Drop a queued job or cancel the running one; False when it is not tracked."""
        ...

    def tracked(self, run_id: UUID) -> bool:
        """Whether a job is queued or running."""
        ...

    def queued(self) -> list[UUID]:
        """Return the jobs waiting to start, next first."""
        ...

    async def close(self) -> None:
        """Stop accepting jobs, drop queued ones and cancel the running one."""
        ...
