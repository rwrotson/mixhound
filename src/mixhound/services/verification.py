import logging
from collections.abc import Iterable
from urllib.parse import urlsplit

from mixhound.models.domain import (
    DiscoveryListeningLink,
    DiscoveryResearchResult,
    DiscoverySource,
    DiscoveryTrack,
)
from mixhound.services.cache import DAY, CacheSpace, remember
from mixhound.services.evidence.verdicts import MIN_VERIFIED_SCORE
from mixhound.services.identity import recording_key
from mixhound.services.links import choose_preferred
from mixhound.services.listening import service_name, supported_link
from mixhound.services.pages import GONE_STATUSES, PageInspector
from mixhound.services.ports import (
    Cache,
    CatalogRecording,
    CatalogUnavailableError,
    FetchError,
    TrackCatalog,
)
from mixhound.services.run_state import RunState, alternates, bounded
from mixhound.services.urls import host_matches

log = logging.getLogger(__name__)

MAX_SOURCES_CHECKED = 6
MAX_LINKS_CHECKED = 8
PAGE_MATCHES = CacheSpace("page-match", 14 * DAY)
# Sources checked first: the most specific pages name the recording most reliably.
_SOURCE_ORDER = {"track": 0, "album": 1, "review": 2, "other": 3, "artist": 4}


def likely_result(state: RunState, track: DiscoveryTrack) -> bool:
    """A confirmed candidate with a reasoned similarity high enough to be published."""
    return (
        track.id in state.confirmed
        and bool(track.similarities)
        and (track.similarity_score or 0) >= MIN_VERIFIED_SCORE
    )


def promote_album_sources(result: DiscoveryResearchResult) -> None:
    """Offer a confirmed Bandcamp album page as a listening link; playback picks the track."""
    for track in result.results + result.unverified_leads:
        known = {link.url for link in track.listening_links}
        for source in track.sources:
            host = (urlsplit(source.url).hostname or "").lower()
            if (
                source.target_type == "album"
                and source.reachable is True
                and source.matches_recording is True
                and source.url not in known
                and urlsplit(source.url).scheme == "https"
                and host_matches(host, "bandcamp.com")
                and urlsplit(source.url).path.startswith("/album/")
            ):
                track.listening_links.append(
                    DiscoveryListeningLink(
                        url=source.url,
                        service="bandcamp",
                        target_type="album",
                        reachable=True,
                    ),
                )
                known.add(source.url)


def _add_catalog_recording(
    state: RunState, track: DiscoveryTrack, recording: CatalogRecording
) -> None:
    """Fill identifiers and add the catalog's pages and playable links of a confirmed match."""
    if recording.mbid:
        track.external_recording_id = track.external_recording_id or f"mbid:{recording.mbid}"
    if track.year is None:
        track.year = recording.year
    if recording.isrcs and not track.isrc:
        track.isrc = recording.isrcs[0]
    _add_catalog_pages(state, track, recording)
    _add_catalog_links(state, track, recording)


def _add_catalog_pages(state: RunState, track: DiscoveryTrack, recording: CatalogRecording) -> None:
    for page in recording.pages:
        if all(source.url != page.url for source in track.sources):
            track.sources.append(
                DiscoverySource(
                    url=page.url,
                    source_type=page.source,
                    target_type="track",
                    claim=f"{page.source} catalog: {recording.artist} — {recording.title}",
                    reachable=True,
                    matches_recording=True,
                ),
            )
        state.trust(track, page.url)


def _add_catalog_links(state: RunState, track: DiscoveryTrack, recording: CatalogRecording) -> None:
    """Add the catalog's track pages on services the consumer can play."""
    for link in recording.links:
        service = service_name(link.url)
        if service is None or not supported_link(link.url, service, link.target):
            continue
        if all(known.url != link.url for known in track.listening_links):
            track.listening_links.append(
                DiscoveryListeningLink(
                    url=link.url, service=service, target_type="track", reachable=True
                ),
            )
        state.trust(track, link.url)


class ResultVerifier:
    """Server-side confirmation of results: catalogs, source pages and listening links."""

    def __init__(self, catalog: TrackCatalog, pages: PageInspector | None, cache: Cache) -> None:
        self.catalog = catalog
        self.pages = pages
        self.cache = cache

    # Catalog confirmation -------------------------------------------------------------

    async def confirm_catalog(self, state: RunState, result: DiscoveryResearchResult) -> None:
        """Confirm identities through one catalog request per result; fill year and ISRC."""

        async def confirm(track: DiscoveryTrack) -> None:
            state.check()
            key = "|".join(
                (
                    recording_key(track.artist, track.title, qualifier=track.version_qualifier),
                    track.isrc or "",
                ),
            )
            if key in state.catalog_answers:
                found = state.catalog_answers[key]
            elif state.catalog_down:
                return
            else:
                try:
                    found = await self.catalog.candidate(
                        track.artist, track.title, version=track.version_qualifier, isrc=track.isrc
                    )
                except CatalogUnavailableError as error:
                    log.warning("catalog_unavailable error=%s", error)
                    state.catalog_down = True
                    return
                state.catalog_answers[key] = found
            if found is None:
                return
            state.recordings[track.id] = found
            if found.confirmed:
                _add_catalog_recording(state, track, found)

        await bounded(result.results + result.unverified_leads, confirm)

    # Independent page checks ----------------------------------------------------------

    async def confirm_sources(self, state: RunState, result: DiscoveryResearchResult) -> None:
        """Fetch the research's own sources and keep only what the pages show."""

        async def confirm(track: DiscoveryTrack) -> None:
            if track.id in state.confirmed:
                return  # A catalog already confirmed it; the sources stay as claims.
            pending = sorted(
                track.sources, key=lambda source: _SOURCE_ORDER.get(source.target_type, 5)
            )
            for source in pending[:MAX_SOURCES_CHECKED]:
                state.check()
                matches = await self.page_matches(source.url, track)
                if matches is None:
                    continue
                source.reachable, source.matches_recording = True, matches
                if matches:
                    state.trust(track, source.url)
                    break

        await bounded(result.results + result.unverified_leads, confirm)

    async def page_matches(self, url: str, track: DiscoveryTrack) -> bool | None:
        """Whether the page names this recording; None when it could not be fetched."""
        pages = self.pages
        if pages is None:
            return None
        key = "|".join(
            (
                url,
                track.artist,
                track.title,
                track.version_qualifier or "",
                track.artist_alt or "",
                track.title_alt or "",
            ),
        )

        async def compute() -> bool | None:
            try:
                return await pages.verify_recording(
                    url,
                    track.artist,
                    track.title,
                    track.version_qualifier,
                    alternates=alternates(track),
                )
            except FetchError, OSError, ValueError:
                return None

        return await remember(self.cache, PAGE_MATCHES, key, compute)

    # Listening links ------------------------------------------------------------------

    async def confirm_links(self, state: RunState, result: DiscoveryResearchResult) -> None:
        """Check each result's listening links, then pick its preferred link.

        Playback is not checked here: the consumer of the results does that.
        """

        async def confirm(track: DiscoveryTrack) -> None:
            known = state.trusted_urls.get(track.id, set())
            for link in track.listening_links[:MAX_LINKS_CHECKED]:
                state.check()
                if link.url in known:
                    link.reachable = True if link.reachable is None else link.reachable
                else:
                    await self._check_link(link, track)
                    if link.reachable is True and link.target_type == "track":
                        state.trust(track, link.url)
            choose_preferred(track)

        await bounded(result.results + result.unverified_leads, confirm)

    async def _check_link(self, link: DiscoveryListeningLink, track: DiscoveryTrack) -> None:
        """Mark a vanished page unreachable, a matching one reachable, and others unknown."""
        if link.target_type == "album":
            # An album page lists the track among others; only its existence is checked.
            gone = self.pages is not None and await self.pages.gone(link.url)
            link.reachable = False if gone else link.reachable
            return
        if self.pages is None:
            link.reachable = None
            return
        try:
            matches = await self.pages.verify_recording(
                link.url,
                track.artist,
                track.title,
                track.version_qualifier,
                alternates=alternates(track),
            )
        except FetchError as error:
            matches = False
            if error.status in GONE_STATUSES:
                link.reachable = False
                return
        except OSError, ValueError:
            matches = False
        link.reachable = True if matches else None

    async def verify_proposed(
        self,
        state: RunState,
        proposals: Iterable[tuple[DiscoveryTrack, DiscoveryListeningLink]],
    ) -> None:
        """Add proposed track pages whose content names the recording."""

        async def verify(item: tuple[DiscoveryTrack, DiscoveryListeningLink]) -> None:
            track, link = item
            if await self.page_matches(link.url, track):
                link.reachable = True
                if all(other.url != link.url for other in track.listening_links):
                    track.listening_links.append(link)
                state.trust(track, link.url)

        await bounded(proposals, verify)
