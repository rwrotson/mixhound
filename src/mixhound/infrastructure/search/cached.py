from mixhound.services.cache import DAY, CacheCodec, CacheSpace, remember
from mixhound.services.ports import Cache, SearchHit, WebSearch

SEARCHES = CacheSpace("web-search", DAY)


def _encode(hits: list[SearchHit]) -> object:
    return [[hit.title, hit.url, hit.snippet] for hit in hits]


def _decode(raw: object) -> list[SearchHit]:
    if not isinstance(raw, list):
        raise TypeError("cached search results are not a list")
    return [SearchHit(str(title), str(url), str(snippet)) for title, url, snippet in raw]


CODEC = CacheCodec[list[SearchHit]](keep=bool, encode=_encode, decode=_decode)


class CachedSearch:
    """Reuses a day-old answer to the same query instead of asking upstream again."""

    def __init__(self, search: WebSearch, cache: Cache) -> None:
        self.inner = search
        self.cache = cache

    async def search(self, query: str, *, count: int | None = None) -> list[SearchHit]:
        """Return cached results for the query, searching only on a miss."""
        key = f"{count or 0}:{query.strip().casefold()}"
        return await remember(
            self.cache, SEARCHES, key, lambda: self.inner.search(query, count=count), CODEC
        )
