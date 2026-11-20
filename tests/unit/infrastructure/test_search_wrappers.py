import pytest

from mixhound.infrastructure.cache import MemoryCache
from mixhound.infrastructure.search.cached import SEARCHES, CachedSearch
from mixhound.infrastructure.search.fallback import FallbackSearch
from mixhound.services.ports import SearchHit, SearchUnavailableError

HIT = SearchHit("Archangel", "https://example.org/a", "x")


class Engine:
    def __init__(self, hits: list[SearchHit] | None = None, error: str | None = None) -> None:
        self.hits = hits or []
        self.error = error
        self.queries: list[tuple[str, int | None]] = []

    async def search(self, query: str, *, count: int | None = None) -> list[SearchHit]:
        self.queries.append((query, count))
        if self.error is not None:
            raise SearchUnavailableError(self.error)
        return self.hits


async def test_fallback_is_not_asked_when_the_primary_answers() -> None:
    primary, secondary = Engine([HIT]), Engine()

    assert await FallbackSearch(primary, secondary).search("q", count=3) == [HIT]
    assert primary.queries == [("q", 3)]
    assert secondary.queries == []


async def test_fallback_answers_when_the_primary_fails() -> None:
    secondary = Engine([HIT])

    assert await FallbackSearch(Engine(error="quota"), secondary).search("q") == [HIT]
    assert secondary.queries == [("q", None)]


async def test_both_failures_are_reported() -> None:
    search = FallbackSearch(Engine(error="quota"), Engine(error="down"))

    with pytest.raises(SearchUnavailableError, match="quota; fallback: down"):
        await search.search("q")


async def test_cached_search_asks_upstream_once_per_query() -> None:
    engine = Engine([HIT])
    search = CachedSearch(engine, MemoryCache())

    assert await search.search("Burial ", count=2) == [HIT]
    assert await search.search("burial", count=2) == [HIT]
    assert await search.search("burial") == [HIT]
    assert engine.queries == [("Burial ", 2), ("burial", None)]


async def test_empty_results_are_not_cached() -> None:
    engine = Engine()
    search = CachedSearch(engine, MemoryCache())

    await search.search("q")
    await search.search("q")

    assert len(engine.queries) == 2


async def test_an_unreadable_cache_entry_is_searched_again() -> None:
    cache = MemoryCache()
    await cache.set(SEARCHES.name, "0:q", {"not": "a list"}, 60)
    engine = Engine([HIT])

    assert await CachedSearch(engine, cache).search("q") == [HIT]
    assert engine.queries == [("q", None)]
