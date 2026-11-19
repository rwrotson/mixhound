from collections.abc import Awaitable, Callable

from mixhound.infrastructure.cache import MemoryCache
from mixhound.services.cache import CacheCodec, CacheSpace, remember

SPACE = CacheSpace("ns", 10)


async def test_remember_keeps_definitive_answers() -> None:
    cache = MemoryCache()
    calls: list[object] = []

    def compute(value: object) -> Callable[[], Awaitable[object]]:
        async def inner() -> object:
            calls.append(value)
            return value

        return inner

    assert await remember(cache, SPACE, "a", compute({"v": 1})) == {"v": 1}
    assert await remember(cache, SPACE, "a", compute({"v": 2})) == {"v": 1}
    assert await remember(cache, SPACE, "unknown", compute(None)) is None
    assert await remember(cache, SPACE, "unknown", compute(None)) is None
    assert calls == [{"v": 1}, None, None]


async def test_a_codec_encodes_and_a_failed_decode_recomputes() -> None:
    cache = MemoryCache()
    codec = CacheCodec[tuple[int, int]](encode=list, decode=lambda raw: (raw[0], raw[1]))  # type: ignore[index]

    async def pair() -> tuple[int, int]:
        return (1, 2)

    assert await remember(cache, SPACE, "p", pair, codec) == (1, 2)
    assert cache.entries["ns", "p"][1] == [1, 2]
    assert await remember(cache, SPACE, "p", pair, codec) == (1, 2)
    await cache.set("ns", "p", None, 10)
    assert await remember(cache, SPACE, "p", pair, codec) == (1, 2)
