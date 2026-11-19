from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from mixhound.services.ports import Cache

DAY = 86_400


def _is_known(value: object) -> bool:
    return value is not None


def _identity(value: object) -> object:
    return value


@dataclass(frozen=True)
class CacheCodec[V]:
    """Whether a computed value is cached, and its JSON form in both directions."""

    keep: Callable[[V], bool] = _is_known
    encode: Callable[[V], object] = _identity
    decode: Callable[[object], V] = _identity  # type: ignore[assignment]  # V is the stored type.


@dataclass(frozen=True)
class CacheSpace:
    """A cache namespace and how long its entries stay fresh."""

    name: str
    ttl: float


async def remember[T](
    cache: Cache,
    space: CacheSpace,
    key: str,
    compute: Callable[[], Awaitable[T]],
    codec: CacheCodec[T] | None = None,
) -> T:
    """Return the cached value or compute, store (when the codec keeps it) and return it."""
    codec = codec or CacheCodec[T]()
    found, raw = await cache.get(space.name, key)
    if found:
        try:
            return codec.decode(raw)
        except TypeError, ValueError, KeyError:
            pass
    value = await compute()
    if codec.keep(value):
        await cache.set(space.name, key, codec.encode(value), space.ttl)
    return value
