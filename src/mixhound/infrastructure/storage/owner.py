import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def claim_owner(path: Path) -> Iterator[bool]:
    """Hold the exclusive lock of the stored runs while inside; False when another process has it.

    A second process (e.g. one that will fail to bind the port) starts before it fails; without
    the lock it would mark the live process's runs interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True
