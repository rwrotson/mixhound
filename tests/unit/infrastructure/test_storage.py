import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pytest

from mixhound.infrastructure.storage.files import (
    FileDiagnostics,
    FileRuns,
    FileSessions,
    FileStore,
    StoredDataError,
)
from mixhound.infrastructure.storage.memory import MemoryDiagnostics, MemoryRuns, MemorySessions
from mixhound.infrastructure.storage.records import RunInputChangedError
from mixhound.memory.repository import RunRepository, SessionRepository
from mixhound.models.domain import (
    DiscoveryRun,
    DiscoveryRunInput,
    DiscoverySession,
    DiscoverySettings,
)

type Factory[T] = Callable[[Path], T]

SESSIONS: list[Factory[SessionRepository]] = [
    lambda _: MemorySessions(),
    lambda root: FileSessions(FileStore(root)),
]
RUNS: list[Factory[RunRepository]] = [
    lambda _: MemoryRuns(),
    lambda root: FileRuns(FileStore(root)),
]


@pytest.mark.parametrize("factory", SESSIONS, ids=["memory", "files"])
async def test_sessions_keep_order_settings_and_updates(
    factory: Factory[SessionRepository], tmp_path: Path
) -> None:
    repository = factory(tmp_path)
    first, second = DiscoverySession(name="one"), DiscoverySession(name="two")

    assert await repository.list() == []
    assert await repository.default_settings() == DiscoverySettings()
    await repository.save(first)
    await repository.save(second)
    first.name = "renamed"
    await repository.save(first)
    await repository.save_default_settings(DiscoverySettings(depth="deep"))

    assert [item.name for item in await repository.list()] == ["renamed", "two"]
    assert await repository.get(second.id) == second
    assert await repository.get(uuid4()) is None
    assert (await repository.default_settings()).depth == "deep"


@pytest.mark.parametrize("factory", RUNS, ids=["memory", "files"])
async def test_runs_keep_their_input_and_list_unfinished_ones(
    factory: Factory[RunRepository], tmp_path: Path
) -> None:
    repository = factory(tmp_path)
    session = uuid4()
    queued = DiscoveryRun(session_id=session, input=DiscoveryRunInput())
    done = DiscoveryRun(session_id=session, input=DiscoveryRunInput(), status="completed")

    await repository.save(queued)
    await repository.save(done)
    queued.status = "running"
    await repository.save(queued)

    assert await repository.get(queued.id) == queued
    assert await repository.get(uuid4()) is None
    assert [run.id for run in await repository.unfinished()] == [queued.id]
    queued.input = DiscoveryRunInput(message="changed")
    with pytest.raises(RunInputChangedError, match="immutable"):
        await repository.save(queued)


async def test_memory_repositories_return_copies() -> None:
    sessions = MemorySessions()
    session = DiscoverySession(name="one")
    await sessions.save(session)

    (await sessions.list())[0].name = "changed"

    assert (await sessions.get(session.id)) == session


async def test_files_are_private_and_survive_a_new_store(tmp_path: Path) -> None:
    session = DiscoverySession(name="kept")
    await FileSessions(FileStore(tmp_path)).save(session)

    assert await FileSessions(FileStore(tmp_path)).get(session.id) == session
    assert (tmp_path / "sessions" / f"{session.id}.json").stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.rglob("*.tmp")) == []


async def test_unreadable_files_are_reported(tmp_path: Path) -> None:
    run = DiscoveryRun(session_id=uuid4(), input=DiscoveryRunInput())
    runs = FileRuns(FileStore(tmp_path))
    await runs.save(run)
    (tmp_path / "runs" / f"{run.id}.json").write_text("{")

    with pytest.raises(StoredDataError, match="Unreadable"):
        await runs.get(run.id)


async def test_a_session_file_missing_from_disk_is_skipped(tmp_path: Path) -> None:
    sessions = FileSessions(FileStore(tmp_path))
    session = DiscoverySession(name="gone")
    await sessions.save(session)
    (tmp_path / "sessions" / f"{session.id}.json").unlink()

    assert await sessions.list() == []


async def test_diagnostics_are_written_and_old_ones_pruned(tmp_path: Path) -> None:
    now = [100 * 86_400.0]
    diagnostics = FileDiagnostics(FileStore(tmp_path), clock=lambda: now[0])
    old, new = uuid4(), uuid4()

    await diagnostics.write(old, {"run": "old"})
    os.utime(tmp_path / "diagnostics" / f"{old}.json", (0, 0))
    await diagnostics.write(new, {"run": "new", "id": new})

    assert [path.stem for path in (tmp_path / "diagnostics").glob("*.json")] == [str(new)]

    memory = MemoryDiagnostics()
    await memory.write(new, {"a": 1})
    assert memory.payloads == {new: {"a": 1}}
