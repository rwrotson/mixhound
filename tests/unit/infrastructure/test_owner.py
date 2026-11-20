from pathlib import Path

from mixhound.infrastructure.storage.owner import claim_owner


def test_only_one_holder_owns_the_runs(tmp_path: Path) -> None:
    lock = tmp_path / "store" / "owner.lock"

    with claim_owner(lock) as first, claim_owner(lock) as second:
        assert (first, second) == (True, False)
    with claim_owner(lock) as again:
        assert again
