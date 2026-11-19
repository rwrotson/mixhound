from mixhound.models.domain import DiscoveryRun
from mixhound.services.errors import ConflictError

ACTIVE_STATUSES = frozenset({"queued", "running"})


class RunInputChangedError(ConflictError):
    """A save would change the input snapshot of a stored run."""


def check_run_update(previous: DiscoveryRun | None, run: DiscoveryRun) -> None:
    """Reject a save that changes a stored run's input or creation time."""
    if previous is not None and (
        previous.input != run.input or previous.created_at != run.created_at
    ):
        raise RunInputChangedError("Run input snapshot is immutable")
