class MixhoundError(Exception):
    """Base of expected application failures."""


class NotFoundError(MixhoundError, LookupError):
    """A requested record does not exist."""


class ConflictError(MixhoundError):
    """A request conflicts with the current state of a record."""
