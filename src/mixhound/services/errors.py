class MixhoundError(Exception):
    """Base of expected application failures."""


class NotFoundError(MixhoundError, LookupError):
    """A requested record does not exist."""


class ConflictError(MixhoundError):
    """A request conflicts with the current state of a record."""


class InvalidInputError(MixhoundError, ValueError):
    """A request carries values the application cannot use."""
