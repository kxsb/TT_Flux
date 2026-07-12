class UnknownVideoError(LookupError):
    """The requested video does not exist in the local library."""


class UnknownRunError(LookupError):
    """The requested run does not exist."""


class InvalidRunStateError(RuntimeError):
    """The run cannot be operated from its current state."""


class InvalidClipRangeError(ValueError):
    """The requested clip range is invalid."""


__all__ = [
    "InvalidClipRangeError",
    "InvalidRunStateError",
    "UnknownRunError",
    "UnknownVideoError",
]
