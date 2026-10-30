"""Named errors used by HorizonKit.

The library prefers explicit, typed failures over silently producing a
number from data that cannot support it.
"""


class HorizonKitError(ValueError):
    """Base class for all HorizonKit input and estimation errors."""


class InsufficientDataError(HorizonKitError):
    """Raised when a requested estimand is not supported by the design."""


class NotIdentifiedError(HorizonKitError):
    """Raised by strict callers when a law is not identified by the data."""


class InvalidRunLogError(HorizonKitError):
    """Raised when a run log is malformed or has impossible values."""
