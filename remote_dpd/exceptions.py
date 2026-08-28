class RemoteDPDError(Exception):
    """Base error for recoverable service and protocol failures."""


class MatProtocolError(RemoteDPDError):
    """The MAT file is missing an expected variable or has invalid data."""


class UnsupportedMatVersion(MatProtocolError):
    """The MAT file requires an optional reader that is not installed."""
