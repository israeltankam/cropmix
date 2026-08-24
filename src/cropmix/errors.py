"""Cropmix-specific exceptions."""


class CropmixError(Exception):
    """Base class for Cropmix exceptions."""


class ValidationError(CropmixError, ValueError):
    """Raised when a model object is internally inconsistent."""


class UnsupportedModelError(CropmixError, NotImplementedError):
    """Raised when a scientifically unsupported model combination is requested."""


class EpiPvrError(CropmixError):
    """Raised when the optional EpiPvr bridge fails."""
