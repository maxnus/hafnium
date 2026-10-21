"""Exception hierarchy (CONVENTIONS.md §6).

Every exception raised by the library derives from :class:`HafniumError`.
"""


class HafniumError(Exception):
    """Base class of all exceptions raised by Hafnium."""


class InputError(HafniumError, ValueError):
    """A user-supplied specification is invalid or inconsistent."""


class StructureError(HafniumError, ValueError):
    """Block structures, rings or array shapes are inconsistent."""


class BackendError(HafniumError):
    """The active array backend cannot perform the requested operation."""
