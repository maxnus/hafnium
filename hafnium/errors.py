"""Exception hierarchy (CONVENTIONS.md §6).

Every exception raised by the library derives from :class:`HafniumError`.
"""

from typing import Any


class HafniumError(Exception):
    """Base class of all exceptions raised by Hafnium."""


class InputError(HafniumError, ValueError):
    """A user-supplied specification is invalid or inconsistent."""


class StructureError(HafniumError, ValueError):
    """Block structures, rings or array shapes are inconsistent."""


class BackendError(HafniumError):
    """The active array backend cannot perform the requested operation."""


class BasisSetError(HafniumError, ValueError):
    """A basis set is unavailable, malformed or unsupported."""


class ConvergenceError(HafniumError):
    """An iterative procedure stopped before reaching its tolerances.

    ``last`` holds the final state of the procedure (for the SCF an
    :class:`~hafnium.scf.ScfResult` with ``converged`` false), so the caller
    can inspect it, tighten the options, or restart from it.
    """

    def __init__(self, message: str, last: Any) -> None:
        super().__init__(message)
        self.last = last
