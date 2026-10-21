"""Lookup of basis set data.

Bundled files first, then the on-disk cache, then the Basis Set Exchange.
"""

from __future__ import annotations

import functools
import json
import os
import re
import tempfile
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import Any

import platformdirs

from hafnium.basis.bse import (
    Fetcher,
    as_object,
    fetch_basis,
    fetch_url,
    normalize_name,
)
from hafnium.basis.shell import Shell
from hafnium.errors import BasisSetError

_SPHERICAL_TYPES = {"gto", "gto_spherical"}
_UNSAFE = re.compile(r"[^a-z0-9+_.-]")


def default_cache_dir() -> Path:
    """Directory where fetched basis set data is cached for this user."""
    return Path(platformdirs.user_cache_dir("hafnium")) / "basis"


def storage_key(name: str) -> str:
    """Filesystem-safe form of a basis set name, used for bundled and cached files."""
    key = _UNSAFE.sub("_", normalize_name(name))
    if not key.strip("._"):
        raise BasisSetError(f"invalid basis set name {name!r}")
    return key


def shells_from_element(element: dict[str, Any]) -> tuple[Shell, ...]:
    """Build shells from one element entry of the Basis Set Exchange JSON.

    Combined shells such as SP are split into one shell per angular momentum,
    each keeping its own coefficient row.
    """
    if "ecp_potentials" in element:
        raise BasisSetError("effective core potentials are not supported yet")
    shells: list[Shell] = []
    for entry in element.get("electron_shells", []):
        function_type = entry.get("function_type", "gto")
        if function_type not in _SPHERICAL_TYPES:
            raise BasisSetError(
                f"only spherical Gaussian shells are supported, got {function_type!r}"
            )
        momenta: list[int] = list(entry["angular_momentum"])
        rows: list[list[str]] = list(entry["coefficients"])
        if len(momenta) == 1:
            shells.append(Shell.from_rows(momenta[0], entry["exponents"], rows))
            continue
        if len(rows) != len(momenta):
            raise BasisSetError("a combined shell has one coefficient row per l value")
        shells.extend(
            Shell.from_rows(l, entry["exponents"], [row])
            for l, row in zip(momenta, rows, strict=True)
        )
    return tuple(shells)


class BasisLibrary:
    """Resolves basis set names to shells per element.

    Lookup order: data bundled with the package, the on-disk cache, and
    finally the Basis Set Exchange, whose response is written to the cache
    one element at a time so later molecules reuse it. Each cache entry
    records the basis set version and revision date it came from.

    Parameters
    ----------
    cache_dir
        Cache location; defaults to the user cache directory.
    fetch
        Function returning the body of a URL; replace it in tests.
    offline
        If true, never contact the Basis Set Exchange.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        fetch: Fetcher = fetch_url,
        offline: bool = False,
    ) -> None:
        self.cache_dir = default_cache_dir() if cache_dir is None else Path(cache_dir)
        self._fetch = fetch
        self.offline = offline

    def shells(
        self, name: str, atomic_numbers: Iterable[int]
    ) -> dict[int, tuple[Shell, ...]]:
        """Shells of basis set ``name`` for each requested atomic number."""
        wanted = sorted(set(atomic_numbers))
        key = storage_key(name)
        found: dict[int, dict[str, Any]] = {}
        bundled = _bundled(key)
        for number in wanted:
            element = bundled.get(str(number)) if bundled else None
            if element is None:
                element = self._cached(key, number)
            if element is not None:
                found[number] = element
        missing = [number for number in wanted if number not in found]
        if missing:
            found.update(self._download(key, missing))
        return {number: shells_from_element(found[number]) for number in wanted}

    def _cache_path(self, key: str, number: int) -> Path:
        return self.cache_dir / key / f"{number}.json"

    def _cached(self, key: str, number: int) -> dict[str, Any] | None:
        """Return the cached element entry, or ``None``; a corrupt file is a miss."""
        path = self._cache_path(key, number)
        if not path.is_file():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                entry = as_object(json.load(handle))
        except (OSError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            return None
        element = None if entry is None else as_object(entry.get("element"))
        if element is None:
            path.unlink(missing_ok=True)
        return element

    def _download(self, key: str, numbers: list[int]) -> dict[int, dict[str, Any]]:
        if self.offline:
            raise BasisSetError(
                f"basis set {key!r} for elements {numbers} is neither bundled "
                "nor cached, and the library is offline"
            )
        data = fetch_basis(key, numbers, fetch=self._fetch)
        elements: dict[str, Any] = data["elements"]
        metadata = {
            "name": data.get("name"),
            "version": data.get("version"),
            "revision_date": data.get("revision_date"),
        }
        result: dict[int, dict[str, Any]] = {}
        for number in numbers:
            element = elements.get(str(number))
            if element is None:
                raise BasisSetError(
                    f"basis set {key!r} does not define element {number}"
                )
            result[number] = element
            self._store(key, number, {**metadata, "element": element})
        return result

    def _store(self, key: str, number: int, entry: dict[str, Any]) -> None:
        """Write a cache entry atomically: a temporary file, then a rename."""
        path = self._cache_path(key, number)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(entry, handle)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


@functools.cache
def _bundled(key: str) -> dict[str, Any] | None:
    """Element entries of a bundled basis set, parsed once per process."""
    resource = resources.files("hafnium.basis") / "data" / f"{key}.json"
    if not resource.is_file():
        return None
    data: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    elements: dict[str, Any] = data["elements"]
    return elements


def bundled_basis_sets() -> tuple[str, ...]:
    """Names of the basis sets shipped with the package."""
    directory = resources.files("hafnium.basis") / "data"
    return tuple(
        sorted(
            entry.name.removesuffix(".json")
            for entry in directory.iterdir()
            if entry.name.endswith(".json")
        )
    )
