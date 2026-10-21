"""Client for the Basis Set Exchange REST API.

The API returns the same JSON schema as the bundled files; see
``hafnium/basis/data/README.md`` for provenance and citation.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from importlib.metadata import version as _distribution_version
from typing import Any, cast

from hafnium.errors import BasisSetError

BSE_BASIS_URL = "https://www.basissetexchange.org/api/basis/{name}/format/json/"
"""Endpoint for one basis set; ``elements`` is passed as a query parameter."""

USER_AGENT = (
    f"hafnium/{_distribution_version('hafnium')} (+https://github.com/maxnus/hafnium)"
)

Fetcher = Callable[[str], str]
"""Returns the response body for a URL; injectable for tests and offline use."""


def as_object(value: object) -> dict[str, Any] | None:
    """Return a parsed JSON value as a typed mapping, or ``None`` if it is not one."""
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return None


def normalize_name(name: str) -> str:
    """Canonical form of a basis set name as the Basis Set Exchange keys it."""
    return name.strip().lower().replace(" ", "")


def basis_url(name: str, elements: Iterable[int]) -> str:
    """URL of the JSON representation of ``name`` for the given atomic numbers."""
    numbers = ",".join(str(number) for number in sorted(set(elements)))
    quoted = urllib.parse.quote(normalize_name(name), safe="")
    return BSE_BASIS_URL.format(name=quoted) + f"?elements={numbers}"


def fetch_url(url: str, *, timeout: float = 30.0) -> str:
    """Fetch ``url`` with the library's user agent."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise BasisSetError(
                f"the Basis Set Exchange has no entry for {url}"
            ) from error
        raise BasisSetError(
            f"the Basis Set Exchange returned HTTP {error.code}"
        ) from error
    except urllib.error.URLError as error:
        raise BasisSetError(
            f"cannot reach the Basis Set Exchange ({error.reason}); "
            "use a bundled or cached basis set, or check the network"
        ) from error


def fetch_basis(
    name: str, elements: Iterable[int], *, fetch: Fetcher = fetch_url
) -> dict[str, Any]:
    """Fetch the JSON data of a basis set for the given atomic numbers."""
    body = fetch(basis_url(name, elements))
    try:
        data = as_object(json.loads(body))
    except json.JSONDecodeError as error:
        raise BasisSetError("the Basis Set Exchange returned malformed JSON") from error
    if data is None or as_object(data.get("elements")) is None:
        raise BasisSetError("the Basis Set Exchange response is not a basis set object")
    return data
