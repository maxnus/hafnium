"""Backend helpers and the allowed-API rule."""

import pathlib
import re

import numpy as np
import pytest
from array_api_compat import is_numpy_namespace

import hafnium
from hafnium.backend import (
    active_backend,
    adjoint,
    array_namespace,
    default_namespace,
    generalized_eigh,
    host_namespace,
)
from hafnium.backend.allowed import ALLOWED
from tests.tolerances import EIGEN_TOL

NAMESPACE_CALL = re.compile(
    r"(?:\bxp|\barray_namespace\([^()]*\))"
    r"\.([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
)


def test_method_code_uses_only_the_allowed_array_api() -> None:
    root = pathlib.Path(hafnium.__file__).parent
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "backend" in path.relative_to(root).parts:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            violations.extend(
                f"{path.relative_to(root)}:{number}: xp.{name}"
                for name in NAMESPACE_CALL.findall(line)
                if name not in ALLOWED
            )
    assert not violations, "\n".join(violations)


def test_array_namespace_of_numpy_arrays_is_numpy() -> None:
    xp = array_namespace(np.zeros(2), np.ones(3))
    assert is_numpy_namespace(xp)
    assert xp.asarray([1.0]).dtype == np.float64


def test_default_namespace_is_the_namespace_of_numpy_arrays() -> None:
    assert default_namespace() is array_namespace(np.zeros(1))
    assert host_namespace() is default_namespace()
    assert active_backend().name == "numpy"


def test_the_allowed_subset_exists_on_every_backend_namespace() -> None:
    namespaces = [host_namespace()]
    jax = pytest.importorskip("jax")
    namespaces.append(jax.numpy)
    for xp in namespaces:
        for name in sorted(ALLOWED):
            target = xp
            for part in name.split("."):
                assert hasattr(target, part), f"{xp.__name__} lacks {name}"
                target = getattr(target, part)


def test_namespace_call_pattern_catches_chained_calls() -> None:
    assert NAMESPACE_CALL.findall("xp.linalg.eigh(a)") == ["linalg.eigh"]
    assert NAMESPACE_CALL.findall("array_namespace(a, b).conj(a)") == ["conj"]
    assert NAMESPACE_CALL.findall("array_namespace(*parts).concat(parts)") == ["concat"]


def test_adjoint_conjugates_and_transposes() -> None:
    x = np.array([[1.0 + 2.0j, 3.0], [4.0j, 5.0]])
    assert np.array_equal(adjoint(x), x.conj().T)


@pytest.mark.parametrize("complex_valued", [False, True])
def test_generalized_eigh(complex_valued: bool) -> None:
    rng = np.random.default_rng(0)
    n = 6
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    if complex_valued:
        a = a + 1j * rng.standard_normal((n, n))
    a = (a + a.conj().T) / 2
    b = b @ b.T + n * np.eye(n)
    w, v = generalized_eigh(a, b)
    assert np.allclose(a @ v, b @ v * w[None, :], atol=EIGEN_TOL)
    assert np.allclose(adjoint(v) @ b @ v, np.eye(n), atol=EIGEN_TOL)
    assert np.all(np.diff(w) >= 0)
