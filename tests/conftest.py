"""Fixtures shared by every test module."""

from collections.abc import Iterator

import pytest

from hafnium.backend import Backend, use_backend


@pytest.fixture(params=["numpy", "jax"])
def backend(request: pytest.FixtureRequest) -> Iterator[Backend]:
    """Run the test under each array backend; JAX is skipped when not installed."""
    name: str = request.param
    if name == "jax":
        pytest.importorskip("jax")
    with use_backend(name) as active:
        yield active
