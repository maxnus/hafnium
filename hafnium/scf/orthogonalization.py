"""Canonical orthogonalization of the atomic basis, per k-point."""

from __future__ import annotations

import logging

from hafnium.backend import Array, array_namespace, eigh
from hafnium.errors import InputError

logger = logging.getLogger(__name__)

DEFAULT_OVERLAP_THRESHOLD = 1e-10
"""Overlap eigenvalues below this are removed by canonical orthogonalization."""


def canonical_orthogonalizer(
    overlap: Array, threshold: float = DEFAULT_OVERLAP_THRESHOLD
) -> Array:
    """``X[ao, mo] = V s^{-1/2}`` over the eigenpairs of ``S`` with ``s > threshold``.

    ``X† S X`` is the identity on the retained space. For a batch of
    overlaps ``[kpt, ao, ao]`` the result is ``[kpt, ao, mo]`` with the same
    number of retained functions at every k-point, the smallest count any
    k-point allows, so that the blocks of a structured matrix keep one
    shape; the number of removed functions is logged, which reads the count
    on the host.
    """
    xp = array_namespace(overlap)
    eigenvalues, eigenvectors = eigh(overlap)
    n = int(overlap.shape[-1])
    kept = int(xp.min(xp.sum(xp.astype(eigenvalues > threshold, xp.int64), axis=-1)))
    if kept == 0:
        raise InputError(
            f"no eigenvalue of the overlap exceeds the threshold {threshold:g}; "
            "the overlap is not positive definite or the threshold is too large"
        )
    if kept < n:
        logger.info(
            "canonical orthogonalization removed %d of %d basis functions", n - kept, n
        )
    retained = eigenvectors[..., :, n - kept :]
    return retained / xp.sqrt(xp.expand_dims(eigenvalues[..., n - kept :], axis=-2))
