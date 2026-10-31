"""What the correlated methods share: orbital spaces and integrals in the orbital basis.

A converged mean-field state is split into its occupied and virtual
canonical orbitals per stored block (:class:`OrbitalSpaces`), which define
the axes of the block tensors of ADR-0030; the fitted integrals are
brought to the orbital basis as a three-index block tensor
(:func:`transform_df_tensor` for a molecule, :func:`transform_periodic_fitting`
for a cell over its k-points), and the orbital energy denominators of
perturbation theory and the diagonal Fock operator of the canonical
orbitals are block tensors over the same axes
(:meth:`OrbitalSpaces.denominators`, :meth:`OrbitalSpaces.fock_block`).
:class:`FittedOrbitalTensors` keeps the fitted tensor between every pair
of spaces and assembles the antisymmetrized integrals ``<pq||rs>`` of any
block from it.
"""

from hafnium.correlated.integrals import (
    FittedOrbitalTensors,
    FittedPeriodicTensors,
    auxiliary_axis,
    fitted_periodic_tensors,
    transfer_axis,
    transform_df_tensor,
    transform_fitted_tensors,
    transform_periodic_fitting,
)
from hafnium.correlated.orbitals import OCCUPATION_TOLERANCE, OrbitalSpaces, Space

__all__ = [
    "OCCUPATION_TOLERANCE",
    "FittedOrbitalTensors",
    "FittedPeriodicTensors",
    "OrbitalSpaces",
    "Space",
    "auxiliary_axis",
    "fitted_periodic_tensors",
    "transfer_axis",
    "transform_df_tensor",
    "transform_fitted_tensors",
    "transform_periodic_fitting",
]
