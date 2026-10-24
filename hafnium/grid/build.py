"""One entry point for the grid of a molecule or a cell."""

from __future__ import annotations

from hafnium.grid.molecular import GridSpec, MolecularGrid, build_molecular_grid
from hafnium.grid.periodic import build_cell_grid
from hafnium.molecule import Cell, Molecule


def build_grid(system: Molecule | Cell, spec: GridSpec | None = None) -> MolecularGrid:
    """Build the grid of ``system`` with the sizes of ``spec``.

    :func:`~hafnium.grid.build_molecular_grid` for a molecule,
    :func:`~hafnium.grid.build_cell_grid` for a cell.
    """
    if isinstance(system, Cell):
        return build_cell_grid(system, spec)
    return build_molecular_grid(system, spec)
