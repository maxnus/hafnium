"""Helpers that rearrange shell tables for tests of the kernels' symmetry paths."""

import numpy as np

from hafnium.basis import BasisLibrary, BasisSet
from hafnium.integrals import ShellTable
from hafnium.molecule import Molecule

LIBRARY = BasisLibrary(offline=True)


def table_for(molecule: Molecule, basis_name: str) -> ShellTable:
    """The shell table of ``molecule`` in the bundled basis set ``basis_name``."""
    basis = BasisSet.for_molecule(molecule, basis_name, library=LIBRARY)
    return ShellTable.from_basis(basis, molecule.positions_array())


def reversed_table(table: ShellTable) -> tuple[ShellTable, np.ndarray]:
    """``table`` with its shells in reverse order, and the function permutation.

    The kernels take their symmetric shortcuts only when two tables are
    equal; a reversed copy holds the same functions in another order and
    so forces the generic path. ``permutation[j]`` is the index in
    ``table`` of function ``j`` of the reversed table.
    """
    assert table.n_contractions is not None
    order = list(range(table.n_shells))[::-1]
    offsets = table.primitive_offsets
    counts = [int(k) for k in table.n_contractions]
    primitives = [int(offsets[i + 1] - offsets[i]) for i in range(table.n_shells)]
    coefficient_starts = np.concatenate(
        [[0], np.cumsum(np.array(counts) * np.array(primitives))]
    )
    exponents: list[float] = []
    coefficients: list[float] = []
    new_offsets = [0]
    for i in order:
        exponents.extend(table.exponents[offsets[i] : offsets[i + 1]])
        coefficients.extend(
            table.coefficients[coefficient_starts[i] : coefficient_starts[i + 1]]
        )
        new_offsets.append(len(exponents))
    sizes = [
        k * (2 * int(l) + 1) for k, l in zip(counts, table.angular_momenta, strict=True)
    ]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    permutation = np.array(
        [j for i in order for j in range(starts[i], starts[i] + sizes[i])]
    )
    reordered = ShellTable(
        np.asarray(table.angular_momenta)[order],
        np.asarray(table.centers)[order],
        np.array(new_offsets),
        np.array(exponents),
        np.array(coefficients),
        np.array(counts)[order],
    )
    return reordered, permutation


def split_table(table: ShellTable) -> ShellTable:
    """``table`` with every contraction as its own entry, same function order."""
    assert table.n_contractions is not None
    offsets = table.primitive_offsets
    angular_momenta: list[int] = []
    centers: list[np.ndarray] = []
    new_offsets = [0]
    exponents: list[float] = []
    coefficients: list[float] = []
    position = 0
    for i in range(table.n_shells):
        shell_exponents = table.exponents[offsets[i] : offsets[i + 1]]
        for _ in range(int(table.n_contractions[i])):
            angular_momenta.append(int(table.angular_momenta[i]))
            centers.append(np.asarray(table.centers[i]))
            exponents.extend(shell_exponents)
            coefficients.extend(
                table.coefficients[position : position + len(shell_exponents)]
            )
            position += len(shell_exponents)
            new_offsets.append(len(exponents))
    return ShellTable(
        np.array(angular_momenta),
        np.array(centers),
        np.array(new_offsets),
        np.array(exponents),
        np.array(coefficients),
    )
