"""Helpers that rearrange shell tables for tests of the kernels' symmetry paths."""

import numpy as np

from hafnium.integrals import ShellTable


def reversed_table(table: ShellTable) -> tuple[ShellTable, np.ndarray]:
    """``table`` with its shells in reverse order, and the function permutation.

    The kernels take their symmetric shortcuts only when two tables are
    equal; a reversed copy holds the same functions in another order and
    so forces the generic path. ``permutation[j]`` is the index in
    ``table`` of function ``j`` of the reversed table.
    """
    order = list(range(table.n_shells))[::-1]
    offsets = table.primitive_offsets
    exponents: list[float] = []
    coefficients: list[float] = []
    new_offsets = [0]
    for i in order:
        exponents.extend(table.exponents[offsets[i] : offsets[i + 1]])
        coefficients.extend(table.coefficients[offsets[i] : offsets[i + 1]])
        new_offsets.append(len(exponents))
    sizes = [2 * int(l) + 1 for l in table.angular_momenta]
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
    )
    return reordered, permutation
