"""The fitted integrals in the orbital basis, and the orbital energy denominators."""

from __future__ import annotations

from hafnium.backend import Array, adjoint, array_namespace
from hafnium.correlated.orbitals import OrbitalSpaces, Space
from hafnium.errors import InputError, StructureError
from hafnium.structured import (
    BlockStructure,
    BlockTensor,
    IrrepSectors,
    Spin,
    TensorAxis,
    TensorStructure,
)
from hafnium.structured.tensor import Key
from hafnium.structured.validation import dtype_for


def auxiliary_axis(n_aux: int, sectors: IrrepSectors | None = None) -> TensorAxis:
    """Build the tensor axis of the auxiliary functions, unconjugated.

    With ``sectors``, the irrep sectors of the auxiliary basis, the axis
    runs over the symmetry-adapted auxiliary functions sector by sector;
    without, over the ``n_aux`` functions as one sector. The axis is a
    spatial, real structure: an auxiliary function carries no spin.
    """
    if sectors is None:
        return TensorAxis(BlockStructure.spatial(), (n_aux,))
    if sectors.n_functions != n_aux:
        raise InputError(
            f"the auxiliary sectors cover {sectors.n_functions} functions, the "
            f"fitted tensor {n_aux}"
        )
    return TensorAxis(BlockStructure.spatial().with_irreps(sectors), sectors.dimensions)


def transform_df_tensor(
    df_tensor: Array,
    spaces: OrbitalSpaces,
    bra: Space,
    ket: Space,
    *,
    auxiliary: IrrepSectors | None = None,
) -> BlockTensor:
    """Bring the fitted tensor to the orbital basis, ``B[P, p, q]`` block by block.

    ``B[P, p, q] = Σ_uv conj(C[u, p]) B[P, u, v] C[v, q]`` with ``p`` in the
    ``bra`` space, conjugated, and ``q`` in the ``ket`` space; a spinor
    block sums its alpha and beta components, between which the integrals
    do not couple. With irrep sectors on the state the auxiliary basis is
    adapted to the same group, ``auxiliary`` being its sectors, so that
    the auxiliary index carries an irrep as well and the blocks the
    characters forbid are left out (ADR-0030).

    Parameters
    ----------
    df_tensor : Array[aux, ao, ao]
        The fitted tensor with the metric absorbed,
        :meth:`hafnium.integrals.DensityFitting.df_tensor`.
    """
    structure = spaces.structure
    if (structure.irreps is None) != (auxiliary is None):
        raise InputError(
            "a state over irrep sectors needs the auxiliary basis adapted to the "
            "same group, and a state without sectors an unadapted one"
        )
    if df_tensor.ndim != 3 or df_tensor.shape[1] != df_tensor.shape[2]:
        raise InputError("the fitted tensor has shape [aux, ao, ao]")
    if int(df_tensor.shape[1]) != spaces.n_ao:
        raise InputError(
            f"the fitted tensor covers {int(df_tensor.shape[1])} atomic orbitals, "
            f"the orbitals {spaces.n_ao}"
        )
    xp = array_namespace(df_tensor)
    axes = (
        auxiliary_axis(int(df_tensor.shape[0]), auxiliary),
        spaces.axis(bra, conjugated=True),
        spaces.axis(ket, conjugated=False),
    )
    tensor_structure = TensorStructure(axes, ring=spaces.ring)
    # The real fitted tensor is not copied: a complex orbital block promotes
    # the products, and from_blocks casts real blocks of a complex ring.
    adapted = df_tensor
    if auxiliary is not None:
        salcs = xp.concat(auxiliary.coefficients, axis=1)
        adapted = xp.tensordot(xp.linalg.matrix_transpose(salcs), adapted, axes=1)
    base = structure.sector_structure
    bra_dense = spaces.coefficients(bra).materialized(orbitals=True)
    ket_dense = spaces.coefficients(ket).materialized(orbitals=True)
    dense: dict[int, Array] = {
        index: _transform(adapted, bra_dense[index], ket_dense[index], block.spin)
        for index, block in enumerate(base.blocks)
        if block.is_stored
    }
    width = len(base.blocks)
    offsets = (
        _spin_offsets(axes[0], 1),
        _spin_offsets(axes[1], width),
        _spin_offsets(axes[2], width),
    )
    blocks: dict[Key, Array] = {}
    for key in tensor_structure.stored_keys:
        p_aux, p_bra, p_ket = key
        if p_bra % width != p_ket % width or p_bra % width not in dense:
            raise StructureError(
                "a stored block of the fitted tensor couples spin blocks that the "
                "orbital transformation does not provide"
            )
        blocks[key] = dense[p_bra % width][
            offsets[0][p_aux],
            offsets[1][p_bra],
            offsets[2][p_ket],
        ]
    return BlockTensor.from_blocks(tensor_structure, blocks, xp=xp)


def _spin_offsets(axis: TensorAxis, width: int) -> tuple[slice, ...]:
    """Columns of every sector among the orbitals of its spin block.

    The blocks of a structure repeat a spin pattern of ``width`` blocks per
    irrep sector, and the orbitals of a spin block run over its sectors in
    order; with ``width`` one the slices are those of the dense axis.
    """
    starts = [0] * width
    slices: list[slice] = []
    for sector, size in enumerate(axis.sizes):
        start = starts[sector % width]
        slices.append(slice(start, start + size))
        starts[sector % width] = start + size
    return tuple(slices)


def _transform(df_tensor: Array, bra: Array, ket: Array, spin: Spin) -> Array:
    """``Σ_uv conj(C[u, p]) B[P, u, v] C[v, q]``, summed over the spins of a spinor."""
    if spin is Spin.SPINOR:
        n = int(df_tensor.shape[1])
        return _transform(df_tensor, bra[:n], ket[:n], Spin.NONE) + _transform(
            df_tensor, bra[n:], ket[n:], Spin.NONE
        )
    xp = array_namespace(df_tensor, bra, ket)
    half = df_tensor @ ket
    return xp.expand_dims(adjoint(bra), axis=0) @ half


def energy_denominators(
    spaces: OrbitalSpaces, structure: TensorStructure
) -> BlockTensor:
    """``Σ_occupied ε − Σ_virtual ε`` over the axes of ``structure``.

    Every axis runs over the occupied or the virtual orbitals of
    ``spaces``; the orbital energies of occupied indices enter with a
    plus, those of virtual indices with a minus, so that for the axes
    ``[i, j, a, b]`` the result is ``ε_i + ε_j − ε_a − ε_b``.
    """
    spaces_of_axes = [spaces.space_of(axis) for axis in structure.axes]
    signs = [1.0 if space is Space.OCCUPIED else -1.0 for space in spaces_of_axes]
    xp = array_namespace(*spaces.occupied_energies.blocks)
    dtype = dtype_for(xp, structure.ring)

    def build(key: Key) -> Array:
        total: Array | None = None
        for position, (space, sign, sector) in enumerate(
            zip(spaces_of_axes, signs, key, strict=True)
        ):
            shape = [1] * len(key)
            shape[position] = -1
            term = sign * xp.reshape(spaces.sector_energies(space, sector), shape)
            total = term if total is None else total + term
        assert total is not None
        return xp.astype(xp.broadcast_to(total, structure.shape(key)), dtype)

    return BlockTensor.from_function(structure, build)
