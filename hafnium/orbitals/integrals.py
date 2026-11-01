"""The fitted integrals in the orbital basis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from hafnium.backend import Array, adjoint, array_namespace
from hafnium.basis import BasisSet
from hafnium.errors import InputError, StructureError
from hafnium.integrals import DensityFitting, IntegralEngine, density_fitting_k
from hafnium.integrals.density_fitting import DEFAULT_METRIC_THRESHOLD
from hafnium.integrals.periodic.fitting import PeriodicDensityFitting
from hafnium.molecule import Cell, Molecule
from hafnium.orbitals.spaces import OrbitalSpaces, Space
from hafnium.scf import ScfResult
from hafnium.structured import (
    BlockStructure,
    BlockTensor,
    IrrepSectors,
    Ring,
    Spin,
    TensorAxis,
    TensorStructure,
    contract,
)
from hafnium.structured.tensor import Key
from hafnium.symmetry import KPoints, PointGroup, SymmetryAdaptation


def auxiliary_axis(n_aux: int, sectors: IrrepSectors | None = None) -> TensorAxis:
    """Build the tensor axis of the auxiliary functions of a molecule, conjugated.

    With ``sectors``, the irrep sectors of the auxiliary basis, the axis
    runs over the symmetry-adapted auxiliary functions sector by sector;
    without, over the ``n_aux`` functions as one sector. The axis is a
    spatial, real structure: an auxiliary function carries no spin. It
    enters conjugated, as the auxiliary function of ``(P|uv)`` does.
    """
    if sectors is None:
        return TensorAxis(BlockStructure.spatial(), (n_aux,), conjugated=True)
    if sectors.n_functions != n_aux:
        raise InputError(
            f"the auxiliary sectors cover {sectors.n_functions} functions, the "
            f"fitted tensor {n_aux}"
        )
    return TensorAxis(
        BlockStructure.spatial().with_irreps(sectors),
        sectors.dimensions,
        conjugated=True,
    )


def transfer_axis(fitting: PeriodicDensityFitting) -> TensorAxis:
    """Build the tensor axis of the auxiliary functions of a cell, conjugated.

    The sectors are the momentum transfers of the mesh, each holding the
    auxiliary functions in the eigenbasis of the metric at that transfer
    (ADR-0019). The axis is a spatial complex structure over the transfer
    mesh: the eigenvectors at ``q`` and ``−q`` carry independent phases,
    so no block derives from another along it. The auxiliary function of
    ``(P^q | u k1 v k2)`` enters conjugated and carries the transfer
    ``q = k2 − k1`` of the pair.
    """
    structure = BlockStructure.spatial(Ring.COMPLEX).with_k_axis(
        fitting.transfers.axis()
    )
    return TensorAxis(
        structure, (fitting.n_aux,) * fitting.transfers.n_kpt, conjugated=True
    )


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
    if structure.k_axis is not None:
        raise InputError(
            "the orbitals belong to a cell; transform its periodic fitting with "
            "transform_periodic_fitting"
        )
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
    bra_dense = spaces.materialized(bra)
    ket_dense = spaces.materialized(ket)
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


def transform_periodic_fitting(
    fitting: PeriodicDensityFitting,
    spaces: OrbitalSpaces,
    bra: Space,
    ket: Space,
    *,
    signed: bool = False,
    threshold: float = DEFAULT_METRIC_THRESHOLD,
) -> BlockTensor:
    """Bring the fitted tensor of a cell to the Bloch orbital basis, block by block.

    ``B[P^q, p k1, q k2] = Σ_uv conj(C[u, p](k1)) B[k1, k2, P, u, v] C[v, q](k2)
    / √N_k`` for the pairs of points whose transfer is ``q``, with ``p``
    in the ``bra`` space, conjugated, and ``q`` in the ``ket`` space; the
    factor ``1/√N_k`` puts the orbital-basis integrals in the supercell
    normalization, so that the correlated equations of the supercell
    apply as they stand and its energies divide by ``N_k`` per cell
    (ADR-0031). With ``signed`` the signature of the metric eigenvalues
    multiplies the auxiliary index, so that the product of a signed and an
    unsigned tensor, ``Σ_P B conj(B')``, is a fitted integral. The
    orbitals at a derived point are the time-reversed orbitals of its
    stored partner, as the tensor's derived blocks assume. To transform
    one fitting several times, form its tensors once with
    :func:`fitted_periodic_tensors` and call :func:`transform_fitted_tensors`.

    Parameters
    ----------
    fitting
        The periodic fitting of the cell over the mesh of the state.
    threshold
        Eigenvalues of the metric dropped, see
        :meth:`PeriodicDensityFitting.df_tensor`.
    """
    return transform_fitted_tensors(
        fitted_periodic_tensors(fitting, threshold), spaces, bra, ket, signed=signed
    )


@dataclass(frozen=True, slots=True, eq=False)
class FittedPeriodicTensors:
    """The fitted tensor of a cell with its signature, formed once.

    Parameters
    ----------
    fitting
        The periodic fitting the tensors come from.
    df_tensor : Array[kpt, kpt, aux, ao, ao]
        :meth:`PeriodicDensityFitting.df_tensor`.
    signature : Array[kpt, kpt, aux]
        :meth:`PeriodicDensityFitting.signature` at the same threshold.
    """

    fitting: PeriodicDensityFitting
    df_tensor: Array
    signature: Array


def fitted_periodic_tensors(
    fitting: PeriodicDensityFitting, threshold: float = DEFAULT_METRIC_THRESHOLD
) -> FittedPeriodicTensors:
    """Form the fitted tensor and the signature of ``fitting`` once."""
    return FittedPeriodicTensors(
        fitting,
        fitting.df_tensor(threshold=threshold),
        fitting.signature(threshold=threshold),
    )


def transform_fitted_tensors(
    tensors: FittedPeriodicTensors,
    spaces: OrbitalSpaces,
    bra: Space,
    ket: Space,
    *,
    signed: bool = False,
) -> BlockTensor:
    """Transform formed tensors of a cell, see :func:`transform_periodic_fitting`."""
    fitting = tensors.fitting
    structure = spaces.structure
    axis = structure.k_axis
    if axis is None:
        raise InputError(
            "the orbitals are molecular; transform the fitted tensor of a molecule "
            "with transform_df_tensor"
        )
    if not axis.same_points(fitting.kpoints.axis()):
        raise InputError(
            "the orbitals and the periodic fitting run over different k-point sets"
        )
    if fitting.n_ao != spaces.n_ao:
        raise InputError(
            f"the fitted tensor covers {fitting.n_ao} atomic orbitals, the orbitals "
            f"{spaces.n_ao}"
        )
    df_tensor, signature = tensors.df_tensor, tensors.signature
    xp = array_namespace(df_tensor)
    axes = (
        transfer_axis(fitting),
        spaces.axis(bra, conjugated=True),
        spaces.axis(ket, conjugated=False),
    )
    tensor_structure = TensorStructure(axes, ring=Ring.COMPLEX)
    base = structure.sector_structure
    bra_dense = spaces.materialized(bra)
    ket_dense = spaces.materialized(ket)
    n_kpt = fitting.kpoints.n_kpt
    scale = 1.0 / math.sqrt(n_kpt)
    blocks: dict[Key, Array] = {}
    for key in tensor_structure.stored_keys:
        transfer, p_bra, p_ket = key
        spin_bra, k_bra = divmod(p_bra, n_kpt)
        spin_ket, k_ket = divmod(p_ket, n_kpt)
        if (
            spin_bra != spin_ket
            or int(fitting.transfer_index[k_bra, k_ket]) != transfer
        ):
            raise StructureError(
                "a stored block of the fitted tensor does not match a pair of "
                "k-points of one spin block"
            )
        block = _transform(
            df_tensor[k_bra, k_ket],
            bra_dense[spin_bra][k_bra],
            ket_dense[spin_ket][k_ket],
            base.blocks[spin_bra].spin,
        )
        if signed:
            block = block * signature[k_bra, k_ket][:, None, None]
        blocks[key] = block * scale
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


@dataclass(frozen=True, slots=True, eq=False)
class FittedOrbitalTensors:
    """The fitted tensor between every pair of orbital spaces, formed on demand.

    ``B[P, p, q]`` for ``p`` in one space and ``q`` in another is
    :func:`transform_df_tensor` of a molecule's fitted tensor or
    :func:`transform_fitted_tensors` of a cell's, kept once per pair of
    spaces; a cell's tensors come in the plain and the signed form (the
    signature of the metric on the auxiliary index), a molecule's signed
    form is the plain one. From them :meth:`antisymmetrized` assembles the
    integrals ``<pq||rs>`` of any block.

    Parameters
    ----------
    spaces
        The orbital spaces of the reference.
    n_cells
        Cells in the supercell of the mesh, the number of k-points, so
        that an energy of the supercell divides by it per cell; one for
        a molecule.
    """

    spaces: OrbitalSpaces
    n_cells: int
    _molecular: tuple[Array, IrrepSectors | None] | None
    _periodic: FittedPeriodicTensors | None
    _cache: dict[tuple[Space, Space, bool], BlockTensor] = field(
        default_factory=lambda: {}, repr=False
    )

    @classmethod
    def from_df_tensor(
        cls,
        df_tensor: Array,
        spaces: OrbitalSpaces,
        *,
        auxiliary: IrrepSectors | None = None,
    ) -> FittedOrbitalTensors:
        """Wrap the fitted tensor of a molecule, see :func:`transform_df_tensor`."""
        if spaces.structure.k_axis is not None:
            raise InputError(
                "the orbitals belong to a cell; transform its periodic fitting with "
                "transform_periodic_fitting"
            )
        return cls(spaces, 1, (df_tensor, auxiliary), None)

    @classmethod
    def from_periodic_fitting(
        cls,
        fitting: PeriodicDensityFitting,
        spaces: OrbitalSpaces,
        *,
        threshold: float = DEFAULT_METRIC_THRESHOLD,
    ) -> FittedOrbitalTensors:
        """Wrap the fitting of a cell, see :func:`transform_periodic_fitting`."""
        if spaces.structure.k_axis is None:
            raise InputError(
                "the orbitals are molecular; transform the fitted tensor of a "
                "molecule with transform_df_tensor"
            )
        return cls(
            spaces,
            fitting.kpoints.n_kpt,
            None,
            fitted_periodic_tensors(fitting, threshold),
        )

    @classmethod
    def build(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        system: Molecule | Cell,
        auxiliary: BasisSet,
        reference: ScfResult,
        *,
        kpoints: KPoints | None = None,
        threshold: float = DEFAULT_METRIC_THRESHOLD,
    ) -> FittedOrbitalTensors:
        """Fit the two-electron integrals of ``system`` for ``reference``.

        The ``auxiliary`` basis fits with the Coulomb metric, whose
        eigenvalues below ``threshold`` of the largest are dropped
        (:meth:`~hafnium.integrals.DensityFitting.df_tensor`). A molecular
        reference over irrep sectors adapts the auxiliary basis to its
        group, in whose standard frame the molecule is. A cell takes the
        ``kpoints`` of the reference and fits with momentum transfer
        (ADR-0019). Any Fukutome class serves as the reference; its
        orbitals must be occupied by whole electrons.
        """
        spaces = OrbitalSpaces.from_scf(reference)
        if isinstance(system, Cell):
            if kpoints is None:
                raise InputError("a cell needs the k-points of the reference")
            fitting = density_fitting_k(engine, basis, auxiliary, system, kpoints)
            return cls.from_periodic_fitting(fitting, spaces, threshold=threshold)
        if kpoints is not None:
            raise InputError("a molecule has no k-points; drop them")
        sectors = reference.structure.irreps
        adapted = (
            None
            if sectors is None
            else SymmetryAdaptation.from_basis(
                auxiliary, system, PointGroup.named(sectors.group)
            ).irreps()
        )
        molecular = DensityFitting.build(engine, basis, auxiliary, system)
        return cls.from_df_tensor(
            molecular.df_tensor(threshold=threshold), spaces, auxiliary=adapted
        )

    @property
    def ring(self) -> Ring:
        """Ring the tensors are stored over: complex for a cell's."""
        return Ring.COMPLEX if self._periodic is not None else self.spaces.ring

    def tensor(self, bra: Space, ket: Space, *, signed: bool = False) -> BlockTensor:
        """Return ``B[P, p, q]``, ``p`` in ``bra`` and conjugated, ``q`` in ``ket``.

        With ``signed`` the signature of a cell's metric multiplies the
        auxiliary index; a molecule has none.
        """
        signed = signed and self._periodic is not None
        cached = self._cache.get((bra, ket, signed))
        if cached is not None:
            return cached
        if self._molecular is not None:
            df_tensor, auxiliary = self._molecular
            tensor = transform_df_tensor(
                df_tensor, self.spaces, bra, ket, auxiliary=auxiliary
            )
        else:
            assert self._periodic is not None
            tensor = transform_fitted_tensors(
                self._periodic, self.spaces, bra, ket, signed=signed
            )
        self._cache[bra, ket, signed] = tensor
        return tensor

    def antisymmetrized(self, slots: tuple[Space, Space, Space, Space]) -> BlockTensor:
        """``<pq||rs> = (pr|qs) − (ps|qr)`` as the block tensor ``[p, q, r, s]``.

        ``slots`` holds the space of each index, occupied, occupied,
        virtual, virtual for ``<ij||ab>``; ``p`` and ``q`` are conjugated
        (CONVENTIONS.md §3).
        ``(pr|qs) = Σ_P B[P, p, r] conj(B[P, s, q])`` pairs the fitted
        tensor with the conjugate of the signed one, ``B[P, q, s] =
        conj(B[P, s, q])`` for the real auxiliary functions of a molecule
        and, with the signature of the metric, for the Bloch functions of
        a cell; the sum over the auxiliary index then pairs an
        unconjugated with a conjugated axis.
        """
        p, q, r, s = slots
        direct = contract(
            "Ppr,Psq->pqrs", self.tensor(p, r), self.tensor(s, q, signed=True).conj()
        )
        exchange = contract(
            "Pps,Prq->pqrs", self.tensor(p, s), self.tensor(r, q, signed=True).conj()
        )
        return direct - exchange
