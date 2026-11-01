"""The analytic nuclear gradient of a density-fitted Hartree–Fock energy (ADR-0035).

The energy ``E = E_nuc + tr(D h) + tr(D J) / 2 − tr(D K) / 2`` is
stationary in the orbitals, so its derivative with respect to a nuclear
position ``R`` is the derivative of the integrals at fixed density plus the
term of the energy-weighted density ``W = D F D`` against the derivative
of the overlap: ``dE/dR = dE_nuc/dR + tr(D h^R) − tr(W S^R) + ½ tr(D J^R)
− ½ tr(D K^R)``. With density fitting the two-electron terms are
``E_J = ½ q·c`` and ``E_K = −½ Σ_PQ (M⁻¹)_PQ tr(L_P D L_Q D)`` for the
three-centre integrals ``L_P = (P|uv)``, the metric ``M = (P|Q)``, the
fitted charges ``q_P = tr(L_P D)`` and the coefficients ``c = M⁻¹ q``;
their derivatives pair the three-centre derivatives with an effective
density per auxiliary function and the metric derivative with a matrix
over the auxiliary basis. Every derivative integral is formed for the bra
centre; the ket terms follow from the symmetry of the integrals, and the
sum over the centres of an atom from the atom of every function.
"""

from __future__ import annotations

from dataclasses import dataclass

from hafnium.backend import Array, Namespace, array_namespace, host_array
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals import (
    DensityFitting,
    DensityFittingDerivatives,
    IntegralEngine,
    kinetic_derivative,
    nuclear_attraction_derivative,
    overlap_derivative,
)
from hafnium.integrals.density_fitting import (
    DEFAULT_METRIC_THRESHOLD,
    inverse_square_root,
)
from hafnium.molecule import Molecule
from hafnium.scf import ScfResult
from hafnium.structured import BlockMatrix


@dataclass(frozen=True, slots=True)
class NuclearGradient:
    """The derivative of a mean-field energy with respect to the nuclear positions.

    Each term is ``[atom, 3]`` in Hartree per Bohr: the repulsion of the
    nuclei; the one-electron terms, the kinetic energy and the nuclear
    attraction with the force of the electrons on the nuclei; the overlap
    term of the energy-weighted density; and the Coulomb and the exchange
    terms through the fitted integrals.
    """

    nuclear_repulsion: Array
    one_electron: Array
    overlap: Array
    coulomb: Array
    exchange: Array

    @property
    def total(self) -> Array:
        """The gradient ``[atom, 3]``."""
        return (
            self.nuclear_repulsion
            + self.one_electron
            + self.overlap
            + self.coulomb
            + self.exchange
        )


def nuclear_gradient(
    engine: IntegralEngine,
    basis: BasisSet,
    auxiliary: BasisSet,
    molecule: Molecule,
    result: ScfResult,
) -> NuclearGradient:
    """Gradient of the density-fitted Hartree–Fock energy of ``result``.

    ``result`` is the converged state of ``molecule`` in ``basis`` with the
    two-electron terms fitted in ``auxiliary``, in any Fukutome class and
    with or without point-group sectors; the gradient is first order in
    its residual orbital error.

    Raises
    ------
    InputError
        If ``result`` is a Kohn–Sham state, whose gradient needs the
        derivative of the exchange–correlation term as well, or if it was
        not converged in ``basis``.
    """
    if result.energies.exchange_correlation != 0.0:
        raise InputError(
            "the gradient of a Kohn–Sham energy is not implemented: "
            "the result carries an exchange–correlation energy"
        )
    density = result.density
    if density.n_spatial != basis.n_functions:
        raise InputError(
            f"the state has {density.n_spatial} spatial functions, "
            f"the basis set {basis.n_functions}"
        )
    total = _spatial(density.spin_trace())
    weighted = _spatial((density @ result.fock @ density).spin_trace())
    xp = array_namespace(total)
    atoms = _functions_by_atom(basis, molecule.n_atoms, xp)
    aux_atoms = _functions_by_atom(auxiliary, molecule.n_atoms, xp)

    by_center, by_nucleus = nuclear_attraction_derivative(engine, basis, molecule)
    one_electron = _on_centers(
        kinetic_derivative(engine, basis, molecule) + by_center, total, atoms
    ) + xp.einsum("kxuv,vu->kx", by_nucleus, total)
    overlap = -_on_centers(overlap_derivative(engine, basis, molecule), weighted, atoms)

    fitting = DensityFitting.build(engine, basis, auxiliary, molecule)
    derivatives = DensityFittingDerivatives.build(engine, basis, auxiliary, molecule)
    half = inverse_square_root(fitting.metric, DEFAULT_METRIC_THRESHOLD)
    inverse = half @ half
    three_center = fitting.three_center
    charges = xp.einsum("Puv,vu->P", three_center, total)
    coefficients = inverse @ charges
    coulomb = _two_electron(
        derivatives,
        coefficients[:, None, None] * total[None, :, :],
        -0.5 * coefficients[:, None] * coefficients[None, :],
        atoms,
        aux_atoms,
    )
    pairs = _exchange_pairs(density.to_dense(), three_center)
    n = int(total.shape[0])
    exchange_pairs = (
        xp.reshape(three_center, (-1, n * n))
        @ xp.reshape(xp.permute_dims(pairs, (0, 2, 1)), (-1, n * n)).T
    )
    exchange = _two_electron(
        derivatives,
        -xp.einsum("PQ,Quv->Puv", inverse, pairs),
        0.5 * inverse @ exchange_pairs @ inverse,
        atoms,
        aux_atoms,
    )
    return NuclearGradient(
        _nuclear_repulsion_gradient(molecule, xp),
        one_electron,
        overlap,
        coulomb,
        exchange,
    )


def _spatial(matrix: BlockMatrix) -> Array:
    """Real ``[ao, ao]`` block of a spatial block matrix."""
    xp = array_namespace(*matrix.blocks)
    return xp.real(matrix.spin_blocks()[0])


def _functions_by_atom(basis: BasisSet, n_atoms: int, xp: Namespace) -> Array:
    """``[atom, ao]`` with one where the function sits on the atom."""
    labels = basis.function_labels()
    return xp.asarray(
        host_array(
            [
                [1.0 if label.atom == atom else 0.0 for label in labels]
                for atom in range(n_atoms)
            ]
        )
    )


def _on_centers(derivative: Array, weights: Array, atoms: Array) -> Array:
    """``2 Σ_{u on K} Σ_v derivative[x, u, v] weights[v, u]`` as ``[atom, 3]``.

    The factor two adds the ket term of a symmetric integral.
    """
    xp = array_namespace(derivative, weights)
    rows = 2.0 * xp.einsum("xuv,vu->xu", derivative, weights)
    return (rows @ atoms.T).T


def _two_electron(
    derivatives: DensityFittingDerivatives,
    effective: Array,
    metric_weights: Array,
    atoms: Array,
    aux_atoms: Array,
) -> Array:
    """Gradient of ``Σ_P tr(L_P G_P) + tr(M B)`` at fixed ``G`` and ``B``.

    ``effective[P]`` is the effective density ``G_P`` paired with the
    three-centre integrals and ``metric_weights`` the symmetric matrix
    ``B`` paired with the metric; the orbital term pairs the bra derivative
    with ``G_P + G_Pᵀ`` and the metric term doubles the bra derivative, as
    both integrals are symmetric.
    """
    xp = array_namespace(effective, metric_weights)
    symmetric = effective + xp.permute_dims(effective, (0, 2, 1))
    on_auxiliary = xp.einsum("xPuv,Puv->xP", derivatives.auxiliary, effective)
    on_auxiliary = on_auxiliary + 2.0 * xp.einsum(
        "xPQ,PQ->xP", derivatives.metric, metric_weights
    )
    on_orbital = xp.einsum("xPuv,Puv->xu", derivatives.orbital, symmetric)
    return (on_auxiliary @ aux_atoms.T + on_orbital @ atoms.T).T


def _exchange_pairs(spinor: Array, three_center: Array) -> Array:
    """``T_Q = Σ_s (D L̃_Q D)_{ss}`` as ``[aux, ao, ao]``, real.

    ``D`` is the spin-orbital density ``[2n, 2n]`` and ``L̃_Q`` the
    three-centre integrals of ``Q`` on both spins; the spin trace of the
    product is Hermitian, and its real part is what pairs with the real
    integrals.
    """
    xp = array_namespace(spinor, three_center)
    n = int(three_center.shape[1])
    rows = xp.reshape(spinor, (4 * n, n))
    first = xp.reshape(xp.matmul(rows, three_center), (-1, 2, n, 2 * n))
    pairs = first[:, 0] @ spinor[:, :n] + first[:, 1] @ spinor[:, n:]
    return xp.real(pairs)


def _nuclear_repulsion_gradient(molecule: Molecule, xp: Namespace) -> Array:
    """``−Σ_{L≠K} Z_K Z_L (R_K − R_L) / |R_K − R_L|³`` as ``[atom, 3]``."""
    positions = xp.asarray(molecule.positions_array())
    charges = xp.asarray(host_array(molecule.nuclear_charges))
    separations = positions[:, None, :] - positions[None, :, :]
    distances = xp.sqrt(xp.sum(separations**2, axis=-1))
    n_atoms = int(positions.shape[0])
    safe = xp.where(xp.eye(n_atoms) > 0.0, xp.ones_like(distances), distances)
    weights = xp.where(
        xp.eye(n_atoms) > 0.0,
        xp.zeros_like(distances),
        charges[:, None] * charges[None, :] / safe**3,
    )
    return -xp.sum(weights[:, :, None] * separations, axis=1)
