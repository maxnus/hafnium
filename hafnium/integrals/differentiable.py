"""Integrals as differentiable functions of the shell centres (ADR-0037).

The compiled kernels read a shell table and return host arrays; nothing in
them can be traced. :class:`DifferentiableIntegrals` holds the static part
of the shell tables of a basis and its auxiliary basis, the angular momenta,
exponents and contraction layout, and exposes each integral type as a
function of the centres of the shells, an array ``[shell, 3]``. Under JAX
the function is a host callback with a custom vector-Jacobian product from
the derivative integrals of ADR-0035, so it traces under ``jit`` and
differentiates under ``grad``; under any other backend it evaluates the
kernel directly. The centres of the shells follow the atoms through
:meth:`DifferentiableIntegrals.orbital_centers`, a gather written in the
array namespace, so a derivative with respect to the atomic positions is the
chain rule through it. The exponents and contraction coefficients are
static for now; their derivatives are a later part of M9.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from hafnium.backend import (
    Array,
    array_namespace,
    host_array,
    host_namespace,
    is_jax_array,
    to_host,
)
from hafnium.basis import BasisSet
from hafnium.errors import InputError
from hafnium.integrals.engine import (
    COULOMB,
    CoulombKernel,
    IntegralEngine,
    PointCharges,
)
from hafnium.integrals.shell_table import ShellTable
from hafnium.molecule import AtomicSystem


@dataclass(frozen=True, slots=True, eq=False)
class DifferentiableIntegrals:
    """The integrals of a basis as functions of the centres of its shells.

    Parameters
    ----------
    engine
        The integral engine that evaluates the kernels on the host.
    orbital
        The shell table of the orbital basis; its centres are replaced by
        the argument of every method.
    orbital_atoms : Array[shell]
        The atom of every orbital shell.
    auxiliary
        The shell table of the auxiliary basis, for the fitting integrals;
        ``None`` without one.
    auxiliary_atoms : Array[shell]
        The atom of every auxiliary shell; ``None`` without an auxiliary
        basis.
    """

    engine: IntegralEngine
    orbital: ShellTable
    orbital_atoms: Array
    auxiliary: ShellTable | None = None
    auxiliary_atoms: Array | None = None
    _shell_maps: tuple[Array, Array | None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "orbital_atoms", host_array(self.orbital_atoms, "int64")
        )
        if self.orbital_atoms.shape != (self.orbital.n_shells,):
            raise InputError("one atom per orbital shell")
        if (self.auxiliary is None) != (self.auxiliary_atoms is None):
            raise InputError("an auxiliary basis comes with the atoms of its shells")
        auxiliary_map = None
        if self.auxiliary is not None and self.auxiliary_atoms is not None:
            atoms = host_array(self.auxiliary_atoms, "int64")
            object.__setattr__(self, "auxiliary_atoms", atoms)
            if atoms.shape != (self.auxiliary.n_shells,):
                raise InputError("one atom per auxiliary shell")
            auxiliary_map = _shell_map(self.auxiliary)
        object.__setattr__(
            self, "_shell_maps", (_shell_map(self.orbital), auxiliary_map)
        )

    @classmethod
    def from_basis(
        cls,
        engine: IntegralEngine,
        basis: BasisSet,
        system: AtomicSystem,
        auxiliary: BasisSet | None = None,
    ) -> DifferentiableIntegrals:
        """Lay out ``basis``, and ``auxiliary``, on the atoms of ``system``."""
        fitting = (
            None if auxiliary is None else ShellTable.from_basis(auxiliary, system)
        )
        return cls(
            engine,
            ShellTable.from_basis(basis, system),
            host_array(basis.shell_atoms, "int64"),
            fitting,
            None if auxiliary is None else host_array(auxiliary.shell_atoms, "int64"),
        )

    # ----------------------------------------------------------------- centres

    def orbital_centers(self, positions: Array) -> Array:
        """Return the centres ``[shell, 3]`` of the orbital shells at ``positions``."""
        return _gather(positions, self.orbital_atoms)

    def auxiliary_centers(self, positions: Array) -> Array:
        """Return the centres ``[shell, 3]`` of the auxiliary shells."""
        return _gather(positions, self._auxiliary_atoms())

    # --------------------------------------------------------------- integrals

    def overlap(self, centers: Array) -> Array:
        """Return the overlap ``[ao, ao]`` of the orbital basis at ``centers``."""
        if is_jax_array(centers):
            return _rules().overlap(self, centers)
        return self.host_overlap(centers)

    def kinetic(self, centers: Array) -> Array:
        """Return the kinetic energy ``[ao, ao]`` of the orbital basis."""
        if is_jax_array(centers):
            return _rules().kinetic(self, centers)
        return self.host_kinetic(centers)

    def nuclear_attraction(
        self,
        centers: Array,
        charge_positions: Array,
        charges: Array,
        *,
        kernel: CoulombKernel = COULOMB,
    ) -> Array:
        """Return the attraction ``[ao, ao]`` to ``charges`` at ``charge_positions``.

        Differentiable in the centres and in the positions of the charges;
        the charges themselves are constants.
        """
        if is_jax_array(charges):
            raise InputError(
                "the charges are constants of the attraction; differentiate the "
                "positions of the charges instead"
            )
        if is_jax_array(centers) or is_jax_array(charge_positions):
            return _rules().nuclear_attraction(
                self,
                kernel,
                tuple(float(z) for z in to_host(charges)),
                centers,
                charge_positions,
            )
        return self.host_nuclear_attraction(
            centers, charge_positions, host_array(charges), kernel
        )

    def metric(
        self, auxiliary_centers: Array, *, kernel: CoulombKernel = COULOMB
    ) -> Array:
        """Return the two-centre integrals ``(P|Q)`` ``[aux, aux]``."""
        if is_jax_array(auxiliary_centers):
            return _rules().metric(self, kernel, auxiliary_centers)
        return self.host_metric(auxiliary_centers, kernel)

    def three_center(
        self,
        auxiliary_centers: Array,
        centers: Array,
        *,
        kernel: CoulombKernel = COULOMB,
    ) -> Array:
        """Return the three-centre integrals ``(P|uv)`` ``[aux, ao, ao]``."""
        if is_jax_array(auxiliary_centers) or is_jax_array(centers):
            return _rules().three_center(self, kernel, auxiliary_centers, centers)
        return self.host_three_center(auxiliary_centers, centers, kernel)

    # ------------------------------------------------- host evaluation (kernels)

    def orbital_table(self, centers: Array) -> ShellTable:
        """Return the orbital shell table with its centres replaced by ``centers``."""
        return _with_centers(self.orbital, centers)

    def auxiliary_table(self, centers: Array) -> ShellTable:
        """Return the auxiliary shell table with its centres replaced by ``centers``."""
        return _with_centers(self.fitting(), centers)

    def fitting(self) -> ShellTable:
        """Return the auxiliary shell table, which the fitting integrals need."""
        if self.auxiliary is None:
            raise InputError("no auxiliary basis was given")
        return self.auxiliary

    def host_overlap(self, centers: Array) -> Array:
        """Evaluate :meth:`overlap` on the host, in the namespace of ``centers``."""
        table = self.orbital_table(centers)
        return _back(centers, self.engine.overlap(table, table, no_translation())[0])

    def host_kinetic(self, centers: Array) -> Array:
        """Evaluate :meth:`kinetic` on the host, in the namespace of ``centers``."""
        table = self.orbital_table(centers)
        return _back(centers, self.engine.kinetic(table, table, no_translation())[0])

    def host_nuclear_attraction(
        self,
        centers: Array,
        charge_positions: Array,
        charges: Array,
        kernel: CoulombKernel,
    ) -> Array:
        """Evaluate :meth:`nuclear_attraction` on the host."""
        table = self.orbital_table(centers)
        attraction = self.engine.nuclear_attraction(
            table,
            table,
            no_translation(),
            PointCharges(host_array(charges), to_host(charge_positions)),
            kernel=kernel,
        )[0]
        return _back(centers, attraction)

    def host_metric(self, auxiliary_centers: Array, kernel: CoulombKernel) -> Array:
        """Evaluate :meth:`metric` on the host."""
        table = self.auxiliary_table(auxiliary_centers)
        metric = self.engine.coulomb_two_center(
            table, table, no_translation(), kernel=kernel
        )[0]
        return _back(auxiliary_centers, metric)

    def host_three_center(
        self, auxiliary_centers: Array, centers: Array, kernel: CoulombKernel
    ) -> Array:
        """Evaluate :meth:`three_center` on the host."""
        fitting = self.auxiliary_table(auxiliary_centers)
        table = self.orbital_table(centers)
        integrals = self.engine.coulomb_three_center(
            fitting, table, table, no_pair_translation(), kernel=kernel
        )[0]
        return _back(centers, integrals)

    # ------------------------------------------------- the derivative kernels

    def orbital_shells(self) -> Array:
        """``[shell, ao]`` with one where the function belongs to the shell."""
        return self._shell_maps[0]

    def auxiliary_shells(self) -> Array:
        """``[shell, aux]`` with one where the function belongs to the shell."""
        self.fitting()
        auxiliary_map = self._shell_maps[1]
        assert auxiliary_map is not None
        return auxiliary_map

    def _auxiliary_atoms(self) -> Array:
        if self.auxiliary_atoms is None:
            raise InputError("no auxiliary basis was given")
        return self.auxiliary_atoms


def nuclear_repulsion(positions: Array, charges: Array) -> Array:
    """``Σ_{K<L} Z_K Z_L / |R_K − R_L|`` of point ``charges`` at ``positions``.

    A 0-d array in the namespace of ``positions``, differentiable in them.
    """
    xp = array_namespace(positions)
    charges = xp.asarray(charges)
    n = int(positions.shape[0])
    separations = positions[:, None, :] - positions[None, :, :]
    squared = xp.sum(separations**2, axis=-1)
    upper = xp.triu(xp.ones((n, n), dtype=xp.bool), k=1)
    safe = xp.where(upper, squared, xp.ones_like(squared))
    products = charges[:, None] * charges[None, :]
    return xp.sum(xp.where(upper, products / xp.sqrt(safe), xp.zeros_like(squared)))


def _rules() -> Any:
    """Import the JAX rules when a JAX array arrives, so JAX stays optional."""
    from hafnium.integrals import _jax_rules

    return _jax_rules


def _gather(positions: Array, atoms: Array) -> Array:
    xp = array_namespace(positions)
    return xp.take(positions, xp.asarray(atoms), axis=0)


def _with_centers(table: ShellTable, centers: Array) -> ShellTable:
    host = to_host(centers)
    if host.shape != (table.n_shells, 3):
        raise InputError(f"the centres are [shell, 3] with {table.n_shells} shells")
    return replace(table, centers=host)


def _back(like: Array, host: Array) -> Array:
    """``host`` in the namespace of ``like``."""
    return array_namespace(like).asarray(host)


def no_translation() -> Array:
    """Return the single zero translation of a molecule, fresh for every call."""
    return host_array([[0.0, 0.0, 0.0]])


def no_pair_translation() -> Array:
    """Return the single zero translation of both functions of a pair."""
    return host_array([[[0.0] * 3, [0.0] * 3]])


def _shell_map(table: ShellTable) -> Array:
    xp = host_namespace()
    counts = [
        int(k) * (2 * int(l) + 1)
        for k, l in zip(table.contraction_counts, table.angular_momenta, strict=True)
    ]
    shell_of_function = xp.asarray(
        [shell for shell, count in enumerate(counts) for _ in range(count)],
        dtype=xp.int64,
    )
    n_shells = len(counts)
    return xp.astype(
        xp.arange(n_shells)[:, None] == shell_of_function[None, :], xp.float64
    )
