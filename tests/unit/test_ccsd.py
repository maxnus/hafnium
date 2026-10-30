"""DF-CCSD over block tensors: one generated program for every class (ADR-0032)."""

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.cc import (
    AmplitudeDiis,
    CcsdOptions,
    CcsdResult,
    ccsd_from_df_tensor,
    df_ccsd,
    solve_ccsd,
)
from hafnium.correlated import (
    FittedOrbitalTensors,
    OrbitalSpaces,
    Space,
    fock_block,
)
from hafnium.errors import ConvergenceError, InputError
from hafnium.molecule import Molecule
from hafnium.mp import mp2_from_df_tensor
from hafnium.scf import (
    DensityFittedCoulombExchange,
    Hamiltonian,
    ScfResult,
    solve_scf,
)
from hafnium.symmetry import KPoints, SymmetrySpec, detect_point_group
from tests.hamiltonians import ENGINE, hamiltonian_for
from tests.molecules import hydrogen, water, water_cation
from tests.shell_tables import LIBRARY
from tests.tolerances import CCSD_ENERGY_TOL, MP2_ENERGY_TOL, STRUCTURE_TOL
from tests.unit.test_mp2 import AUXILIARY, TIGHT, fitted_tensor
from tests.unit.test_scf import CLASSES, OPEN_SHELL_CLASSES

# The comparisons across classes and with FCI are at CCSD_ENERGY_TOL, so
# the amplitudes converge two orders further.
OPTIONS = CcsdOptions(energy_tolerance=1e-12, amplitude_tolerance=1e-10)


@pytest.fixture(scope="module")
def water_hamiltonian() -> Hamiltonian:
    return hamiltonian_for(water(), "sto-3g", AUXILIARY)[1]


@pytest.fixture(scope="module")
def cation_hamiltonian() -> Hamiltonian:
    return hamiltonian_for(water_cation(), "sto-3g", AUXILIARY)[1]


@pytest.fixture(scope="module")
def restricted_ccsd(water_hamiltonian: Hamiltonian) -> CcsdResult:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name("RHF"), options=TIGHT
    )
    return ccsd_from_df_tensor(
        reference, fitted_tensor(water_hamiltonian), options=OPTIONS
    )


def spin_orbital_integrals(
    reference: ScfResult, df_tensor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(pq|rs)`` over the spin orbitals of ``reference`` and ``<pq||rs>``."""
    n = df_tensor.shape[1]
    eri = np.einsum("Puv,Pwx->uvwx", df_tensor, df_tensor)
    c = reference.mo_coefficients.to_dense(orbitals=True)
    chemists = np.zeros((c.shape[1],) * 4, dtype=c.dtype)
    for first in (c[:n], c[n:]):
        half = np.einsum("up,vq,uvwx->pqwx", first.conj(), first, eri)
        for second in (c[:n], c[n:]):
            chemists += np.einsum("pqwx,wr,xs->pqrs", half, second.conj(), second)
    antisymmetrized = np.einsum("prqs->pqrs", chemists) - np.einsum(
        "psqr->pqrs", chemists
    )
    return chemists, antisymmetrized


def two_electron_fci_correlation(
    reference: ScfResult, hamiltonian: Hamiltonian, df_tensor: np.ndarray
) -> float:
    """The exact correlation energy of two electrons in the fitted integrals.

    The Hamiltonian in the basis of all two-electron determinants
    ``a†_p a†_q |0>`` (``p < q``) by the Slater–Condon rules, its lowest
    eigenvalue minus the energy of the reference determinant.
    """
    _, antisymmetrized = spin_orbital_integrals(reference, df_tensor)
    core = np.asarray(hamiltonian.core_hamiltonian)
    n = core.shape[0]
    c = reference.mo_coefficients.to_dense(orbitals=True)
    one_body = c[:n].conj().T @ core @ c[:n] + c[n:].conj().T @ core @ c[n:]
    occupied = [
        int(k) for k in np.flatnonzero(reference.mo_occupations.to_dense() > 0.5)
    ]
    assert len(occupied) == 2
    pairs = [(p, q) for p in range(c.shape[1]) for q in range(p + 1, c.shape[1])]
    matrix = np.zeros((len(pairs), len(pairs)), dtype=c.dtype)
    for row, (p, q) in enumerate(pairs):
        for column, (r, s) in enumerate(pairs):
            value = antisymmetrized[p, q, r, s]
            if q == s:
                value += one_body[p, r]
            if p == r:
                value += one_body[q, s]
            if p == s:
                value -= one_body[q, r]
            if q == r:
                value -= one_body[p, s]
            matrix[row, column] = value
    lowest = float(np.linalg.eigvalsh(matrix)[0])
    determinant = pairs.index((occupied[0], occupied[1]))
    return lowest - float(matrix[determinant, determinant].real)


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_gives_the_closed_shell_energy(
    name: str, water_hamiltonian: Hamiltonian, restricted_ccsd: CcsdResult
) -> None:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    df_tensor = fitted_tensor(water_hamiltonian)
    result = ccsd_from_df_tensor(reference, df_tensor, options=OPTIONS)
    assert result.converged
    assert result.correlation_energy == pytest.approx(
        restricted_ccsd.correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert (
        result.correlation_energy
        < mp2_from_df_tensor(reference, df_tensor).correlation_energy
        < 0.0
    )
    # The singles start from zero and the doubles from first order, so the
    # first iteration evaluates the MP2 energy.
    assert result.history[0].correlation_energy == pytest.approx(
        mp2_from_df_tensor(reference, df_tensor).correlation_energy,
        abs=MP2_ENERGY_TOL,
    )
    assert result.reference_energy == pytest.approx(reference.energy)
    assert result.total_energy == pytest.approx(
        reference.energy + result.correlation_energy
    )
    assert result.n_iterations == len(result.history) > 1
    assert [axis.conjugated for axis in result.singles.structure.axes] == [False, True]
    assert [axis.conjugated for axis in result.doubles.structure.axes] == [
        False,
        False,
        True,
        True,
    ]
    derived = reference.structure.derived_partner is not None
    assert (result.doubles.structure.n_stored < len(result.doubles.structure.keys)) == (
        derived
    )


@pytest.mark.parametrize("name", OPEN_SHELL_CLASSES)
def test_open_shell_classes_agree(name: str, cation_hamiltonian: Hamiltonian) -> None:
    unrestricted = solve_scf(
        cation_hamiltonian, SymmetrySpec.from_name("UHF"), options=TIGHT
    )
    df_tensor = fitted_tensor(cation_hamiltonian)
    expected = ccsd_from_df_tensor(unrestricted, df_tensor, options=OPTIONS)
    reference = solve_scf(
        cation_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    result = ccsd_from_df_tensor(reference, df_tensor, options=OPTIONS)
    assert result.correlation_energy == pytest.approx(
        expected.correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert result.correlation_energy < 0.0


@pytest.mark.parametrize("name", ["RHF", "UHF", "complex GHF"])
def test_two_electron_ccsd_is_exact(name: str) -> None:
    """For two electrons ``e^T`` spans the full space: CCSD equals FCI."""
    molecule = hydrogen()
    _, hamiltonian = hamiltonian_for(molecule, "cc-pvdz", AUXILIARY)
    reference = solve_scf(hamiltonian, SymmetrySpec.from_name(name), options=TIGHT)
    df_tensor = fitted_tensor(hamiltonian)
    result = ccsd_from_df_tensor(reference, df_tensor, options=OPTIONS)
    assert result.correlation_energy == pytest.approx(
        two_electron_fci_correlation(reference, hamiltonian, df_tensor),
        abs=CCSD_ENERGY_TOL,
    )


@pytest.mark.parametrize("name", ["RHF", "UHF", "GHF"])
def test_blocked_reference_gives_the_same_energy(name: str) -> None:
    molecule = water()
    detection = detect_point_group(molecule)
    oriented = detection.orient(molecule)
    basis = BasisSet.for_atoms(oriented, "sto-3g", library=LIBRARY)
    auxiliary = BasisSet.for_atoms(oriented, AUXILIARY, library=LIBRARY)
    two_body = DensityFittedCoulombExchange.from_molecule(
        ENGINE, basis, auxiliary, oriented
    )
    plain = Hamiltonian.from_molecule(ENGINE, basis, oriented, two_body)
    blocked = Hamiltonian.from_molecule(
        ENGINE, basis, oriented, two_body, point_group=detection.group
    )
    spec = SymmetrySpec.from_name(name)
    expected = df_ccsd(
        ENGINE,
        basis,
        oriented,
        auxiliary,
        solve_scf(plain, spec, options=TIGHT),
        options=OPTIONS,
    )
    reference = solve_scf(
        blocked, spec.with_point_group(detection.group), options=TIGHT
    )
    result = df_ccsd(ENGINE, basis, oriented, auxiliary, reference, options=OPTIONS)
    assert result.correlation_energy == pytest.approx(
        expected.correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert all(
        axis.structure.irreps is not None for axis in result.doubles.structure.axes
    )
    assert result.doubles.structure.n_stored > expected.doubles.structure.n_stored


def test_integral_blocks_match_the_dense_spin_orbital_formula(
    cation_hamiltonian: Hamiltonian,
) -> None:
    reference = solve_scf(cation_hamiltonian, SymmetrySpec.from_name("complex UHF"))
    df_tensor = fitted_tensor(cation_hamiltonian)
    spaces = OrbitalSpaces.from_scf(reference)
    fitted = FittedOrbitalTensors.from_df_tensor(df_tensor, spaces)
    _, antisymmetrized = spin_orbital_integrals(reference, df_tensor)
    occupied = reference.mo_occupations.to_dense() > 0.5
    selectors = {"o": occupied, "v": ~occupied}
    for slots in ("oovv", "ovvv", "vvoo", "ooov", "vvvv", "ovov"):
        expected = antisymmetrized
        for axis, letter in enumerate(slots):
            expected = np.compress(selectors[letter], expected, axis=axis)
        block = fitted.antisymmetrized(slots)
        assert [a.conjugated for a in block.structure.axes] == [
            True,
            True,
            False,
            False,
        ]
        np.testing.assert_allclose(block.to_dense(), expected, atol=STRUCTURE_TOL)
    assert fitted.ring is spaces.ring
    assert fitted.tensor(Space.OCCUPIED, Space.VIRTUAL, signed=True) is fitted.tensor(
        Space.OCCUPIED, Space.VIRTUAL
    )
    with pytest.raises(InputError, match="four slots"):
        fitted.antisymmetrized("oov")
    with pytest.raises(InputError, match="occupied"):
        fitted.antisymmetrized("ooxv")


def test_fock_blocks_are_diagonal_in_the_canonical_orbitals(
    water_hamiltonian: Hamiltonian,
) -> None:
    reference = solve_scf(water_hamiltonian, SymmetrySpec.from_name("UHF"))
    spaces = OrbitalSpaces.from_scf(reference)
    for slots, energies in (
        ("oo", spaces.occupied_energies),
        ("vv", spaces.virtual_energies),
    ):
        block = fock_block(spaces, slots)
        assert block.structure.keys == ((0, 0), (1, 1))
        for position in range(2):
            np.testing.assert_allclose(
                block.block((position, position)),
                np.diag(energies.blocks[position]),
                atol=STRUCTURE_TOL,
            )
    assert np.all(fock_block(spaces, "ov").to_dense() == 0.0)
    assert np.all(fock_block(spaces, "vo").to_dense() == 0.0)
    with pytest.raises(InputError, match="two slots"):
        fock_block(spaces, "oov")


def test_options_and_convergence_are_reported(water_hamiltonian: Hamiltonian) -> None:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name("RHF"), options=TIGHT
    )
    df_tensor = fitted_tensor(water_hamiltonian)
    with pytest.raises(ConvergenceError) as excinfo:
        ccsd_from_df_tensor(reference, df_tensor, options=CcsdOptions(max_iterations=2))
    last = excinfo.value.last
    assert isinstance(last, CcsdResult)
    assert not last.converged
    assert last.n_iterations == 2
    # Without DIIS the iteration still converges, more slowly.
    plain = ccsd_from_df_tensor(reference, df_tensor, options=CcsdOptions(diis_size=1))
    extrapolated = ccsd_from_df_tensor(reference, df_tensor)
    assert plain.correlation_energy == pytest.approx(
        extrapolated.correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert plain.n_iterations > extrapolated.n_iterations
    for bad in (
        {"max_iterations": 0},
        {"energy_tolerance": 0.0},
        {"amplitude_tolerance": -1.0},
        {"diis_size": 0},
    ):
        with pytest.raises(InputError):
            CcsdOptions(**bad)  # type: ignore[arg-type]
    # Converged amplitudes restart and stop the iteration at once.
    fitted = FittedOrbitalTensors.from_df_tensor(
        df_tensor, OrbitalSpaces.from_scf(reference)
    )
    again = solve_ccsd(
        reference,
        fitted,
        CcsdOptions(max_iterations=1),
        guess=(extrapolated.singles, extrapolated.doubles),
    )
    assert again.converged
    assert again.n_iterations == 1
    assert again.correlation_energy == pytest.approx(
        extrapolated.correlation_energy, abs=CCSD_ENERGY_TOL
    )
    with pytest.raises(InputError, match="starting amplitudes"):
        solve_ccsd(
            reference, fitted, guess=(extrapolated.doubles, extrapolated.singles)
        )


def test_amplitude_diis_validates_its_history(water_hamiltonian: Hamiltonian) -> None:
    reference = solve_scf(water_hamiltonian, SymmetrySpec.from_name("RHF"))
    amplitudes = mp2_from_df_tensor(
        reference, fitted_tensor(water_hamiltonian)
    ).amplitudes
    with pytest.raises(InputError, match="at least one"):
        AmplitudeDiis(0)
    diis = AmplitudeDiis(2)
    with pytest.raises(InputError, match="empty"):
        diis.extrapolate()
    with pytest.raises(InputError, match="error tensor"):
        diis.push((amplitudes,), ())
    diis = diis.push((amplitudes,), (amplitudes,))
    assert len(diis) == 1
    assert diis.extrapolate()[0] is amplitudes
    diis = diis.push((2.0 * amplitudes,), (0.5 * amplitudes,))
    diis = diis.push((3.0 * amplitudes,), (0.25 * amplitudes,))
    assert len(diis) == 2
    (combined,) = diis.extrapolate()
    # The errors are parallel, so the combination minimizes along the line.
    np.testing.assert_allclose(
        combined.to_dense(),
        (2.0 * -1.0 + 3.0 * 2.0) * amplitudes.to_dense(),
        atol=STRUCTURE_TOL,
    )


def test_molecular_inputs_are_validated(water_hamiltonian: Hamiltonian) -> None:
    reference = solve_scf(water_hamiltonian, SymmetrySpec.from_name("RHF"))
    molecule = Molecule.from_atoms([("He", (0.0, 0.0, 0.0))])
    basis = BasisSet.for_atoms(molecule, "sto-3g", library=LIBRARY)
    auxiliary = BasisSet.for_atoms(molecule, AUXILIARY, library=LIBRARY)
    with pytest.raises(InputError, match="no k-points"):
        df_ccsd(ENGINE, basis, molecule, auxiliary, reference, kpoints=KPoints.gamma())
