"""DF-CCSD over block tensors: one generated program for every class (ADR-0032)."""

import numpy as np
import pytest

from hafnium.basis import BasisSet
from hafnium.cc import (
    AmplitudeDiis,
    CcsdLambdaResult,
    CcsdOptions,
    CcsdPtResult,
    CcsdResult,
    TriplesOptions,
    ccsd_from_df_tensor,
    ccsd_lambda_from_df_tensor,
    ccsd_pt_from_df_tensor,
    df_ccsd,
    df_ccsd_lambda,
    df_ccsd_pt,
    solve_ccsd,
    solve_ccsd_pt,
    solve_lambda,
    triples_correction,
)
from hafnium.cc.ccsd_pt import window_size
from hafnium.errors import ConvergenceError, InputError
from hafnium.molecule import Molecule
from hafnium.mp import mp2_from_df_tensor
from hafnium.orbitals import FittedOrbitalTensors, OrbitalSpaces, Space
from hafnium.scf import (
    DensityFittedCoulombExchange,
    Hamiltonian,
    ScfResult,
    solve_scf,
)
from hafnium.structured import BlockVector
from hafnium.symmetry import KPoints, SymmetrySpec, detect_point_group
from tests.densities import electron_count, natural_occupations
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
def restricted_ccsd(water_hamiltonian: Hamiltonian) -> CcsdPtResult:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name("RHF"), options=TIGHT
    )
    return ccsd_pt_from_df_tensor(
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


def dense_triples(
    reference: ScfResult, df_tensor: np.ndarray, result: CcsdResult
) -> float:
    """The (T) correction over dense spin-orbital arrays.

    ``W = P(i/jk) P(a/bc) [Σ_e t_jk^ae <ei||bc> − Σ_m t_im^bc <ma||jk>]``
    and ``V = P(i/jk) P(a/bc) t_i^a <jk||bc>``, the brackets antisymmetric
    in the pairs the permutations leave alone, and
    ``E(T) = 1/36 Σ conj(W + V) W / D``.
    """
    _, v = spin_orbital_integrals(reference, df_tensor)
    occupied = reference.mo_occupations.to_dense() > 0.5
    o, u = np.flatnonzero(occupied), np.flatnonzero(~occupied)
    t1, t2 = result.singles.to_dense(), result.doubles.to_dense()
    base = np.einsum("jkae,eibc->ijkabc", t2, v[np.ix_(u, o, u, u)]) - np.einsum(
        "imbc,majk->ijkabc", t2, v[np.ix_(o, u, o, o)]
    )

    def antisymmetrized(x: np.ndarray) -> np.ndarray:
        x = x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(2, 1, 0, 3, 4, 5)
        return x - x.transpose(0, 1, 2, 4, 3, 5) - x.transpose(0, 1, 2, 5, 4, 3)

    connected = antisymmetrized(base)
    disconnected = antisymmetrized(
        np.einsum("ia,jkbc->ijkabc", t1, v[np.ix_(o, o, u, u)])
    )
    energies = reference.mo_energies.to_dense()
    e_occ, e_vir = energies[o], energies[u]
    denominators = (
        e_occ[:, None, None, None, None, None]
        + e_occ[None, :, None, None, None, None]
        + e_occ[None, None, :, None, None, None]
        - e_vir[None, None, None, :, None, None]
        - e_vir[None, None, None, None, :, None]
        - e_vir[None, None, None, None, None, :]
    )
    total = np.einsum(
        "ijkabc,ijkabc",
        np.conj(connected + disconnected),
        connected / denominators,
    )
    return float(np.real(total)) / 36


def two_electron_fci(
    reference: ScfResult, hamiltonian: Hamiltonian, df_tensor: np.ndarray
) -> tuple[list[tuple[int, int]], np.ndarray, int]:
    """The two-electron Hamiltonian over the determinants ``a†_p a†_q |0>``.

    Returns the pairs ``p < q`` labelling the determinants, the matrix of
    the Hamiltonian over them by the Slater–Condon rules, and the position
    of the reference determinant.
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
    return pairs, matrix, pairs.index((occupied[0], occupied[1]))


def two_electron_fci_correlation(
    reference: ScfResult, hamiltonian: Hamiltonian, df_tensor: np.ndarray
) -> float:
    """The exact correlation energy of two electrons in the fitted integrals.

    The lowest eigenvalue of :func:`two_electron_fci` minus the energy of
    the reference determinant.
    """
    _, matrix, determinant = two_electron_fci(reference, hamiltonian, df_tensor)
    lowest = float(np.linalg.eigvalsh(matrix)[0])
    return lowest - float(matrix[determinant, determinant].real)


def two_electron_fci_density(
    reference: ScfResult, hamiltonian: Hamiltonian, df_tensor: np.ndarray
) -> np.ndarray:
    """The exact density ``D[u, v] = <a†_v a_u>`` of two electrons over spin orbitals.

    With ``Ψ = Σ_{p<q} c_pq a†_p a†_q |0>`` and ``C`` the antisymmetric
    matrix of the ``c_pq``, ``a_s Ψ = Σ_q C[s, q] a†_q |0>``, so the
    density in the orbital basis is ``C C†``.
    """
    pairs, matrix, _ = two_electron_fci(reference, hamiltonian, df_tensor)
    _, vectors = np.linalg.eigh(matrix)
    coefficients = reference.mo_coefficients.to_dense(orbitals=True)
    n = coefficients.shape[1]
    amplitudes = np.zeros((n, n), dtype=vectors.dtype)
    for (p, q), value in zip(pairs, vectors[:, 0], strict=True):
        amplitudes[p, q] = value
        amplitudes[q, p] = -value
    orbital_basis = amplitudes @ amplitudes.conj().T
    return coefficients @ orbital_basis @ coefficients.conj().T


def shifted_energies(
    spaces: OrbitalSpaces, space: Space, index: int, shift: float
) -> OrbitalSpaces:
    """The spaces with one orbital energy of the first block shifted."""
    energies = spaces.energies(space)
    first = np.array(energies.blocks[0])
    first[index] += shift
    vector = BlockVector(energies.structure, (first, *energies.blocks[1:]))
    return OrbitalSpaces(
        spaces.occupied,
        spaces.virtual,
        vector if space is Space.OCCUPIED else spaces.occupied_energies,
        vector if space is Space.VIRTUAL else spaces.virtual_energies,
    )


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_gives_the_closed_shell_energy(
    name: str, water_hamiltonian: Hamiltonian, restricted_ccsd: CcsdPtResult
) -> None:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    df_tensor = fitted_tensor(water_hamiltonian)
    result = ccsd_pt_from_df_tensor(reference, df_tensor, options=OPTIONS)
    assert isinstance(result, CcsdResult)
    assert result.converged
    assert result.correlation_energy == pytest.approx(
        restricted_ccsd.correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert result.triples_correction == pytest.approx(
        restricted_ccsd.triples_correction, abs=CCSD_ENERGY_TOL
    )
    assert result.triples_correction < 0.0
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
        reference.energy + result.correlation_energy + result.triples_correction
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
    expected = ccsd_pt_from_df_tensor(unrestricted, df_tensor, options=OPTIONS)
    reference = solve_scf(
        cation_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    result = ccsd_pt_from_df_tensor(reference, df_tensor, options=OPTIONS)
    assert result.correlation_energy == pytest.approx(
        expected.correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert result.correlation_energy < 0.0
    assert result.triples_correction == pytest.approx(
        expected.triples_correction, abs=CCSD_ENERGY_TOL
    )


@pytest.mark.parametrize("name", ["RHF", "UHF", "complex GHF"])
def test_two_electron_ccsd_is_exact(name: str) -> None:
    """For two electrons ``e^T`` spans the full space: CCSD equals FCI."""
    molecule = hydrogen()
    _, hamiltonian = hamiltonian_for(molecule, "cc-pvdz", AUXILIARY)
    reference = solve_scf(hamiltonian, SymmetrySpec.from_name(name), options=TIGHT)
    df_tensor = fitted_tensor(hamiltonian)
    result = ccsd_pt_from_df_tensor(reference, df_tensor, options=OPTIONS)
    assert result.correlation_energy == pytest.approx(
        two_electron_fci_correlation(reference, hamiltonian, df_tensor),
        abs=CCSD_ENERGY_TOL,
    )
    # No triple excitation exists for two electrons.
    assert result.triples_correction == pytest.approx(0.0, abs=CCSD_ENERGY_TOL)


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
    expected = df_ccsd_pt(
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
    result = df_ccsd_pt(ENGINE, basis, oriented, auxiliary, reference, options=OPTIONS)
    assert result.correlation_energy == pytest.approx(
        expected.correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert result.triples_correction == pytest.approx(
        expected.triples_correction, abs=CCSD_ENERGY_TOL
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
    for letters in ("oovv", "ovvv", "vvoo", "ooov", "vvvv", "ovov"):
        expected = antisymmetrized
        for axis, letter in enumerate(letters):
            expected = np.compress(selectors[letter], expected, axis=axis)
        p, q, r, s = (
            Space.OCCUPIED if letter == "o" else Space.VIRTUAL for letter in letters
        )
        block = fitted.antisymmetrized((p, q, r, s))
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


def test_fock_blocks_are_diagonal_in_the_canonical_orbitals(
    water_hamiltonian: Hamiltonian,
) -> None:
    reference = solve_scf(water_hamiltonian, SymmetrySpec.from_name("UHF"))
    spaces = OrbitalSpaces.from_scf(reference)
    for space, energies in (
        (Space.OCCUPIED, spaces.occupied_energies),
        (Space.VIRTUAL, spaces.virtual_energies),
    ):
        block = spaces.fock_block((space, space))
        assert block.structure.keys == ((0, 0), (1, 1))
        for position in range(2):
            np.testing.assert_allclose(
                block.block((position, position)),
                np.diag(energies.blocks[position]),
                atol=STRUCTURE_TOL,
            )
    assert np.all(spaces.fock_block((Space.OCCUPIED, Space.VIRTUAL)).to_dense() == 0.0)
    assert np.all(spaces.fock_block((Space.VIRTUAL, Space.OCCUPIED)).to_dense() == 0.0)


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


@pytest.mark.parametrize("name", ["GHF", "complex UHF"])
def test_triples_match_the_dense_formula_over_any_window(
    name: str, water_hamiltonian: Hamiltonian
) -> None:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    df_tensor = fitted_tensor(water_hamiltonian)
    result = ccsd_from_df_tensor(reference, df_tensor, options=OPTIONS)
    assert not hasattr(result, "triples_correction")
    expected = dense_triples(reference, df_tensor, result)
    fitted = FittedOrbitalTensors.from_df_tensor(
        df_tensor, OrbitalSpaces.from_scf(reference)
    )
    # The CCSD(T) result carries the CCSD fields and the correction.
    with_triples = solve_ccsd_pt(
        reference, fitted, OPTIONS, guess=(result.singles, result.doubles)
    )
    assert with_triples.n_iterations == 1
    assert with_triples.correlation_energy == pytest.approx(
        result.correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert with_triples.triples_correction == pytest.approx(
        expected, abs=CCSD_ENERGY_TOL
    )
    assert with_triples.total_energy == pytest.approx(
        result.total_energy + with_triples.triples_correction
    )
    assert CcsdPtResult.from_ccsd(result, 0.5).total_energy == pytest.approx(
        result.total_energy + 0.5
    )
    occupied = result.doubles.structure.axes[0]
    for options in (
        TriplesOptions(window=1),
        TriplesOptions(window=3),
        TriplesOptions(window=max(occupied.sizes)),
        TriplesOptions(memory_limit=1),
        TriplesOptions(),
    ):
        correction = triples_correction(fitted, result.singles, result.doubles, options)
        assert correction == pytest.approx(expected, abs=CCSD_ENERGY_TOL)
    # The window grows with the memory limit, from one orbital to a sector.
    doubles = result.doubles.structure
    sizes = [window_size(doubles, 2**n) for n in range(0, 41, 4)]
    assert sizes[0] == 1
    assert sizes[-1] == max(occupied.sizes)
    assert sizes == sorted(sizes)
    for bad in ({"window": 0}, {"memory_limit": 0}):
        with pytest.raises(InputError):
            TriplesOptions(**bad)  # type: ignore[arg-type]


# ----------------------------------------------------------------- Λ equations


@pytest.fixture(scope="module")
def restricted_lambda(
    water_hamiltonian: Hamiltonian,
) -> tuple[ScfResult, CcsdLambdaResult]:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name("RHF"), options=TIGHT
    )
    return reference, ccsd_lambda_from_df_tensor(
        reference, fitted_tensor(water_hamiltonian), options=OPTIONS
    )


@pytest.mark.parametrize("name", CLASSES)
def test_every_class_gives_the_closed_shell_density(
    name: str,
    water_hamiltonian: Hamiltonian,
    restricted_lambda: tuple[ScfResult, CcsdLambdaResult],
) -> None:
    """The Λ equations and the density agree across the classes (ADR-0034)."""
    expected_reference, expected = restricted_lambda
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    result = ccsd_lambda_from_df_tensor(
        reference, fitted_tensor(water_hamiltonian), options=OPTIONS
    )
    assert isinstance(result, CcsdResult)
    assert result.converged
    assert result.lambda_converged
    assert result.correlation_energy == pytest.approx(
        expected.correlation_energy, abs=CCSD_ENERGY_TOL
    )
    assert result.total_energy == pytest.approx(
        reference.energy + result.correlation_energy
    )
    overlap = water_hamiltonian.metric(reference.structure)
    assert electron_count(result.density, overlap) == pytest.approx(10.0, abs=1e-9)
    assert natural_occupations(result.density, overlap) == pytest.approx(
        natural_occupations(
            expected.density, water_hamiltonian.metric(expected_reference.structure)
        ),
        abs=1e-9,
    )
    dense = np.asarray(result.density.to_dense())
    assert np.allclose(dense, dense.conj().T, atol=1e-12)
    assert result.lambda_singles.structure == result.singles.structure.conjugated()
    assert result.lambda_doubles.structure == result.doubles.structure.conjugated()
    assert result.n_lambda_iterations == len(result.lambda_history) > 1
    assert result.lambda_history[-1].update_norm < OPTIONS.amplitude_tolerance
    assert result.density.structure == reference.structure


@pytest.mark.parametrize("name", OPEN_SHELL_CLASSES)
def test_open_shell_classes_agree_on_the_density(
    name: str, cation_hamiltonian: Hamiltonian
) -> None:
    unrestricted = solve_scf(
        cation_hamiltonian, SymmetrySpec.from_name("UHF"), options=TIGHT
    )
    df_tensor = fitted_tensor(cation_hamiltonian)
    expected = ccsd_lambda_from_df_tensor(unrestricted, df_tensor, options=OPTIONS)
    reference = solve_scf(
        cation_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    result = ccsd_lambda_from_df_tensor(reference, df_tensor, options=OPTIONS)
    overlap = cation_hamiltonian.metric(reference.structure)
    assert electron_count(result.density, overlap) == pytest.approx(9.0, abs=1e-9)
    assert natural_occupations(result.density, overlap) == pytest.approx(
        natural_occupations(
            expected.density, cation_hamiltonian.metric(unrestricted.structure)
        ),
        abs=1e-9,
    )


@pytest.mark.parametrize("name", ["RHF", "UHF", "complex GHF"])
def test_two_electron_density_is_exact(name: str) -> None:
    """For two electrons the Λ density is the full-CI density."""
    molecule = hydrogen()
    _, hamiltonian = hamiltonian_for(molecule, "cc-pvdz", AUXILIARY)
    reference = solve_scf(hamiltonian, SymmetrySpec.from_name(name), options=TIGHT)
    df_tensor = fitted_tensor(hamiltonian)
    result = ccsd_lambda_from_df_tensor(reference, df_tensor, options=OPTIONS)
    assert result.correlation_energy == pytest.approx(
        two_electron_fci_correlation(reference, hamiltonian, df_tensor),
        abs=CCSD_ENERGY_TOL,
    )
    expected = two_electron_fci_density(reference, hamiltonian, df_tensor)
    assert np.asarray(result.density.to_dense()) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("name", ["UHF", "complex GHF"])
def test_density_is_the_derivative_of_the_energy(
    name: str, water_hamiltonian: Hamiltonian
) -> None:
    """``dE_CCSD/dε_p = D_pp − n_p``: the Λ equations make the functional stationary.

    One orbital energy of the first block is shifted and the correlation
    energy differentiated by central differences; the classes without
    derived blocks are used, so that the shift moves one spin orbital.
    """
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name(name), options=TIGHT
    )
    df_tensor = fitted_tensor(water_hamiltonian)
    spaces = OrbitalSpaces.from_scf(reference)
    fitted = FittedOrbitalTensors.from_df_tensor(df_tensor, spaces)
    result = solve_ccsd(reference, fitted, OPTIONS)
    with_lambda = solve_lambda(fitted, result, OPTIONS)
    coefficients = reference.mo_coefficients.to_dense(orbitals=True)
    overlap = water_hamiltonian.metric(reference.structure).to_dense()
    orbital_basis = (
        coefficients.conj().T
        @ overlap
        @ np.asarray(with_lambda.density.to_dense())
        @ overlap
        @ coefficients
    )
    occupations = np.asarray(reference.mo_occupations.to_dense())
    first_block = np.asarray(reference.mo_occupations.blocks[0])
    columns = {
        Space.OCCUPIED: np.flatnonzero(first_block > 0.5),
        Space.VIRTUAL: np.flatnonzero(first_block < 0.5),
    }
    step = 1e-4
    for space, index in ((Space.OCCUPIED, 1), (Space.VIRTUAL, 0)):
        energies: list[float] = []
        for shift in (step, -step):
            shifted = FittedOrbitalTensors.from_df_tensor(
                df_tensor, shifted_energies(spaces, space, index, shift)
            )
            energies.append(
                solve_ccsd(
                    reference,
                    shifted,
                    OPTIONS,
                    guess=(result.singles, result.doubles),
                ).correlation_energy
            )
        derivative = (energies[0] - energies[1]) / (2 * step)
        column = int(columns[space][index])
        assert derivative == pytest.approx(
            float(orbital_basis[column, column].real) - float(occupations[column]),
            abs=1e-6,
        )


def test_blocked_reference_gives_the_same_density() -> None:
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
    spec = SymmetrySpec.from_name("RHF")
    plain_reference = solve_scf(plain, spec, options=TIGHT)
    expected = df_ccsd_lambda(
        ENGINE, basis, oriented, auxiliary, plain_reference, options=OPTIONS
    )
    reference = solve_scf(
        blocked, spec.with_point_group(detection.group), options=TIGHT
    )
    result = df_ccsd_lambda(
        ENGINE, basis, oriented, auxiliary, reference, options=OPTIONS
    )
    assert result.density.structure == reference.structure
    assert result.density.structure.irreps is not None
    overlap = blocked.metric(reference.structure)
    assert electron_count(result.density, overlap) == pytest.approx(10.0, abs=1e-9)
    assert natural_occupations(result.density, overlap) == pytest.approx(
        natural_occupations(expected.density, plain.metric(plain_reference.structure)),
        abs=1e-9,
    )
    assert np.asarray(result.density.to_dense()) == pytest.approx(
        np.asarray(expected.density.to_dense()), abs=1e-8
    )


def test_lambda_guess_and_convergence_are_reported(
    water_hamiltonian: Hamiltonian,
) -> None:
    reference = solve_scf(
        water_hamiltonian, SymmetrySpec.from_name("RHF"), options=TIGHT
    )
    fitted = FittedOrbitalTensors.from_df_tensor(
        fitted_tensor(water_hamiltonian), OrbitalSpaces.from_scf(reference)
    )
    result = solve_ccsd(reference, fitted, OPTIONS)
    with_lambda = solve_lambda(fitted, result, OPTIONS)
    assert with_lambda.singles is result.singles
    assert with_lambda.history == result.history
    assert with_lambda.total_energy == result.total_energy
    # Restarting from the solution converges at once.
    again = solve_lambda(
        fitted,
        result,
        OPTIONS,
        guess=(with_lambda.lambda_singles, with_lambda.lambda_doubles),
    )
    assert again.n_lambda_iterations == 1
    assert np.asarray(again.density.to_dense()) == pytest.approx(
        np.asarray(with_lambda.density.to_dense()), abs=1e-10
    )
    # The amplitudes themselves have the wrong structure for a guess.
    with pytest.raises(InputError, match="structure"):
        solve_lambda(fitted, result, OPTIONS, guess=(result.singles, result.doubles))
    with pytest.raises(ConvergenceError, match="Λ iteration") as excinfo:
        solve_lambda(
            fitted,
            result,
            CcsdOptions(max_iterations=2, amplitude_tolerance=1e-12),
        )
    last = excinfo.value.last
    assert isinstance(last, CcsdLambdaResult)
    assert not last.lambda_converged
    assert last.converged
    assert last.n_lambda_iterations == 2
