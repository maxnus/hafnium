# ADR-0010: One SCF driver over structured matrices

Status: accepted, 2026-10-21.

## Context

PLAN.md §5 commits to a single self-consistent field implementation for
all eight Fukutome classes, written against `BlockMatrix`. The first
molecular Hartree–Fock milestone (M3) had to fix how the two-electron
terms, the initial guess, convergence control and results are expressed
in that model, and which of PySCF's practices to keep.

## Decision

- **Two-electron terms.** A `CoulombExchange` implementation provides two
  spatial contractions, `J[u, v] = Σ_wx (uv|wx) D[w, x]` and
  `K[u, v] = Σ_wx (uw|xv) D[w, x]`, from its representation of the
  integrals (the fitted tensor `B[aux, ao, ao]` or the four-centre
  integrals in memory). `coulomb_exchange_matrices` assembles the
  structured `J` and `K` for any class: `J` from the real part of the
  spin trace of the density, which is exact for a Hermitian density and
  keeps `J` real in every class, and `K` block by block, applied to each
  spin sub-block of a spinor matrix. Occupied-orbital factorization of
  the fitted exchange is a later optimization behind the same interface.
- **Guesses are densities.** `solve_scf` takes a starting density in the
  structure of the requested class. The core-Hamiltonian guess places
  half the electrons in each spin with fractional occupation of the
  frontier orbitals (`AveragedOccupation`), because occupying the core
  orbitals with `n_alpha` and `n_beta` directly commits the iteration to
  whichever orbitals the core Hamiltonian orders lowest and, for the water
  cation in def2-SVP, converges to an excited state. The default guess of
  `hartree_fock` is the superposition of atomic densities: every element
  is converged as a neutral, spin-averaged, spherically averaged atom in
  its own functions. Symmetry lowering is the user's `embed` of a
  converged density, optionally perturbed; the driver never lowers a
  structure itself.
- **Occupation policies** are objects with `occupy(mo_energies)`:
  `AufbauOccupation` for integer counts and `AveragedOccupation` for
  fractional counts shared equally within degenerate groups. When a
  policy is given explicitly the electron counts are not checked against
  the class, which is what atomic calculations with half-integer counts
  need.
- **Convergence** is measured by the Frobenius norm of the orthonormalized
  commutator `X† (F D S − S D F) X`, `X = S^{-1/2}`, together with the
  energy change; the same commutator is the DIIS error vector. DIIS is an
  immutable history whose equations are solved through an
  eigendecomposition that drops the null space, as PySCF does, because
  the error vectors become linearly dependent near convergence and in
  small problems (a two-function UHF problem has a one-dimensional error
  space).
- **Results are recomputed from canonical orbitals.** After the last
  iteration the final Fock matrix is diagonalized, the density is rebuilt
  from its occupied orbitals, and the Fock matrix and energy terms of that
  density are reported, so every field of `ScfResult` is consistent.
  Non-convergence raises `ConvergenceError` carrying that result, from
  which the iteration can be restarted.
- Energies, iteration rows and DIIS coefficients move to the host as
  Python floats where logging needs them; the JAX milestone revisits this.

## Consequences

- Adding a class never touches the driver; the eight classes differ only
  in the `BlockStructure` they pass in, and the unit tests run all eight
  on one Hamiltonian.
- Kohn–Sham DFT (M7) adds an exchange–correlation term to the Fock build
  of `Hamiltonian` and a scaling of `K` for hybrids; k-points (M5) add the
  batched axis to the same operations.
- The guess machinery assumes a basis ordered by atom (ADR-0006) and
  neutral atoms; charged or strongly polarized systems may need a
  converged density of a related system as the guess.
- Linear dependence of the atomic basis is not handled: `S^{-1/2}` and
  the Cholesky factor in the eigensolver require a positive-definite
  overlap. Canonical orthogonalization with a threshold is a later
  addition.
