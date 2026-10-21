# Changelog

All notable changes to Hafnium are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Design plan, coding conventions and the first architecture decision records.
- Periodic boundary conditions in one to three dimensions with k-point
  sampling added to the design: k-points as sectors of the structured
  matrices, time-reversal pairing of ±k, Bloch and lattice conventions
  (ADR-0007), and a revised roadmap.
- McMurchie–Davidson chosen for every integral type in the native engine
  (ADR-0008).
- Native one-electron integrals: the Boys function, Hermite expansion and
  Hermite Coulomb intermediates, the Cartesian-to-solid-harmonic
  transformation, and overlap, kinetic and nuclear attraction kernels with
  ket translations and an erf-attenuated option; the `IntegralEngine`
  protocol, `RustEngine`, `ShellTable` and molecular convenience functions;
  reference tests against libcint through PySCF with recorded data.
- p shells are ordered `x, y, z` as in libcint, PySCF and Molden
  (ADR-0009); `FunctionLabel.m` follows the storage order.
- Native two- and three-centre Coulomb integrals from one generic
  pair-pair Hermite kernel, with range separation and translations of the
  auxiliary and ket functions; `DensityFitting` with the metric, the
  three-centre integrals and the fitted tensor `B[aux, ao, ao]`; a reusable
  Hermite workspace, kernel parallelism over the leading shell index with
  rayon, and `configure_threads`; reference tests against libcint and
  PySCF's density fitting.
- Native four-centre Coulomb integrals `(uv|wx)` with range separation,
  translations of three of the four functions, Schwarz screening by
  shell-pair bounds and permutational symmetry wherever the tables and
  translations allow it; `schwarz_bounds` on the engine and `eri` for
  molecules; the three-centre kernel evaluates only `μ ≤ ν` for equal
  tables; reference tests against libcint up to g functions and for mixed
  basis sets.
- Native Fourier transforms of basis functions and of basis-function pairs
  with the kernel `exp(−iG·r)`, analytic in the Hermite basis, with ket
  translations; the Cartesian-to-spherical transform accepts complex
  values; tests against closed forms, numerical quadrature, the overlap at
  `G = 0`, the Coulomb metric assembled in reciprocal space, and PySCF's
  analytical transforms.
- Molecular Hartree–Fock: one SCF driver over structured matrices for all
  eight Fukutome classes, with Coulomb and exchange builders from the
  fitted tensor or the in-core four-centre integrals, the core-Hamiltonian
  and superposition-of-atomic-densities guesses, aufbau and averaged
  occupation policies, DIIS, energy terms and convergence history
  (ADR-0010); `ConvergenceError` carries the last state; reference tests
  against PySCF for RHF, UHF and GHF with exact and fitted integrals.
- Repository bootstrap: Python package skeleton, Rust workspace with a
  compiled kernel module built by maturin, tests, linting, type checking,
  documentation site and CI.
- Backend module: array-namespace discovery, the allowed array API subset
  with an enforcing test, Hermitian and generalized eigensolvers.
- Symmetry-structured matrices: `BlockStructure`, `BlockMatrix`, `BlockVector`,
  lossless embedding and averaging projection between structures,
  Kramers-adapted eigensolvers for paired classes, density construction.
- `SymmetrySpec` mapping the eight Fukutome classes to block structures,
  with electron-count validation.
- `Molecule` and `Atom` with XYZ parsing, electron counts and nuclear
  repulsion; CODATA 2018 unit constants.
- `Shell`, `BasisSet` and `BasisLibrary`: contraction normalization, the
  Basis Set Exchange client with a per-element cache, and bundled STO-3G,
  def2-SVP, cc-pVDZ and def2-universal-JKFIT data for H through Ar.
