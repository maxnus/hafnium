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
