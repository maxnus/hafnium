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
- Relativistic corrections added to the design (PLAN §5.10, M12): the
  exact two-component Hamiltonian as a one-body operator, spin-free in
  every symmetry class and with spin–orbit coupling in the generalized
  classes, the Gaussian nuclear model as an erf-attenuated attraction,
  the speed of light as an explicit parameter, and the open decisions on
  the nuclear model, the two-electron spin–orbit correction and ECPs.
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
- General contractions are evaluated once per shell in every integral
  kernel instead of once per contraction (ADR-0011): `ShellTable` entries
  carry all contraction rows of a shell, the primitive Hermite expansions
  are shared, and the four-centre kernel scatters its blocks in parallel.
  The four-centre integrals of benzene in cc-pVDZ cost eight times less
  CPU time than before, within a factor of two of libcint, as the
  segmented def2-SVP basis already was.
- Periodic foundations (M4, part one): `Lattice` with one to three
  translation vectors, reciprocal vectors, fractional coordinates,
  wrapping, translations within a cutoff and supercells; `AtomicSystem`
  as the common base of `Molecule` and `Cell`, which are siblings
  (ADR-0013), with `BasisSet.for_molecule` renamed `for_atoms`; `Cell`
  with a lattice, extended-XYZ input and output that keep the lattice,
  supercell construction; `KPoints` meshes (Γ-centred, shifted,
  Monkhorst–Pack) and explicit lists with weights, time-reversal partners
  and invariant momenta; Ewald nuclear repulsion in one to three
  dimensions with one regularization of the `G = 0` term (ADR-0012), the
  lattice loops in Rust and the cutoffs chosen in Python; tests against
  Madelung constants, direct summation, PySCF's `energy_nuc` and
  independence of the range-separation parameter. `ShellTable.from_basis`
  takes the system and rejects a basis built for other atoms. Decided: no
  libcint adapter (ADR-0014).
- Periodic foundations (M4, part two): the k axis on the structured
  matrices (ADR-0015). `KAxis` holds the partner map and weights of a
  k-point set, `BlockStructure.with_k_axis` attaches it, and blocks
  become batches `[kpt, ao, ao]` over the stored k-points: the half zone
  for the real and paired classes, with the block at `−k` derived by
  conjugation or the Kramers map and paired UHF's
  `beta(k) = conj(alpha(−k))`, every point for the complex classes. Traces
  apply the weights, `eigh` runs per k-point with Kramers pairing at the
  invariant points, and embedding, projection and the dense form work
  across the hierarchy at k; `KPoints.axis()` and
  `SymmetrySpec.structure(kpoints)` build the structures. The property
  tests run for all eight classes over k-point sets, and the half-zone
  storage is checked against the full-zone storage.
- Periodic foundations (M4, part three): Bloch sums of the overlap and
  kinetic integrals at k-points (ADR-0016). `lattice_images` chooses the
  translations from the radii at which the basis functions fall below a
  precision, `bloch_sum` forms `Σ_T exp(i k·T) M_T` from any engine output
  with an image axis, and `overlap_k` and `kinetic_k` return the
  `[kpt, ao, ao]` matrices of a cell over its k-points. Tests: a large box
  at Γ against the molecular matrices in one to three dimensions, band
  folding of the kinetic bands between a supercell and the primitive mesh,
  Hermiticity and time reversal, independence of the precision and of the
  cutoff, and PySCF's `pbc_intor` in one to three dimensions with recorded
  data. M4 is complete.
- SCF robustness: spin expectation values `⟨S⟩` and `⟨S²⟩` of any class on
  the result, level shifting and density damping options, and canonical
  orthogonalization that removes linearly dependent basis combinations.
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
- Periodic Hartree–Fock (M5, part one): the probe-charge Ewald correction
  chosen for the exchange divergence, with `madelung_constant` of a k-point
  mesh in every dimension and `KPoints.mesh_size` (ADR-0017); the
  regularized nuclear attraction of three-dimensional cells,
  `nuclear_attraction_k` and `core_hamiltonian_k`, from the screened
  short-range kernel over nuclear images and the reciprocal-space sum of
  the pair transforms with the `G = 0` finite part of ADR-0012 (ADR-0018);
  `RangeSeparation` with the cutoffs, `reaching_images`, and the public
  `reciprocal_cutoff` and `default_omega` of the Ewald module.
- Native short-range `erfc(ωr)/r` nuclear attraction, rigorous screening
  of the nuclear attraction and the pair Fourier transform against a
  threshold, and parallel images in the one-electron kernels (ADR-0018);
  `CoulombKernel` and `PointCharges` on the engine protocol.

