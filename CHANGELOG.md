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
- Periodic Hartree–Fock (M5, part two): density fitting with momentum
  transfer, `density_fitting_k` and `PeriodicDensityFitting` with the
  metric per transfer and the tensor `[kpt, kpt, aux, ao, ao]`, fitted
  integrals `eri` per k-point quadruple, `KPoints.momentum_transfers`,
  `integer_coordinates` and `function_absolute_integral` (ADR-0019); the Bloch
  three-centre kernel `coulomb_three_center_bloch` with `BlochImages`,
  `CoulombKernel` and a screening threshold on every Coulomb kernel and
  the Schwarz bounds.
- Periodic Hartree–Fock (M5, part three): k-point Coulomb and exchange
  matrices from the fitted tensor with the exchange correction
  (`KPointDensityFittedCoulombExchange`), `Hamiltonian.from_cell`,
  `BlockMatrix.from_spatial`, occupations and spin expectation values
  across k-points (per cell; the reference tests map to PySCF's supercell
  convention), the k-point set on `SymmetrySpec` (`with_kpoints`), and
  `hartree_fock` for cells; `canonical_orthogonalizer` per k-point in its
  own module, built over the storage ring of the block (ADR-0020).
- Periodic Hartree–Fock (M5, part four): the reciprocal-space kernels of
  slabs and chains by quadrature over the perpendicular components of the
  three-dimensional transform with the finite part of ADR-0012 on the
  `G_∥ = 0` line, `RangeSeparation.reciprocal_sum` with `ReciprocalSum`
  and `PerpendicularProfile` replacing `reciprocal_vectors` and
  `long_range_kernel`, `Lattice.perpendicular_vectors`, the dimension of
  the bound on `reciprocal_cutoff`, three lattice coordinates in every
  dimension from `integer_coordinates`; the fitting metric signed along
  the charge of the auxiliary functions (`signed_inverse_square_root`,
  `PeriodicDensityFitting.signature`, the `signature` of
  `KPointDensityFittedCoulombExchange`); the nuclear attraction, the
  density fitting and the Hartree–Fock classes of cells in one to three
  dimensions, with a slab KRHF reference against PySCF (ADR-0021).
- Spatial symmetry (M6, part one): the Abelian point groups as sign
  operations and parity irreps (`Operation`, `PointGroup`, `Irrep`,
  `solid_harmonic_parity`), detection of the full point group of a
  molecule with the standard frame of its Abelian subgroup
  (`detect_point_group`, `PointGroupDetection.orient`), and the
  signed-permutation representation of the group on a basis with its
  SALCs, irrep blocks of operators, orbital irrep weights and labels, and
  the symmetrization of degenerate orbitals (`SymmetryAdaptation`);
  `SymmetryError`; tests against PySCF for twenty molecules (ADR-0022).
- Spatial symmetry (M6, part two): the irrep sectors of an Abelian point
  group as an axis of the structured matrices (`IrrepSectors`,
  `Block.irrep`, `BlockStructure.irreps`, `with_irreps`,
  `sector_structure`, `BlockMatrix.sectors`, `from_sectors`,
  `to_dense(orbitals=True)`), the SCF blocked over the sectors with the
  Coulomb and exchange matrices built in the atomic basis and the
  one-electron matrices projected onto the sectors, occupations shared
  over the sectors of a spin block or fixed per irrep
  (`AufbauOccupation.irrep_electrons`), `SymmetrySpec.point_group`,
  `Hamiltonian.from_molecule(point_group=)`, `Hamiltonian.structure`,
  `ScfResult.electrons_per_irrep`; all eight classes reproduce the
  unconstrained results for symmetric molecules and energies, orbital
  energies, labels and electrons per irrep match PySCF (ADR-0023).
- Spatial symmetry (M6, part three): space groups of crystals through
  spglib, a new dependency (`SpaceGroup`,
  `detect_space_group(cell, basis=)`, `SpaceGroup.restricted_to`), chains
  and slabs padded along their open directions, the Wigner matrices
  of real solid harmonics from their polynomial table
  (`solid_harmonic_rotation`, `basis_rotation`,
  `hafnium.basis.solid_harmonic_coefficients`), the action of the group
  on the Bloch basis with lattice phases (`SpaceGroupAdaptation`), the
  point operations as an attribute of the k axis (`KPointOperations`,
  `KAxis.operations`, `KAxis.derivations`) so that every class stores
  the irreducible wedge and derives the rotated points, projection onto
  the wedge as the group average (`BlockMatrix.from_spatial(check=)`),
  `SymmetrySpec.with_space_group`, `Hamiltonian.from_cell(space_group=)`;
  all eight classes reproduce the full-zone results for a non-symmorphic
  crystal, a hexagonal slab and a chain, and energies, wedge sizes,
  weights and orbital energies match PySCF's symmetric KRHF (ADR-0024).
- Lattice sums choose their images by the exponents of the functions
  (`pair_reach`, `coulomb_reach`, `periodic_amplitude`; `reaching_images`
  takes a reach per pair of atoms), with the tail of the neglected images
  bounded by the precision, and the Bloch three-centre kernel tabulates
  its screening once per step, accumulates per Born–von Kármán class of
  the images with the k-points (`BlochImages.classes`,
  `BlochImages.kpoints` replace the phase tables) and evaluates every
  pair of images once; the fitted tensor of a silicon cell builds 6.7×
  faster (ADR-0025).
- Molecular Kohn–Sham DFT (M7, part one): quadrature grids from the
  Treutler–Ahlrichs radial mapping, the Lebedev–Laikov rules, NWChem
  pruning and Becke's partition with Treutler's adjustment, sized by
  PySCF's level tables so that they coincide with PySCF's grids
  (`hafnium.grid`: `GridSpec`, `MolecularGrid`, `build_grid`); basis
  functions with first and second derivatives on grid points from the
  compiled layer (`IntegralEngine.basis_values`); the functionals of
  libxc through the pure-Rust `xcx` crate, a new dependency of the
  kernel crate, as linear combinations with their exact exchange
  (`hafnium.xc`: `Functional`, `evaluate_functional`, the named
  functionals from SVWN to ωB97M-V); the exchange–correlation term of the
  Hamiltonian for the restricted and unrestricted classes, real and
  complex, with point-group sectors (`ExchangeCorrelation`,
  `Hamiltonian.exchange_correlation`, `long_range_two_body`,
  `EnergyTerms.exchange_correlation`), global and range-separated
  hybrids through the existing exact and fitted builders with the
  long-range kernel (`coulomb_matrix` and `exchange_matrix` split the
  former pair), `kohn_sham` with `KohnShamOptions`; RKS and UKS energies
  match PySCF to 1e-10 Hartree for LDA, GGA, meta-GGA, hybrid and
  range-separated functionals (ADR-0026). The limit on function
  arguments rises from six to seven.
- Periodic Kohn–Sham DFT (M7, part two): a cell's grid partitions space
  among the atoms and their lattice images with the compactly supported
  step of Stratmann, Scuseria and Frisch, which makes the periodic
  partition of unity local, takes a larger Lebedev rule on the shells
  shared with the neighbours and drops the points of zero weight
  (`build_grid` of a `Cell`, `build_cell_grid`, `GridSpec.partition`,
  `partition_radius` and `shared_angular_points`, `partition_weights`
  from the compiled layer, `becke_partition` with the `scheme` of the
  step); Bloch sums of the basis functions and their derivatives on grid
  points from the compiled layer (`IntegralEngine.bloch_basis_values`);
  the exchange–correlation term over the full k set, reduced onto the
  stored points as the Coulomb matrix is (`ExchangeCorrelation.from_cell`,
  `Hamiltonian.from_cell` with `exchange_correlation`, `kohn_sham` of a
  cell with the k-points and auxiliary basis of `hartree_fock`); periodic
  RKS and UKS energies match PySCF to 1e-12 Hartree on one and the same
  grid for LDA, GGA, meta-GGA and hybrid functionals (ADR-0027).
  Range-separated hybrids serve molecules only. `reaching_images` takes
  its candidate translations in chunks, so that its memory stays bounded
  for the many points of a grid. `ExchangeCorrelation.from_molecule` and
  `from_cell` take a prebuilt grid and no longer a batch size.
- The Bloch three-centre kernel accumulates real integrals per class of
  the image and class of the step and applies both phase transforms once
  at the end, auxiliary shell by auxiliary shell, with the conjugate
  completion in place (`BlochImages.step_classes`); pure functionals fit
  the Coulomb term alone from the blocks of zero momentum transfer
  (`coulomb_fitting_k`, `PeriodicCoulombFitting`,
  `KPointDensityFittedCoulomb`, chosen by `kohn_sham` for a cell when the
  functional has no exact exchange); the reciprocal-space part is added
  into the kernel's tensor in place (ADR-0028). The silicon 3×3×3 LDA
  build falls from 78 s to 9.7 s, the full tensor to
  38 s.
- Kohn–Sham for the generalized classes (M7, part three): the
  multi-collinear form of a collinear functional (ADR-0029), evaluated in
  the compiled layer with the closed form `f(ρ, |m|)` for an LDA and, for
  GGA and meta-GGA, a quadrature over spin directions with the second
  derivatives of `xcx` on product rules aligned with the magnetization of
  each point and refined to a precision, or on one Lebedev rule as PySCF
  does; the locally collinear branch with its exact derivatives for points
  polarized beyond a threshold, and the unpolarized evaluation where the
  spin variables vanish (`evaluate_noncollinear`, `NoncollinearDensity`,
  `NoncollinearOptions`, `KohnShamOptions.noncollinear`); the
  exchange–correlation term reads a spinor density, forms the density and
  magnetization with their gradients and kinetic energy densities on the
  grid and assembles the potential `v_ρ + v_m·σ`, for molecules and cells.
  The kernel reproduces PySCF's `mcfun` to 1e-14 on the same rule, the
  term PySCF's energy and potential of a noncollinear density, the
  generalized classes reproduce RKS and UKS to 1e-13 in the collinear
  limit, and `mcfun` joins the development dependencies as a test oracle.
- Block tensors and DF-MP2 (M8, part one): `BlockTensor` over the blocks
  of a `BlockStructure`, each axis with a size per sector and a
  conjugation flag, storing the blocks whose spin charges cancel and
  whose irrep characters multiply to the totally symmetric irrep, with
  the blocks of the partner sectors derived by the simultaneous partner
  map, the dense form, elementwise algebra, permutation and a
  two-operand `contract` in einsum notation (ADR-0030); `IrrepSectors`
  carries the characters of its irreps, and a spinor block of a
  `BlockMatrix` needs an even number of rows only, so orbital
  coefficients with an odd column count are valid. The `orbitals`
  package splits a converged state into its orbital spaces
  (`OrbitalSpaces`), brings the fitted tensor to the orbital basis with
  the auxiliary basis symmetry-adapted (`transform_df_tensor`) and forms
  the orbital energy denominators (`energy_denominators`); the `mp`
  package evaluates DF-MP2 in spin-orbital form for every Fukutome class
  with point-group sectors (`df_mp2`, `mp2_from_df_tensor`, `Mp2Result`
  with the amplitudes), matching PySCF's density-fitted RMP2, UMP2 and
  GMP2 to 1e-11 Hartree.
- Block tensors and DF-MP2 over k-points (M8, part two, ADR-0031): the
  sectors of a tensor axis over a k axis are the blocks at every point of
  the full set, `KAxis` carries the exact momenta of a uniform mesh
  (`KPoints.momenta`), the crystal momentum joins the conserved charges,
  the symmetries of the state (`Symmetry.SPIN_FLIP`,
  `Symmetry.TIME_REVERSAL`) replace the single partner rule as the
  generators of derived blocks with one block stored per orbit
  (`TensorStructure.symmetries`, `orbit`, `TensorAxis.image`), and the
  auxiliary axis enters conjugated. `OrbitalSpaces` serves periodic
  references with the same occupied bands at every point;
  `transform_periodic_fitting` brings the periodic fitting to the Bloch
  orbital basis over the momentum transfers in the supercell
  normalization, with the signature of the metric on request
  (`transfer_axis`); `mp2_from_periodic_fitting` and `df_mp2` of a `Cell`
  give the DF-MP2 energy per cell from the molecular code for all eight
  classes, folding onto the supercell at Γ and matching PySCF's KMP2 to
  1e-14 Hartree. The block contraction of `contract` lets `einsum`
  optimize its path, so the products of blocks run through BLAS: the
  DF-MP2 step of benzene in def2-SVP takes 0.11 s for RHF and 0.44 s for
  GHF on 16 cores, against 0.02 s and 0.54 s in PySCF.
- The equation generator and CCSD (M8, part three, ADR-0032):
  `hafnium.cc.generator` derives coupled-cluster equations by Wick's
  theorem over the Fermi vacuum, brings the terms to a canonical form
  over the antisymmetric slots of the integrals and amplitudes, groups
  the doubles into orbits of the index permutations and emits a program
  of pairwise contractions over block tensors; `hafnium.cc._ccsd_equations`
  is that program for the spin-orbital CCSD energy and residuals,
  committed and regenerated by a test; its terms are `Term` objects of
  `Factor` kinds (`Kind`) over `Index` objects of a `Space`, the
  projections `Projection` objects deriving an `Equation` each.
  `FittedOrbitalTensors` keeps the fitted tensor between every pair of
  orbital spaces and assembles the antisymmetrized integrals `<pq||rs>`
  of any block, `OrbitalSpaces.fock_block` is the diagonal Fock operator
  of the canonical orbitals and `OrbitalSpaces.denominators` replaces
  `energy_denominators`, the blocks of the Hamiltonian are keyed by
  tuples of `Space`, `NormalOrderedHamiltonian.from_fitted` forms the
  blocks an equation reads, and MP2 runs on the same class (`second_order` replaces
  `antisymmetrized_integrals`).
  `df_ccsd`, `ccsd_from_df_tensor`, `ccsd_from_periodic_fitting` and
  `solve_ccsd` iterate the equations with the DIIS of `AmplitudeDiis`,
  which shares the solver of the SCF's DIIS, for every Fukutome class,
  with point-group sectors and over k-points; `CcsdOptions` and
  `CcsdResult` with the iteration history. The energies agree across the
  classes to 1e-11 Hartree, equal FCI for two electrons, and match PySCF's
  DF-CCSD (RHF, UHF, GHF) and `KRCCSD` with `keep_exxdiv` to 3e-11
  Hartree; PySCF's k-point CCSD drops the Ewald exchange correction from
  the Fock matrix by default, which its `KUCCSD` and `KGCCSD` cannot
  keep, so those two are compared with the correction switched off on
  both sides, where all three k-point classes agree with PySCF to 1e-10. The ring of `OrbitalSpaces` is complex over a mesh
  with points that time reversal does not leave invariant. A block
  contraction is one `tensordot` per pair of blocks, with the operands
  ordered to spare the permutation of the product, and the keys, stored
  keys and derivations of a tensor structure are memoized over its axes,
  which halves the time of the CCSD iteration over a 3×3×1 mesh; the
  CCSD step takes 0.2–0.5 s for water in def2-SVP in every class (PySCF
  0.4–1.2 s), 63 s for benzene in def2-SVP against PySCF's spin-adapted
  RCCSD at 18 s, and 34 s for the helium cell over 3×3×1 against KRCCSD
  at 10 s.
- The (T) correction (M8, part four, ADR-0033): the generator derives
  the connected and the disconnected triples (`Projection.triples`,
  `Projection.expectation` keeping the terms connected to the
  Hamiltonian or the others, `ccsd_pt_equations` beside `ccsd_equations`
  in `hafnium.cc.generator.equations`) and writes
  `hafnium.cc._ccsd_pt_equations`, a second committed program whose
  functions take one window of the occupied axis per occupied index and
  evaluate every image of a term over the permuted windows. A
  `TensorAxis` has windows (`starts`, `window`, `windows`,
  `related_sectors`, `is_window_of`) that restrict a set of related
  sectors to the same slice and keep the symmetries of the state, and
  `BlockTensor.window` slices a tensor along an axis. `triples_correction`
  sums `conj(W + V) W / D / 36` over the triples of windows with the
  window chosen from a memory limit (`TriplesOptions`); `df_ccsd_pt`,
  `ccsd_pt_from_df_tensor`, `ccsd_pt_from_periodic_fitting` and
  `solve_ccsd_pt` run CCSD and the correction and return a `CcsdPtResult`,
  a `CcsdResult` with the correction included in its `total_energy`
  (`ccsd_pt` spells CCSD(T) in identifiers, since `ccsd_t` reads as
  CCSDT; CONVENTIONS.md §2.2). The
  corrections agree across the eight classes to 1e-11 Hartree, vanish
  for two electrons, do not depend on the window, match a dense
  evaluation of the textbook formula, and match PySCF's CCSD(T) for RHF,
  UHF and GHF references and its k-point KRCCSD(T) with `keep_exxdiv`
  and KGCCSD(T) without the exchange correction to 1e-10 Hartree (PySCF
  has no (T) for KUCCSD, and its KGCCSD shifts the occupied orbital
  energies of the denominators by the Madelung constant, which the
  comparison undoes). The antisymmetrizer of an orbit whose stabilizer
  is the product of its occupied and its virtual parts is factored: the
  occupied images are summed and the virtual permutations applied to
  the sum once per group of orbits, so a batch of triples takes 12
  contractions instead of 36 and a third of the passes over its tensor.
  On 16 cores the (T) step of water in def2-SVP takes 0.2–0.9 s against
  PySCF's 0.03–0.9 s, and that of the helium cell over 3×1×1 0.3–1.4 s
  against PySCF's 1.2 s (KRCCSD) and 8.0 s (KGCCSD).
  `OrbitalSpaces.denominators` takes the space of
  every axis from the caller (`space_of` removed), which also fixes the
  MP2 denominators of a state with as many virtual as occupied orbitals
  (H₂ in STO-3G), where the virtual axis was read as occupied;
  `expectation` replaces `connected_expectation` with a `connected`
  flag, `cluster_products` lists cluster builders, and
  `ccsd_module_source` and `ccsd_pt_module_source` replace
  `module_source`.
- The Λ equations of CCSD and its one-particle density (M8, part five,
  ADR-0034): the generator derives the Λ equations as the linked terms
  of `<Φ| (1 + Λ) (H_N e^T)_C |Φ_μ>` (`Projection.lambda_equation`, the
  ket the adjoint of the bra, `expectation` with `linked` dropping the
  terms in which the Hamiltonian with its clusters closes on itself) and
  the blocks of the one-particle density `<Φ| (1 + Λ) ({p† q} e^T)_C |Φ>`
  (`density_block`, an `Equation` without antisymmetry), and writes
  `hafnium.cc._ccsd_lambda_equations`, whose residual functions take the
  de-excitation amplitudes `l1`, `l2` and whose `one_particle_density`
  returns the blocks by their spaces; every generated function now
  declares its variables first, since the type checker follows at most
  64 assignments of an undeclared one. `solve_lambda` iterates the
  equations, linear in Λ, from the
  conjugate amplitudes with the DIIS of the amplitude iteration and the
  tolerances of `CcsdOptions`, and forms the Hermitian part of the
  density in the atomic basis through `OrbitalSpaces.to_atomic_basis`;
  `df_ccsd_lambda`, `ccsd_lambda_from_df_tensor`,
  `ccsd_lambda_from_periodic_fitting` and `solve_ccsd_lambda` run CCSD
  and the Λ equations and return a `CcsdLambdaResult`, a `CcsdResult`
  with `lambda_singles`, `lambda_doubles` and `density`;
  `natural_orbitals` inverts `density_matrix`. The density agrees across
  the eight classes, with point-group sectors and over k-points, is the
  derivative of the correlation energy with respect to the orbital
  energies, equals the full-CI density for two electrons, folds onto the
  supercell at Γ, and its natural occupations match PySCF's after
  `solve_lambda` and `make_rdm1` for RHF, UHF and GHF references to
  1e-10 (PySCF has no Λ over k-points). The CCSD iteration measures its
  update by `AmplitudeDiis.overlap`.
- Derivative integrals and the Hartree–Fock gradient (M9, part one,
  ADR-0035): the first derivatives of the overlap, kinetic energy,
  nuclear attraction, two-centre and three-centre Coulomb integrals with
  respect to the centres of the functions and the positions of the
  charges, from the undifferentiated kernels with the angular momentum
  raised and lowered or the Hermite integrals shifted, as derivative
  methods of the engine; the analytic nuclear gradient of density-fitted
  Hartree–Fock, `hafnium.properties.nuclear_gradient`, assembled in the
  dense atomic basis at fixed density from the energy-weighted density
  `D F D` for all eight Fukutome classes with point-group sectors,
  reported by term; tested against finite differences of the integrals
  and of the energy, PySCF's `ip` integrals and its density-fitted RHF
  and UHF gradients.
- JAX as a backend (ADR-0036): `hafnium.backend.use_backend("jax")` and
  `set_backend` select the active backend through a context variable, in
  which the integrals and every array not derived from an input are
  created, with JAX switched to 64-bit arrays on first use; the setup
  around the compiled kernels works in `host_namespace()` and the
  integral helpers hand their results over with `from_host`; `BlockMatrix`,
  `BlockVector` and `BlockTensor` are registered as pytrees with their
  structure as static metadata; every method, molecular and periodic,
  gives the same energies under NumPy and JAX, which the parity tests
  check, and JAX joins the development dependencies.

