# Design plan: a symmetry-aware, differentiable quantum chemistry library

Status: draft for discussion, 2026-10-21.
Package name: `hafnium`, decided 2026-10-21 (ADR-0003). Hf, as in
Hartree–Fock; the element is named after Hafnia, Copenhagen, where it was
discovered at Bohr's institute.

This document is the high-level plan agreed on before implementation starts.
It records goals, the architecture, the key design decisions with their
reasoning, the roadmap, and the decisions still open. Detailed coding rules
live in `CONVENTIONS.md`; individual decisions will be recorded as ADRs
(architecture decision records) under `docs/adr/` once the repository exists.

---

## 1. Goals and non-goals

Goals, in priority order:

1. Mean-field methods. Hartree–Fock in all eight Fukutome symmetry classes
   (real, complex and paired variants of restricted, unrestricted and
   generalized HF) and Kohn–Sham DFT (restricted and unrestricted first,
   noncollinear later), each optionally constrained by a spatial point group.
2. Density fitting (DF) as the primary route to two-electron terms, with
   exact four-centre integrals as the secondary route.
3. Correlated methods: DF-MP2, CCSD and CCSD(T), implemented once in a
   spin-orbital, symmetry-blocked formulation that serves the restricted,
   unrestricted and generalized cases.
4. A configurable array backend: NumPy (CPU, single- or multi-threaded),
   CuPy (GPU) and JAX (CPU/GPU plus automatic differentiation, AD).
5. Basis sets fetched from the Basis Set Exchange (BSE) and cached locally.
6. Code quality as a first-class requirement: static typing, immutable data,
   written conventions, and CI that enforces them.

Non-goals for version 1: periodic boundary conditions, relativistic
Hamiltonians, excited-state methods, semi-empirical methods. The architecture
must not preclude them. Periodic systems add a k-point label to the symmetry
sectors of §5, and two-component relativistic methods are complex GHF with
different integrals, so both fit later.

## 2. Guiding principles

1. Write each algorithm once, against structured abstractions. Symmetry
   lives in the data types (§5), the hardware lives in the array backend
   (§6). Method code never branches on "restricted vs unrestricted" or
   "CPU vs GPU".
2. Immutable inputs, immutable results, pure functions. No global state, no
   objects that are mutated by running a calculation. This is what makes
   JAX tracing, caching, parallel evaluation and testing straightforward,
   and it is the main design contrast to PySCF (§11).
3. A small, boring compiled core. The compiled layer evaluates integrals,
   grids and basis functions on grids. Everything that is a matrix
   contraction stays in Python and runs on whatever array backend is active.
4. Conventions are decided once, written down, and enforced by tooling
   rather than by review comments.
5. Correctness is established against independent references: PySCF and
   libcint as oracles, finite differences for derivatives, and physical
   invariants (symmetry, variational ordering) as property tests.
6. Performance by measurement. The generic path comes first; a specialized
   fast path is added only where profiling shows it matters, and it must
   pass the same tests as the generic path.

## 3. Language decision for the compiled layer

### 3.1 Options

| Criterion | Rust (PyO3 + maturin) | C++20 (nanobind) | C (cffi/ctypes) |
|---|---|---|---|
| Memory and thread safety | Enforced by the compiler | By discipline and sanitizers | By discipline |
| Tooling (format, lint, test, build) | Excellent and uniform (cargo, clippy, rustfmt) | Fragmented (CMake, clang-tidy, ...) | Fragmented |
| Python packaging | maturin builds wheels directly | scikit-build-core, more setup | Simple |
| CPU parallelism | rayon, trivially safe | OpenMP / TBB | OpenMP |
| Native GPU kernels | Immature (cudarc, rust-cuda, wgpu) | Mature (CUDA, HIP, SYCL, Kokkos) | CUDA C |
| Existing integral libraries | None native; libcint usable via FFI (REST does this) | libint2, libcint | libcint |
| Readability for new contributors | High | Medium | Medium |

### 3.2 Decision: Rust (decided 2026-10-21, ADR-0002)

Rust aligns best with the stated priority on maintainability, and the
compiled layer is deliberately small (§3.4), which limits the cost of its
weaknesses. The two weaknesses are addressed as follows.

- No native integral library. Bootstrap with libcint (C, BSD-2-Clause)
  behind the `IntegralEngine` interface so that the Python architecture can
  be built and validated immediately. Then write native Rust integrals
  (Rys quadrature or McMurchie–Davidson; decide in an ADR) and switch the
  default once they match libcint to tight tolerances. libcint remains as a
  test oracle. The Rust project REST ships `rest_libcint`, which shows this
  binding is practical.
- Weak GPU story. Dense linear algebra, which is most of the floating-point
  work in DF-HF, DFT and CC, goes through the array backend (CuPy/JAX) and
  needs no Rust GPU code. GPU-side integral and grid kernels come later and
  can be CUDA C++ compiled into a separate small crate behind the same C ABI,
  or Rust via cudarc once that is mature enough. Nothing in the Python layer
  depends on this choice.

C++20 with nanobind was the alternative considered and rejected; ADR-0002
records the reasoning and the conditions under which the choice would be
revisited (custom GPU kernels needed early, or a hard requirement on
libint2). The interface rules in §3.3 confine any such change to the
compiled crate.

### 3.3 The boundary between Python and compiled code

- The compiled layer exposes a plain C ABI: functions taking contiguous
  buffers (pointer, shape, strides, dtype) and a description of the basis
  (shell table in structure-of-arrays form). No Python objects beyond the
  binding glue.
- Arrays cross the boundary via the buffer protocol (CPU) and DLPack (GPU),
  so NumPy, CuPy and JAX arrays can all be passed without copies.
- JAX reaches the same functions through `jax.ffi` (available since JAX
  0.4.31), with custom differentiation rules registered using derivative
  integrals (§6.3).
- Thread control is explicit: Python configures the rayon thread pool and
  BLAS threads from one setting (`threadpoolctl` for BLAS) to avoid
  oversubscription.

### 3.4 What lives in the compiled layer

Version 1: one-electron integrals (overlap, kinetic, nuclear attraction,
multipoles), four-centre electron repulsion integrals with Schwarz screening,
two- and three-centre integrals for DF, first-derivative integrals of all of
these, DFT grid generation (radial schemes, Lebedev angular grids, Becke
partitioning) and basis-function values and gradients on grid batches.

Later: AO-direct J/K builders, ECP integrals, second-derivative integrals,
GPU kernels for the same functions.

## 4. Architecture overview

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ User API (Python)        Molecule, BasisSet, method specs, results   │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Methods (Python)         scf (hf, ks), mp, cc, properties, gradients │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Structured linear algebra (Python)                                   │
 │   Structure, BlockMatrix, BlockTensor, symmetry descriptors          │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Array backend (Python)   numpy | cupy | jax, via the array API std   │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Compiled kernels (Rust)  integrals, DF 3-index, grids, basis-on-grid │
 └──────────────────────────────────────────────────────────────────────┘
```

Dependency direction is strictly downward. Method code imports structured
linear algebra and the integral interface, never the backend or the kernels
directly.

Proposed repository layout:

```
hafnium/                   Python package
  api/                   user-facing entry points and configuration
  molecule/              Molecule, atoms, units, point-group detection
  basis/                 BasisSet, shells, normalization, BSE client + cache
  symmetry/              groups, irreps, SALCs, spin/time-reversal descriptors
  structured/            Structure, BlockMatrix, BlockTensor
  backend/               array namespace, device/dtype policy, linalg shims
  integrals/             IntegralEngine protocol, rust + libcint adapters, df
  grid/                  DFT grids (Python orchestration of Rust kernels)
  xc/                    functional registry, libxc binding
  scf/                   SCF driver, hf, ks, guesses, convergence, stability
  mp/                    DF-MP2
  cc/                    CCSD, (T); generated contraction code + generator
  properties/            dipoles, population analysis, gradients
  io/                    xyz, molden, cube, checkpoint (HDF5)
  _kernels/              compiled extension (built by maturin)
crates/
  hafnium-core/            Rust library: integrals, grids, basis evaluation
  hafnium-py/              PyO3 bindings (thin)
docs/                    mkdocs site: user guide, theory notes, adr/, conventions
tests/                   unit/, reference/ (vs PySCF, libcint), property/
benchmarks/              timing suite, run on demand and nightly
```

## 5. Symmetry design

This is the central design decision of the library.

### 5.1 The problem

The symmetry choices a user may want to control independently are:

- spin structure: restricted (α and β share spatial orbitals), collinear
  or "unrestricted" (separate α and β blocks, S_z conserved), generalized
  or noncollinear (α and β mix, S_z broken);
- field: real orbitals, complex orbitals, or complex orbitals with
  time-reversal (Kramers) pairing;
- spatial symmetry: none, an Abelian point group, later the full group;
- later: translations (k-points), particle number (HFB).

Implementing every combination by hand is a combinatorial explosion, which
is why PySCF has `RHF`, `UHF`, `GHF`, `ROHF`, `RKS`, `UKS`, `GKS`, each
with its own symmetry-adapted subclass and partially duplicated code.

### 5.2 Key observation: symmetry fixes an algebra, not a code path

A symmetry constraint says that the density, Fock matrix and orbitals
commute with a group G of operators on the one-particle space (unitary
spin rotations and spatial operations, antiunitary time reversal). The set
of operators that commute with G is a *-algebra, and such an algebra is
always a direct sum of full matrix algebras over the real numbers, the
complex numbers or the quaternions, each possibly tensored with an identity
of some multiplicity (the Wedderburn–Artin structure, with the field type
given by the Frobenius–Schur indicator of each irrep). In concrete terms:

> Every symmetry-constrained one-body quantity is a list of independent
> dense blocks. Each block is a matrix over R, C or H, carries a weight
> (multiplicity), and may have a partner block derived from it by a fixed
> rule (copy or complex conjugate).

Everything an SCF needs, namely addition, multiplication, adjoint, trace,
and eigendecomposition, is block-wise. So one implementation of HF, written
against a block-structured matrix type, covers every symmetry combination.
Which combination is active is data (the `BlockStructure`), not code.

The eight Fukutome classes, as tabulated in Table I of Henderson,
Jiménez-Hoyos and Scuseria (arXiv:1710.00932), the companion to the JCTC
2011 paper, map onto block structures as follows. Θ = −iσ_y K is time
reversal and K is complex conjugation. n is the number of spatial basis
functions.

| Fukutome | Common name | Preserved | Stored blocks | Ring | Partner / weight |
|---|---|---|---|---|---|
| TICS | real RHF | S², S_z, K, Θ | one n×n | R | weight 2 |
| CCW | complex RHF | S², S_z | one n×n | C | weight 2 |
| ASDW | real UHF | S_z, K | α and β, n×n each | R | weight 1 each |
| ASW | complex UHF | S_z | α and β | C | weight 1 each |
| ASCW | paired UHF | S_z, Θ | α only | C | β = conj(α); requires n_α = n_β |
| TSDW | real GHF | K | one 2n×2n | R | weight 1 |
| TSW | complex GHF | none | one 2n×2n | C | weight 1 |
| TSCW | paired GHF | Θ | one n×n quaternionic | H | stored as 2n×2n complex with Kramers structure; requires even N |

Spatial symmetry composes with this. For an Abelian point group every
irrep is one-dimensional and real, so each block above splits further into
one block per irrep in the symmetry-adapted (SALC) basis, with no change of
ring or weight. For non-Abelian groups, an irrep of dimension d contributes
a block with multiplicity d, and irreps of complex type (the E irreps of
C_3, C_4, ...) are C-blocks even for real orbitals. The same machinery
covers this, but it is out of scope for version 1 (§5.8).

### 5.3 The structured types

Names are proposals; `CONVENTIONS.md` fixes them.

`BlockStructure` describes an algebra: an ordered list of `Block` records, each
with a sector label (spin sector, irrep, later k-point), a dimension, a
ring (`REAL`, `COMPLEX`, `QUATERNION`), a weight, and an optional partner
rule (`COPY` or `CONJUGATE`) referencing another block. A `BlockStructure` also
knows the mapping from AO basis functions to sector basis functions (the
SALC transformation) and how electron counts are distributed over sectors
(the filling policy: n_α and n_β for collinear sectors, N for a generalized
sector, optional fixed per-irrep occupations).

`BlockMatrix(row_structure, column_structure, blocks)` holds one dense
backend array per stored block. It is immutable and registered as a JAX
pytree, with the structures as static metadata and the arrays as leaves.
Operations: `matmul`, `add`, `scale`, `adjoint`, `conj`, `trace`
(weighted), `eigh` and generalized `eigh` (per block), `apply_partner`
(materialize derived blocks), `to_dense()` (for tests and IO), `embed`
and `project` (next paragraph).

Symmetry lowering and raising are explicit maps between structures:
`embed(x, lower_structure)` relaxes a constraint (restricted → collinear
copies the block; collinear → generalized places blocks on the diagonal of
a 2n×2n block; real → complex casts; paired → unpaired drops the partner
rule), and `project(x, higher_structure)` is the adjoint averaging. These
two maps are what stability analysis and symmetry-breaking searches use:
embed the converged solution into a lower structure, add a perturbation,
re-converge. They are also the basis of consistency tests (§8).

Quaternionic blocks (paired GHF) are implemented in version 1 as their
complex embedding: a 2n×2n complex block with the Kramers constraint
enforced by a projector after each operation that could break it, and an
`eigh` that occupies Kramers pairs together. Dedicated quaternion storage
(half the memory) is a later optimization behind the same interface.

### 5.4 Two-electron terms

Only two places in mean-field theory depend on the spin structure:

- The Coulomb matrix depends on the total density, so J is built from
  `spin_trace(D)`, a spatial n×n matrix obtained by summing the stored
  spin blocks after materializing partners.
- Exchange acts block-wise: per spin block for collinear structures, and on
  the full 2n×2n matrix including the αβ blocks for generalized structures.

Both are expressible as operations of `BlockMatrix`, so a single
`CoulombExchangeBuilder` serves all classes. Implementations: DF (3-index
tensor `B[aux, ao, ao]`, J and K as GEMMs, GPU-friendly), and exact
four-centre integrals (stored in-core, or AO-direct later). Point-group
blocking of the integrals themselves (direct product decomposition, petite
lists) is a later optimization; version 1 builds J and K in the plain AO
basis and transforms to the SALC basis for diagonalization, which is cheap
and already yields the block-diagonal Fock matrix.

For DFT, the density on the grid is a 2×2 spin-density matrix per point
ρ_στ(r), computed generically from the structure. Restricted and collinear
functionals read ρ_α, ρ_β; noncollinear functionals read (ρ, m) and are a
later addition.

### 5.5 Correlated methods

MO-basis integrals and amplitudes are `BlockTensor`s: block-sparse tensors
whose blocks are labelled by tuples of sectors obeying a conservation rule
(Abelian fusion: product of irreps is totally symmetric, S_z is conserved).
Contraction is an `einsum`-like operation over matching blocks. This is
the approach of symmetric tensor libraries (TensorKit, Cyclops, Psi4's DPD).

Coupled-cluster equations are written once in spin-orbital form. The
collinear structure gives UCCSD (blocks αα, αβ, ββ), the generalized
structure gives GCCSD, and the restricted structure gives RCCSD via
derived blocks. The hand-optimized closed-shell spin-adapted RCCSD is
roughly two to three times faster than the spin-orbital route; it is added
later as a fast path if measured to matter, and must pass the same tests.

This is where code generation does earn its place: derive the CC, Λ and
EOM working equations with a Wick-contraction tool (pdaggerq, drudge with
gristmill, SeQuant, or a small in-house generator) and emit them as einsum
programs over `BlockTensor`. The generator and the generated code are both
committed, and the generated files are regenerated in CI to detect drift.

### 5.6 Why not code generation for symmetry

Generating N specialized implementations yields N code paths to read, test
and keep consistent. The structured-algebra approach yields one code path
and N data descriptions, and the descriptions are tested by properties
that hold for all of them (embedding consistency, invariance, variational
ordering). The generality costs some overhead in block bookkeeping, which
is negligible next to integral evaluation and GEMMs.

### 5.7 User-facing control

A calculation specifies its symmetry declaratively, for example:

```python
Symmetry(spin="collinear", field="complex", point_group="C2v")
# equivalently, by Fukutome/common name:
Symmetry.from_name("complex UHF", point_group="C2v")
```

The library maps this to a `BlockStructure`, reports the resulting Fukutome
class, and refuses inconsistent requests (paired UHF with n_α ≠ n_β).
Stability analysis takes a target structure and reports whether lowering
to it decreases the energy.

### 5.8 Scope for version 1

Rings R and C, plus H via the complex embedding for paired GHF. Abelian
point groups (D_2h and its subgroups), with detection of the full point
group for reporting and orientation. Non-Abelian blocking, quaternion
storage, and k-point sectors are later extensions that the data model
already accommodates.

## 6. Array backend and automatic differentiation

### 6.1 Backend abstraction

Method code is written against the Python array API standard, obtained via
`array_api_compat.array_namespace(x)`. NumPy 2 and CuPy 13 expose it on
their root namespaces; JAX exposes it through `jax.numpy`, with some
functions still flagged experimental, which is why a thin internal
`backend` module wraps the few operations where libraries differ (`eigh`
on generalized problems, `einsum` paths, device placement, dtype
promotion). Rules:

- `import numpy` is allowed only inside `hafnium/backend/` and `tests/`.
- The allowed subset of the array API is listed in `backend/allowed.py`
  and checked by a lint test, so that code written on NumPy is known to run
  under JAX and CuPy.
- Dtype and device are properties of the `BlockStructure`'s blocks and the
  backend configuration, never implicit.

Backends in version 1: NumPy (CPU; threads via BLAS), CuPy (GPU), JAX
(CPU/GPU, AD). PyTorch is possible later at low cost.

### 6.2 Designing for JAX from day one, shipping it later

JAX imposes constraints that are cheap to honor from the start and costly
to retrofit:

- Pure functions and immutable data: already principle 2.
- Static shapes: block shapes are static metadata of `BlockStructure`, so
  tracing works. Occupation counts per sector are static too.
- Pytrees: `BlockMatrix` and `BlockTensor` are registered pytrees.
- Iterative solvers: the SCF loop is written as a step function applied by
  a driver, so it can run as a Python loop under NumPy and as
  `lax.while_loop` under JAX. Derivatives of converged quantities use
  implicit differentiation at the fixed point (the coupled-perturbed
  equations in disguise) rather than unrolling iterations. Prior art:
  PySCFAD, DQC.
- Complex parameters: energies are real functions of complex orbitals; the
  gradient convention (which index is conjugated) is fixed in
  `CONVENTIONS.md` and tested against finite differences.

### 6.3 Non-differentiable dependencies

- Integrals: exposed to JAX through `jax.ffi`, with `custom_vjp` rules
  that call the derivative-integral kernels. Those kernels are required for
  analytic gradients anyway, so AD support costs only the registration.
- Exchange–correlation functionals: libxc supplies derivatives up to
  fourth order, so a `custom_jvp` wrapper makes libxc functionals
  differentiable to the order needed for gradients and Hessians. A small
  set of functionals may additionally be implemented natively in array-API
  code for fully traceable use.

## 7. Components

Each component lists its key types and the decisions it owns.

**molecule.** `Molecule` (frozen): atoms with symbols, nuclear charges and
coordinates in Bohr; total charge; spin given as n_α − n_β; detected point
group and a standard orientation. Constructors accept Ångström input and
convert once. Owns: units at the boundary, point-group detection.

**basis.** `Shell` (angular momentum, exponents, contraction
coefficients, spherical or Cartesian), `BasisSet` (shells per atom, AO
ordering, normalization). `BasisLibrary` fetches from the BSE REST API,
caches in the platform cache directory, and ships a small bundled set
(STO-3G, def2-SVP, cc-pVDZ and matching DF auxiliary sets) for offline
tests. Internal format: BSE's JSON schema. Owns: AO ordering and spherical
harmonic phase conventions, normalization.

**symmetry.** Point groups and irrep tables (Abelian in version 1), SALC
construction, `SpinStructure`, `FieldStructure`, `SymmetrySpec`, the
mapping to `BlockStructure` and to Fukutome names.

**structured.** `BlockStructure`, `BlockMatrix`, `BlockTensor`, embed/project
maps, structured `einsum`. Backend-agnostic.

**backend.** Array namespace selection, device and dtype policy, linear
algebra shims, thread configuration, DLPack exchange.

**integrals.** `IntegralEngine` protocol (one-electron, four-centre,
two- and three-centre, derivatives, screening data), `RustEngine`,
`LibcintEngine`, `DensityFitting` (auxiliary basis, metric, Cholesky or
eigen decomposition of the metric, `B[aux, ao, ao]`), `CoulombExchangeBuilder`
implementations (DF, in-core exact, later AO-direct).

**grid.** Radial schemes (Treutler–Ahlrichs, Mura–Knowles), Lebedev
angular grids, Becke partitioning, pruning, batching; basis values and
gradients on batches from Rust.

**xc.** Functional registry by name, libxc binding (`pylibxc` or an own
thin binding; decide in an ADR), evaluation for collinear densities with
LDA, GGA, meta-GGA, hybrid and range-separated hybrids; noncollinear later.

**scf.** One `SelfConsistentField` driver with pluggable parts: initial
guess (core Hamiltonian, superposition of atomic densities, generalized
Wolfsberg–Helmholz, from a previous result), `Hamiltonian` (one-body plus
two-body builders, plus XC for KS), occupation policy (aufbau per sector,
fixed per-irrep occupations, fractional occupation and smearing later),
convergence acceleration (DIIS, EDIIS/ADIIS, level shift, damping; a
second-order solver later), stability analysis and following. Result
objects carry energy components, orbitals, orbital energies, occupations,
the structured density, and convergence history.

**mp, cc.** DF-based MO integral transformation, DF-MP2, CCSD, (T), with
generated contraction code; later Λ equations, EOM-CC, properties.

**properties.** Energy decomposition, dipoles and multipoles, population
analysis, analytic nuclear gradients (derivative integrals plus the AD
path as a cross-check), geometry optimization via an external optimizer.

**io.** xyz, Molden, cube files, HDF5 checkpoints of result objects.

**api.** Entry points, configuration object (backend, threads, cache
directory, logging level), structured logging via the standard `logging`
module. No printing from library code.

## 8. Correctness strategy

- Reference tests compare total energies, orbital energies and gradients
  against PySCF for every Fukutome class that PySCF can produce (RHF,
  UHF, GHF, their complex variants via perturbed guesses, RKS, UKS,
  DF-MP2, CCSD, CCSD(T)). PySCF is a development dependency only, and
  these tests carry a `reference` marker.
- Integral tests compare every integral type against libcint, and every
  derivative integral against finite differences of the undifferentiated
  integral.
- Property tests (Hypothesis where useful):
  - rotational and translational invariance of energies;
  - spin-rotation invariance of GHF energies;
  - embedding consistency: a converged solution embedded into a lower
    structure is still a stationary point with the same energy and
    matching Fock matrix;
  - variational ordering: E(GHF) ≤ E(UHF) ≤ E(RHF), and complex ≤ real;
  - Hermiticity and idempotency of densities, weighted trace equals N;
  - AD gradients equal analytic gradients equal finite differences;
  - results identical across backends to a stated tolerance.
- A reference-value database (JSON with provenance: code, version,
  settings) under `tests/reference/data/` for regression testing without
  PySCF installed.
- CI: lint, type check, Rust checks, unit and property tests on every
  pull request; reference tests and benchmarks nightly; GPU and JAX
  tests nightly on a GPU runner when available.

## 9. Code quality and process

- `CONVENTIONS.md` is the single source of naming, notation, layout and
  style rules. It is reviewed like code.
- Every non-obvious decision gets an ADR: a short file with context,
  decision, consequences. The first ones: language for the compiled layer,
  integral algorithm, spherical harmonic ordering, libxc binding, license.
- Tooling (ADR-0004): `uv` for environments, `maturin` for building,
  `ruff` for formatting and linting, `pyright` in strict mode, `pytest`
  with markers, `hypothesis`, `pre-commit`, `cargo
  clippy --pedantic`, `rustfmt`, `cargo nextest`, `criterion` benchmarks,
  `mkdocs-material` with `mkdocstrings` for documentation.
- Documentation has three parts: user guide, API reference generated from
  docstrings, and theory notes that use exactly the notation of
  `CONVENTIONS.md` so that code and derivations line up.
- Trunk-based development with pull requests, conventional commit messages,
  semantic versioning, a changelog.
- License: Apache-2.0 for the whole repository (ADR-0001). Compatible
  with libcint (BSD-2-Clause), libxc (MPL-2.0), the Basis Set Exchange
  code (BSD-3-Clause) and PySCF (Apache-2.0, test use only). Contributions
  are accepted under the Developer Certificate of Origin, not a CLA. A
  NOTICE file lists bundled third-party components.

## 10. Roadmap

Milestones are ordered by dependency, not by date. Items in the same
milestone can proceed in parallel. Each milestone has an exit criterion.

**M0 Bootstrap.** Repository, `CONVENTIONS.md`, ADR template and first
ADRs, Python package skeleton, Rust crate skeleton with a trivial
PyO3 function built by maturin, CI running lint, type check and tests,
documentation site building, PEP 541 request for the PyPI name filed.
Exit: a wheel builds on Linux and macOS in CI. Status: skeleton, checks
and CI configuration done 2026-10-21; the CI run and the PEP 541 filing
wait for the GitHub remote.

**M1 Foundations.** `Molecule`, `BasisSet` with BSE client and cache,
backend module with the NumPy backend, `BlockStructure` and `BlockMatrix` with
the full operation set and embed/project maps, `SymmetrySpec` for all
Fukutome classes without point groups. Exit: `BlockMatrix` property tests
pass for all eight structures.

**M2 Integrals.** `IntegralEngine` protocol, `LibcintEngine`, DF metric
and 3-index tensor, Schwarz screening. Native Rust one-electron integrals,
then ERIs, then DF integrals, each validated against libcint. Exit: Rust
engine matches libcint to 1e-12 on a test suite of molecules and basis
sets up to f functions.

**M3 Hartree–Fock.** SCF driver, guesses, DIIS, DF and in-core exact J/K,
all eight Fukutome classes from one implementation, PySCF reference tests.
Exit: energies match PySCF for RHF/UHF/GHF and complex variants; paired
classes satisfy embedding consistency tests.

**M4 Point-group symmetry.** Detection and orientation, Abelian irreps,
SALCs, blocked SCF, per-irrep occupations. Exit: symmetry-constrained
results equal unconstrained results for symmetric molecules; irreps of
orbitals reported correctly against PySCF.

**M5 DFT.** Grids and basis-on-grid in Rust, libxc binding, RKS and UKS
with LDA/GGA/meta-GGA/hybrid/range-separated functionals. Exit: matches
PySCF to grid-limited precision on a functional matrix.

**M6 Correlated methods.** DF MO transformation, DF-MP2, `BlockTensor`
with structured einsum, equation generator, CCSD, (T). Exit: matches PySCF
UCCSD/GCCSD and RCCSD energies.

**M7 JAX backend and AD.** Pytree registration, `jax.ffi` bindings with
`custom_vjp` from derivative integrals, libxc `custom_jvp`, implicit
differentiation of the SCF, AD nuclear gradients. Exit: AD gradients match
analytic gradients and finite differences.

**M8 GPU.** CuPy backend validation, DF-HF and DFT on GPU, GPU
basis-on-grid kernel, thread and memory policies. Exit: GPU results equal
CPU results; benchmark suite reports speedups.

**M9 Properties and tools.** Analytic gradients, stability analysis and
symmetry-breaking workflows following the JCTC 2011 paper (dissociation
curves across Fukutome classes as a worked example and regression test),
Molden/cube output, checkpoints.

Later: AO-direct J/K, non-Abelian point groups, noncollinear DFT,
quaternion storage, EOM-CC, ECPs, periodic systems.

## 11. Relationship to PySCF

PySCF is the reference for correctness and a source of algorithms, not a
design template. Where this plan differs deliberately:

| PySCF | This library |
|---|---|
| Mutable `Mole` and method objects; `kernel()` mutates state | Frozen specs and results; functions return new objects |
| One class per symmetry case (RHF, UHF, GHF, ROHF, ...) | One SCF over structured matrices; symmetry is data |
| NumPy only; GPU via a separate fork (gpu4pyscf); AD via a separate project (PySCFAD) | Backend-agnostic from the start |
| Units and conventions implicit in call signatures | Stated in `CONVENTIONS.md` and docstrings |
| Loose typing | Strict typing |

PySCF is never a runtime dependency.

## 12. Open decisions needing your input

1. Integral algorithm for the native engine (Rys quadrature or
   McMurchie–Davidson; Rys is better for high angular momentum ERIs,
   McMurchie–Davidson is simpler and good for one-electron and DF terms;
   both can coexist).
2. libxc binding: `pylibxc` versus an own binding through the Rust crate.
3. Whether JAX is a development target from M1 (recommended: honor its
   constraints from M1, ship the backend in M7).

Decided so far: Apache-2.0 license (2026-10-21, ADR-0001); Rust for the
compiled layer (2026-10-21, ADR-0002); the package name `hafnium`
(2026-10-21, ADR-0003), with the PyPI name still to be obtained, see
ADR-0003 and the risk table; Python 3.11+, NumPy 2, uv, ruff and pyright
strict as the toolchain baseline (2026-10-21, ADR-0004); spherical
functions ordered m = −l..+l with BSE normalization conventions and CODATA
2018 units (2026-10-21, ADR-0006).

## 13. Risks

| Risk | Mitigation |
|---|---|
| The native integral engine is the largest single effort | libcint adapter first; native engine replaces it incrementally behind one interface |
| Structured generality grows into a research project | Version-1 scope in §5.8; generality lives in metadata and tests, not in extra code paths |
| Generic spin-orbital CC is slower than spin-adapted RCCSD | Accept for version 1; add a fast path only when measured |
| JAX constraints conflict with iterative algorithms | Pure step functions and implicit differentiation from the start |
| Array API support in JAX still partly experimental | All backend-specific code isolated in `backend/`, with a lint test for the allowed subset |
| Thread oversubscription between rayon and BLAS | Single thread setting applied to both |
| The PyPI name `hafnium` is held by an empty placeholder package | Contact the owner and file a PEP 541 request before M0; publishing is far off, so wait, with `hafnio` as the fallback if the name is not obtained by the first public release |

## References

- C. A. Jiménez-Hoyos, T. M. Henderson, G. E. Scuseria, J. Chem. Theory
  Comput. 7, 2667 (2011): generalized HF and molecular dissociation.
- T. M. Henderson, C. A. Jiménez-Hoyos, G. E. Scuseria, arXiv:1710.00932,
  Table I: the Fukutome classification with orbital-coefficient structures.
- H. Fukutome, Int. J. Quantum Chem. 20, 955 (1981): original classification.
- libcint: https://github.com/sunqm/libcint (BSD-2-Clause).
- Basis Set Exchange: https://www.basissetexchange.org and the
  `basis_set_exchange` package.
- JAX FFI: https://docs.jax.dev/en/latest/ffi.html
- Python array API standard and `array-api-compat`.
- REST (Rust electronic structure toolkit) and its `rest_libcint` crate as
  prior art for Rust in quantum chemistry.
