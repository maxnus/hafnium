# Design plan: a symmetry-aware, differentiable quantum chemistry library

Status: living document; revised 2026-10-21 to add periodic systems (§5.9,
ADR-0007) and 2026-10-22 to add relativistic corrections (§5.10, M12).
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
   generalized HF) and Kohn–Sham DFT in the same eight classes, the
   generalized ones through the multi-collinear form of the functional,
   each optionally constrained by a spatial point group.
2. Periodic systems in one, two and three dimensions with k-point
   sampling, treated by the same mean-field and correlated code as
   molecules: translational symmetry is one more symmetry in the data
   model (§5.9).
3. Density fitting (DF) as the primary route to two-electron terms, with
   exact four-centre integrals as the secondary route.
4. Correlated methods: DF-MP2, CCSD and CCSD(T), implemented once in a
   spin-orbital, symmetry-blocked formulation that serves the restricted,
   unrestricted and generalized cases.
5. Relativistic corrections through the exact two-component (X2C)
   Hamiltonian: spin-free in every symmetry class, with spin–orbit
   coupling in the generalized classes, for molecules and cells (§5.10).
6. A configurable array backend: NumPy (CPU, single- or multi-threaded),
   CuPy (GPU) and JAX (CPU/GPU plus automatic differentiation, AD).
7. Basis sets fetched from the Basis Set Exchange (BSE) and cached locally.
8. Code quality as a first-class requirement: static typing, immutable data,
   written conventions, and CI that enforces them.

Non-goals for version 1: four-component Hamiltonians, excited-state
methods, semi-empirical methods, plane-wave basis sets and
pseudopotentials, and effective core potentials unless §12 decides
otherwise. The architecture must not preclude them; the two-component
Hamiltonian of §5.10 enters as a one-body operator and leaves the rest of
the library untouched.

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
  in the McMurchie–Davidson scheme (ADR-0008) and switch the default once
  they match libcint to tight tolerances. libcint remains as a
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
two- and three-centre integrals for DF, range-separated (erf- and
erfc-attenuated) variants of every Coulomb-type integral, the Fourier
transform of basis-function pairs and of auxiliary functions, image loops
that accumulate any of these over a given list of lattice translations
with distance screening, first-derivative integrals of all of these and,
by the same mechanism on bra and ket, the kinetic-balance integrals
⟨∂_iχ_μ|V|∂_jχ_ν⟩ of the relativistic Hamiltonian (§5.10), DFT grid
generation (radial schemes, Lebedev angular grids, Becke partitioning) and
basis-function values and gradients on grid batches.

Later: AO-direct J/K builders, ECP integrals, second-derivative integrals,
GPU kernels for the same functions.

## 4. Architecture overview

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ User API (Python)        Molecule, Cell, BasisSet, specs, results    │
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
  molecule/              Molecule, atoms, units, Lattice, Cell
  basis/                 BasisSet, shells, normalization, BSE client + cache
  symmetry/              groups, irreps, SALCs, spin/time-reversal descriptors,
                         k-point sets, space groups
  structured/            BlockStructure, BlockMatrix, BlockTensor
  backend/               array namespace, device/dtype policy, linalg shims
  integrals/             IntegralEngine protocol, RustEngine, df,
                         periodic/ (image lists, Bloch phases, Ewald, kernels)
  grid/                  DFT grids (Python orchestration of Rust kernels)
  xc/                    functional registry, libxc binding
  scf/                   SCF driver, hf, ks, guesses, convergence, stability
  orbitals/              the orbital basis of a converged state: occupied and
                         virtual spaces as tensor axes, the fitted integrals in
                         that basis; shared by mp and cc
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
- translations: Bloch k-points of a periodic system (§5.9), and for
  crystals the space group on top of the point group;
- later: particle number (HFB).

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
with a sector label (spin sector, irrep), a dimension, a
ring (`REAL`, `COMPLEX`, `QUATERNION`), a weight, and an optional partner
rule (`COPY` or `CONJUGATE`) referencing another block. For periodic
systems the k-point set is a batch axis shared by every block, not a
label on individual blocks (§5.9). A `BlockStructure` also
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
functionals read ρ_α, ρ_β; the generalized classes read the density and the
magnetization (ρ, m) with their gradients and take the multi-collinear form
of the collinear functional (ADR-0029), which recovers the collinear
functional exactly where the spin variables are parallel.

### 5.5 Correlated methods

MO-basis integrals and amplitudes are `BlockTensor`s: block-sparse tensors
whose blocks are labelled by tuples of sectors obeying a conservation rule
(Abelian fusion: product of irreps is totally symmetric, S_z is conserved,
crystal momentum is conserved modulo a reciprocal lattice vector).
Contraction is an `einsum`-like operation over matching blocks. This is
the approach of symmetric tensor libraries (TensorKit, Cyclops, Psi4's DPD).
ADR-0030 fixes the sectors as the blocks of a `BlockStructure`, the
charges as the spin and the characters of the irrep, the derived blocks
by the simultaneous partner map of all indices, and the contraction as a
pairing of conjugated with unconjugated indices.

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
group for reporting and orientation. k-point sectors for periodic systems
(§5.9), with time-reversal pairing from the start and space-group
reduction in the symmetry milestone. Non-Abelian blocking and quaternion
storage are later extensions that the data model already accommodates.

### 5.9 Periodic systems: k-points as sectors

Periodic boundary conditions in one, two or three dimensions are part of
version 1. A `Cell` is an `AtomicSystem` with a `Lattice` of d ∈ {1, 2, 3}
translation vectors, a sibling of `Molecule` rather than a subclass
(ADR-0013); the open directions of a chain or a slab carry no vacuum
padding (ADR-0007). Translational symmetry is handled like every
other symmetry: as data in the block structure, not as a separate code
path.

- **Bloch sectors.** The basis functions of a cell are the Bloch sums
  φ_μ^k(r) = Σ_T e^{ik·T} φ_μ(r − T) over lattice translations T. Every
  one-body operator is block-diagonal in k, so a k-point is a sector label
  of a `Block` next to the spin sector and the irrep. All k blocks of a
  sector have the same shape and are stored as one batched array with a
  leading k axis, which keeps `matmul`, `eigh` and `einsum` batched on
  GPUs and traceable under JAX; the k-point set with its weights is static
  metadata of the `BlockStructure`, and per-k weights, summing to one
  over the full zone, extend the block weights so that traces and
  energies are per cell. A system without a lattice is a molecule, so the
  molecular path is the special case of one k-point and no lattice sum.
- **Time reversal pairs k with −k.** Complex conjugation maps a Bloch sum
  at k to the one at −k, so the antiunitary operators a Fukutome class
  preserves (K, Θ) act on k-space as k → −k combined with their molecular
  action. The partner rules of §5.2 therefore apply along the k axis,
  composed with the map k → −k: in the real classes (TICS, ASDW, TSDW) the
  block at −k is the complex conjugate of the block at k; in paired GHF
  (TSCW) the block at −k is the Kramers map ΘXΘ⁻¹ of the block at k, the
  map already used for quaternionic blocks; in paired UHF (ASCW) the beta
  block at k is the complex conjugate of the alpha block at −k, so the
  alpha block is stored at every k and beta is derived everywhere; in the
  complex classes (CCW, ASW, TSW) every k of every block is stored. The
  real classes and paired GHF therefore store a half Brillouin zone, with
  a stored point carrying the weight of both members of its pair, and at
  time-reversal invariant momenta (k ≡ −k, for example Γ, or every point
  of an unshifted 2×2×2 mesh) the derived block coincides with the
  molecular partner rule: the block is real, or quaternionic for paired
  GHF, or beta = conj(alpha) for paired UHF. The eight classes carry over
  unchanged: real RHF with k-points is TICS with F(−k) = F(k)*, complex
  RHF (CCW) breaks that pairing and describes current-carrying states,
  paired GHF (TSCW) keeps Kramers pairs across ±k, and so on. `embed`
  materializes the derived partners, `project` averages them.
- **Translational symmetry breaking.** Density and spin waves with a
  wavevector Q ≠ 0, the literal meaning of the Fukutome names, are
  described in a supercell commensurate with Q (band folding). Explicit
  coupling between k and k + Q is not planned.
- **Space groups.** Point operations R of the crystal map k to Rk, so the
  stored set shrinks to the irreducible wedge of the Brillouin zone with
  weights, and the blocks at the other k follow by the unitary rotation of
  the AO basis: a third partner rule (`ROTATE`) added in the symmetry
  milestone, with space-group detection delegated to spglib (§12,
  ADR-0024).
- **Two-electron terms.** Crystal momentum is conserved: (μk₁ νk₂|λk₃ σk₄)
  vanishes unless k₁ − k₂ + k₃ − k₄ is a reciprocal lattice vector. The DF
  tensor is labelled by the momentum transfer q = k₂ − k₁; the Coulomb
  matrix needs only q = 0, exchange couples every pair (k, k + q). For the
  correlated methods this is an Abelian fusion rule like the irrep
  product, so `BlockTensor` handles it with the same machinery.
- **Coulomb sums.** The electron–nuclear, electron–electron and
  nuclear–nuclear lattice sums diverge separately and converge together
  for a neutral cell. Both the nuclear attraction and the DF tensor use
  the same range separation of the Coulomb kernel: the short-range part is
  a real-space lattice sum of erfc-attenuated integrals, the long-range
  part is evaluated in reciprocal space from the analytic Fourier
  transform of Gaussian pair densities (against the nuclear structure
  factor for the nuclear attraction, against the auxiliary functions for
  the DF tensor), with the G = 0 term removed by a uniform background.
  The exchange divergence needs a correction whose choice is the first
  item of M5: ADR-0007 named the probe-charge Ewald (Madelung) correction,
  which is PySCF's default and was adopted without weighing it against
  the truncated Coulomb kernel (Spencer and Alavi) and the Wigner–Seitz
  truncation (Sundararaman and Arias), which converge faster with the k
  mesh and are the construction that also gives the reduced-dimension
  kernels. Only the two-electron terms are fitted; the one-electron
  Hamiltonian carries no fitting error. The reciprocal-space kernel is 4π/G² in three
  dimensions and the analytic kernel of the reduced dimension for slabs and
  chains, which is why no vacuum padding is needed. The nuclear repulsion
  is an Ewald sum. Results must be independent of the range-separation
  parameter, which is a strong internal consistency test (§8).
- **Energies, charge and spin are per unit cell**; a charged cell is
  neutralized by the uniform background.

### 5.10 Relativistic corrections: a different one-body operator

Relativistic effects are part of version 1 through the exact two-component
Hamiltonian at the one-electron level (X2C-1e: Kutzelnigg and Liu, Iliaš
and Saue, Liu and Peng). The one-electron Dirac Hamiltonian is set up in
the uncontracted basis, decoupled exactly and renormalized, and the
resulting matrix replaces the non-relativistic core Hamiltonian T + V; the
two-electron terms stay non-relativistic. Everything downstream of the core
Hamiltonian is unchanged, which is the reason for this choice over
Douglas–Kroll–Hess and ZORA: X2C needs the same integrals, has no model
potential and no gauge dependence, and is exact for one-electron systems
in the given basis. Four-component Hartree–Fock is a non-goal (§1); the
four-component one-electron matrix that the decoupling builds is the only
four-component object in the library.

- **The decoupling.** In a basis of restricted kinetic balance the matrix
  Dirac equation reads
  [[V, T], [T, W/(4c²) − T]] C = [[S, 0], [0, T/(2c²)]] C E, with S, T
  and V the overlap, kinetic and nuclear-attraction matrices of the
  uncontracted scalar basis and the kinetic-balance matrix
  W_μν = ⟨(σ·p)χ_μ|V|(σ·p)χ_ν⟩ = Σ_i ⟨∂_iχ_μ|V|∂_iχ_ν⟩
  + i Σ_k σ_k Σ_ij ε_ijk ⟨∂_iχ_μ|V|∂_jχ_ν⟩ (CONVENTIONS §3). The
  positive-energy solutions give the decoupling X = C_S C_L⁻¹, the
  renormalization R = (S⁻¹S̃)^{−1/2} with S̃ = S + X†TX/(2c²), and the
  two-component Hamiltonian h = R†(V + TX + X†T + X†(W/(4c²) − T)X)R,
  projected onto the contracted basis with the contraction matrix, which
  is exact because contracted functions are linear combinations of the
  primitives that ADR-0011 keeps. The cost is one diagonalization of a
  matrix of twice (four times with spin–orbit coupling) the uncontracted
  size, negligible next to the SCF. A constant shift of the potential
  shifts W by twice that constant times T, hence the whole Dirac matrix
  along its metric: the decoupling is invariant under the arbitrary
  constant of a regularized periodic potential, provided W is built from
  the same regularized potential as V.
- **Spin-free and spin–orbit variants.** The spin-free Hamiltonian keeps
  the scalar part of W only; it is a real n×n matrix like T + V, so all
  eight Fukutome classes, Kohn–Sham DFT and the correlated methods use it
  unchanged. With spin–orbit coupling the Hamiltonian is a 2n×2n complex
  matrix with the quaternionic (Kramers) structure of §5.2, that is a
  one-body operator of the paired GHF class (TSCW), and symmetry as data
  does the rest: the one-body operator carries its own `BlockStructure`,
  `Hamiltonian` embeds it into the requested class, which succeeds for
  TSCW (Kramers-restricted, even electron counts) and TSW
  (Kramers-unrestricted) and raises a `StructureError` for every other
  class: the spin-separable ones, and real GHF (TSDW), since the
  spin–orbit operator is complex. Nothing in the SCF driver changes; the
  new code is the decoupling and the structured one-body operator. The one-electron spin–orbit operator overestimates
  spin–orbit splittings because the two-electron spin–same-orbit
  screening is missing; the correction (Boettger's screened-nuclear
  spin–orbit scaling, its modified form, or atomic mean-field integrals)
  is one of the decisions of the milestone's ADR (§12). Two-component
  Kohn–Sham needs the noncollinear exchange–correlation evaluation, so
  spin–orbit DFT follows noncollinear DFT; spin-free X2C-DFT needs nothing
  new, and spin–orbit MP2 and CC are the generalized case of the
  spin-orbital formulation of §5.5.
- **Locality.** The full molecular decoupling is the default for
  molecules. The atomic decoupling builds X and R from the Dirac problem of
  each atom alone (the free-atom form of the diagonal local approximation
  of Peng and Reiher) and applies them, block-diagonal over atoms, to the
  full molecular, or lattice-summed, V, T and W; it is linear in the
  number of atoms and independent of the lattice, so it is what cells use
  at every k, and it keeps nuclear gradients simple because X and R are
  then rigid under nuclear displacements. The variant that takes the
  atomic blocks from the molecular matrices instead is an option whose
  gradients carry a response term. The full
  per-k decoupling of the lattice-summed matrices (invariant under the
  Ewald constant, see above) and analytic gradients of the full molecular
  decoupling (Cheng and Gauss) are later items; the AD path of M9
  differentiates through the full decoupling by way of the positive-energy
  projector rather than the eigenvectors, since every eigenvalue of the
  spin–orbit Dirac matrix is Kramers-degenerate and the eigenvector
  derivative of `eigh` is singular there.
- **Nuclear charge model.** A Gaussian nucleus with the exponent
  ζ_A = 3/(2 r_A²), r_A = 0.836 A^{1/3} + 0.570 fm (Visscher and Dyall),
  has the potential Z_A erf(√ζ_A r)/r, which is the erf-attenuated nuclear
  attraction the engine already evaluates, with ω = √ζ_A per nucleus. The
  finite nucleus therefore costs a per-nucleus attenuation parameter in
  the kernels and nothing else, and it removes the weak singularity of the
  relativistic orbitals at a point nucleus, which is why Dyall's sets and
  ANO-RCC are optimized with it (the cc-pVXZ-DK sets are not). Point
  nuclei stay the default for non-relativistic calculations, so molecular
  results do not change; the default for relativistic ones is an open
  decision (§12).
- **Basis sets.** The decoupling runs in the fully uncontracted basis,
  with near-linear dependence among steep functions handled by the
  canonical orthogonalization of M3. Contractions optimized
  non-relativistically are unsuitable for heavy elements; the relativistic
  families on the Basis Set Exchange (x2c-SVPall and x2c-TZVPall,
  cc-pVXZ-DK, ANO-RCC, Dyall's and Sapporo's sets) are the intended
  inputs, and the bundled library gains one of them for offline tests.
  Effective core potentials are not needed for correctness on this
  route, since all-electron X2C covers the periodic table and the
  decoupling itself is cheap, but all-electron basis sets of heavy
  elements make the SCF far more expensive than an ECP valence basis;
  whether version 1 adds ECP integrals for the def2 sets beyond krypton is
  an open decision (§12).
- **Properties.** A one-electron property in the two-component picture
  needs the same transformation of its operator (picture change),
  O → R†(O_LL + O_LS X + X†O_SL + X†O_SS X)R with
  O_SS = ⟨(σ·p)χ|O|(σ·p)χ⟩/(4c²), one more derivative-type integral per
  operator. The decoupling is therefore kept as an immutable result (X, R
  and the contraction matrix) that transforms operators on request;
  picture-change errors are large for core properties and small for
  valence ones, and every property of M11 states whether it applies the
  transformation.
- **The speed of light is an explicit parameter** with the CODATA 2018
  default (CONVENTIONS §3): oracle comparisons pass the oracle's value,
  since the relativistic energy of a heavy atom moves by far more than
  `ENERGY_TOL` between CODATA revisions, and a very large value
  reproduces the non-relativistic Hamiltonian, which is the first test.

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
  tracing works. Occupation counts per sector and the k-point set are
  static too.
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
- Exchange–correlation functionals: `xcx` supplies first and second
  derivatives (ADR-0026), so a `custom_jvp` wrapper makes the functionals
  differentiable to the order needed for gradients and Hessians. A small
  set of functionals may additionally be implemented natively in array-API
  code for fully traceable use.

## 7. Components

Each component lists its key types and the decisions it owns.

**molecule.** `Molecule` (frozen): atoms with symbols, nuclear charges and
coordinates in Bohr; total charge; spin given as n_α − n_β; detected point
group and a standard orientation. Constructors accept Ångström input and
convert once. `Lattice`: one to three translation vectors, reciprocal
vectors, fractional coordinates, translations within a cutoff. `Cell`: a
sibling of `Molecule` under `AtomicSystem` with a `Lattice`, charge and
spin per cell, supercell construction, extended-XYZ input and output that
keep the lattice, and an Ewald nuclear repulsion. `NuclearModel`: point
nuclei or Gaussian nuclei with the Visscher–Dyall exponents (§5.10). Owns:
units at the boundary, the nuclear charge model.

**basis.** `Shell` (angular momentum, exponents, contraction
coefficients, spherical or Cartesian), `BasisSet` (shells per atom, AO
ordering, normalization, the uncontracted basis with the contraction
matrix for the relativistic decoupling of §5.10). `BasisLibrary` fetches
from the BSE REST API, caches in the platform cache directory, and ships a
small bundled set (STO-3G, def2-SVP, cc-pVDZ and matching DF auxiliary
sets, later one relativistic family) for offline tests. Internal format:
BSE's JSON schema; effective-core-potential data is rejected with a clear
error until §12 decides on ECPs. Owns: AO ordering and spherical harmonic
phase conventions, normalization.

**symmetry.** Point groups and irrep tables (Abelian in version 1), SALC
construction, `SpinSymmetry`, `Field`, `SymmetrySpec`, the mapping to
`BlockStructure` and to Fukutome names. k-point sets: uniform
(Γ-centred or shifted) meshes and explicit lists, time-reversal pairing
and weights, momentum conservation tables. Point-group detection of molecules and, through
spglib, space groups of crystals with the irreducible wedge.

**structured.** `BlockStructure`, `BlockMatrix`, `BlockTensor`, embed/project
maps, structured `einsum`, the batched k axis with its partner rules.
Backend-agnostic.

**backend.** Array namespace selection, device and dtype policy, linear
algebra shims, thread configuration, DLPack exchange.

**integrals.** `IntegralEngine` protocol (one-electron, four-centre,
two- and three-centre, range-separated kernels, nuclear attraction with a
per-nucleus attenuation for Gaussian nuclei, pair Fourier transforms,
derivatives including the kinetic-balance integrals of §5.10, screening
data), each over shell pairs at explicit translation vectors so that
molecules and cells share it, `RustEngine` (the only engine, ADR-0014),
`DensityFitting` (auxiliary basis, metric, Cholesky or
eigen decomposition of the metric, `B[aux, ao, ao]`), `CoulombExchangeBuilder`
implementations (DF, in-core exact, later AO-direct). `periodic/`: image
lists and real-space cutoffs, Bloch phases, Ewald sums per dimension,
reciprocal-space Coulomb kernels per dimension, range-separated DF with
momentum transfer, the Madelung exchange correction. The image loops
themselves run in the compiled layer (§3.4), which receives the image list
and the screening threshold from here; the cutoff policy lives only in
Python.

**grid.** Radial schemes (Treutler–Ahlrichs, Mura–Knowles), Lebedev
angular grids, Becke partitioning, pruning, batching; basis values and
gradients on batches from Rust. For cells, the same atom-centred grids
partitioned among the atoms and their lattice images with the local
Stratmann–Scuseria–Frisch step and a larger angular rule on the shells
shared with the neighbours, and Bloch sums of the basis functions from
Rust (ADR-0027).

**xc.** Functional registry by name, the functionals of libxc through the
pure-Rust `xcx` crate with libxc's ids and conventions (ADR-0026),
evaluation for collinear densities with LDA, GGA, meta-GGA, hybrid and
range-separated hybrids, and the multi-collinear evaluation for the
density and magnetization of the generalized classes (ADR-0029).

**scf.** One `SelfConsistentField` driver with pluggable parts: initial
guess (core Hamiltonian, superposition of atomic densities, generalized
Wolfsberg–Helmholz, from a previous result), `Hamiltonian` (a one-body
operator carrying its own block structure, so that the two-component
Hamiltonian of §5.10 embeds into the generalized classes and is rejected
by the others, plus two-body builders, plus XC for KS), occupation
policy (aufbau per sector and across k-points, fixed per-irrep
occupations, fractional occupation and smearing later),
convergence acceleration (DIIS, EDIIS/ADIIS, level shift, damping; a
second-order solver later), stability analysis and following. Result
objects carry energy components, orbitals, orbital energies, occupations,
the structured density, and convergence history.

**relativistic.** The exact two-component decoupling of §5.10: the
one-electron Dirac matrices in the uncontracted basis for either nuclear
model, the positive-energy projection, the decoupling and renormalization
matrices, the spin-free and spin–orbit Hamiltonians as one-body operators
with their block structure, the atomic (local) variant, the two-electron
spin–orbit correction chosen in the milestone's ADR, and the
picture-change transformation of property operators, all kept as an
immutable result. Owns: the speed of light as an explicit parameter.

**mp, cc.** DF-based MO integral transformation, DF-MP2, CCSD, (T), with
generated contraction code; later Λ equations, EOM-CC, properties.

**properties.** Energy decomposition, dipoles and multipoles, population
analysis, each with the picture-change transformation for two-component
results (§5.10), analytic nuclear gradients (derivative integrals plus the AD
path as a cross-check), geometry optimization via an external optimizer,
band structures along k-point paths from a converged density.

**io.** xyz, Molden, cube files, HDF5 checkpoints of result objects.

**api.** Entry points, configuration object (backend, threads, cache
directory, logging level), structured logging via the standard `logging`
module. No printing from library code.

## 8. Correctness strategy

- Reference tests compare total energies, orbital energies and gradients
  against PySCF for every Fukutome class that PySCF can produce (RHF,
  UHF, GHF, their complex variants via perturbed guesses, RKS, UKS,
  DF-MP2, CCSD, CCSD(T)). PySCF is a development dependency only, and
  these tests carry a `reference` marker. PySCF is the oracle, not the
  truth: it has bugs and conventions of its own, so a discrepancy is
  located on both sides before a tolerance is loosened or Hafnium is
  changed, the independent checks below carry equal weight, and any
  oddity found in PySCF is recorded with the test that works around it.
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
- Periodic consistency tests, required for every periodic feature:
  - a Γ-point cell with a large box converges to the molecular result as
    the box grows: to `ENERGY_TOL` for the lattice-summed overlap and
    kinetic matrices (Gaussian tails), algebraically for total energies,
    whose residual at a given box size is checked against the expected
    power law; the Ewald nuclear repulsion is checked against tabulated
    Madelung constants in three, two and one dimensions and against
    PySCF's `energy_nuc` instead, since the nuclei alone are not neutral;
  - a supercell at Γ reproduces the primitive cell with the matching
    k-point mesh (band folding), per cell;
  - storing the full Brillouin zone and storing the half zone with
    time-reversal partners give identical results;
  - results are independent of the range-separation parameter of the
    Coulomb split and of the real-space cutoffs, to a stated tolerance;
  - the three-dimensional kernels and the reduced-dimension kernels agree
    for a slab or chain in a large box, in the limit of large vacuum;
  - reference comparisons against PySCF's `pbc` module (KRHF, KUHF, KGHF,
    KRKS, KMP2, KCCSD) with the same auxiliary basis and the Ewald exchange
    correction, with the fitting-scheme residual recorded alongside.
- Relativistic consistency tests, required for every relativistic feature
  (§5.10):
  - the kinetic-balance integrals, all nine Cartesian components of
    ⟨∂_iχ_μ|V|∂_jχ_ν⟩, against libcint (`int1e_pnucp`, `int1e_pnucxp`)
    and against mixed second differences of the nuclear attraction with
    respect to the two basis-function centres;
  - at a very large speed of light the two-component Hamiltonian and the
    energy reproduce the non-relativistic ones to `ENERGY_TOL`, and a
    constant shift of the potential shifts the Hamiltonian by that
    constant times the overlap;
  - a hydrogen-like ion: the lowest positive-energy eigenvalue of the
    decoupling converges with an even-tempered basis towards the Dirac
    energy c²(√(1 − (Zα)²) − 1) of the point nucleus, for which X2C is
    exact;
  - spin-free energies match PySCF's `sfx2c1e` and two-component energies
    match PySCF's `x2c` module, with PySCF's speed of light and nuclear
    model passed explicitly;
  - with spin–orbit coupling the energy is invariant under rotations of
    the molecule, while the spin-rotation invariance above holds for
    spin-free Hamiltonians only; the Kramers-restricted (TSCW) and
    unrestricted (TSW) results agree for closed shells by the embedding
    test; a spin-separable class is rejected;
  - the atomic decoupling agrees with the full one to the tolerance the
    literature reports; for cells, the large-box test reproduces the
    molecular two-component result, since no periodic two-component
    oracle is relied on.
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
pass for all eight structures. Status: done 2026-10-21 (PRs #2 and #3);
point-group detection moves to M6. Deferred from the review of #3: expose
basis set versions on `BasisSet` and warn when a molecule mixes Basis Set
Exchange revisions (the cache already records them).

**M2 Integrals.** `IntegralEngine` protocol designed for molecules and
cells alike: shell pairs at explicit translation vectors, range-separated
Coulomb kernels, and the Fourier transform of basis-function pairs.
Native Rust kernels in the McMurchie–Davidson scheme (ADR-0008), with
libcint through PySCF as the oracle only (ADR-0014), in this order: Boys function, one-electron integrals,
two- and three-centre Coulomb integrals with range separation, the pair
Fourier transform (native from the start, since libcint has none), then
four-centre integrals, each validated against libcint before it replaces
the adapter. Exit: the Rust engine matches libcint to 1e-12 on a test
suite of molecules and basis sets up to f functions, and the Fourier
transforms match numerical quadrature. Status: Boys function, Hermite
intermediates, the spherical transformation and the one-electron kernels
(overlap, kinetic, nuclear attraction with range separation, all with ket
translations) done 2026-10-21 and matching libcint to 1e-12 up to g
functions (ADR-0009 records the verified p-shell order); the
`IntegralEngine` protocol and `RustEngine` exist. Two- and three-centre
Coulomb integrals with range separation and `DensityFitting` (metric,
three-centre tensor, fitted `B`) done 2026-10-21, matching libcint and
PySCF's fitted repulsion. Four-centre integrals with Schwarz screening,
shell-pair bounds and permutational symmetry done 2026-10-21, matching
libcint to 1e-12 up to g functions. Fourier transforms of functions and
pairs done 2026-10-21, matching numerical quadrature and PySCF's analytical
transforms. The kernel list of M2 is complete; derivative integrals follow
with the gradients of M11 and serve the relativistic integrals of M12. No
libcint adapter is built (ADR-0014, 2026-10-21); the native engine is the
only one. General contractions are evaluated
once per shell since 2026-10-21 (ADR-0011), after benchmarks against PySCF
showed the split representation costing fifteen times libcint's CPU time
on cc-pVDZ; on segmented bases the four-centre kernel matches libcint's
CPU cost. Remaining performance items: an AO-direct Coulomb and exchange
build for exact integrals at scale, per-block allocations in the kernels.

**M3 Hartree–Fock.** SCF driver, guesses, DIIS, DF and in-core exact J/K,
all eight Fukutome classes from one implementation, PySCF reference tests.
Exit: energies match PySCF for RHF/UHF/GHF and complex variants; paired
classes satisfy embedding consistency tests. Status: the driver over
`BlockMatrix`, the core-Hamiltonian and atomic-density guesses, DIIS, and
the fitted and in-core Coulomb and exchange builders done 2026-10-21
(ADR-0010); all eight classes reproduce the closed-shell energy and the
embedded solutions are stationary; RHF, UHF and GHF match PySCF with exact
and fitted integrals. Spin expectation values, level shifting, damping and
canonical orthogonalization done 2026-10-21. M3 is complete; stability
analysis and symmetry-breaking workflows follow in M11.

**M4 Periodic foundations.** `Lattice` and `Cell`, k-point meshes with
time-reversal pairing and weights, the k axis on `BlockStructure` and
`BlockMatrix` (batched blocks, half-zone storage with conjugate and
Kramers partners along k), lattice-summed overlap and kinetic integrals,
Ewald nuclear repulsion in one to three dimensions. Exit: the large-box
test reproduces the molecular overlap and kinetic matrices to
`ENERGY_TOL`; the Ewald sums reproduce tabulated Madelung constants in
three, two and one dimensions and PySCF's `energy_nuc`, independently of
the Ewald splitting parameter; `BlockMatrix` property tests pass for all
eight classes with k-points; the band-folding test holds for the kinetic
band structure. Status: `Lattice`, `Cell`, `KPoints` with time-reversal
pairing, and the Ewald nuclear repulsion in one to three dimensions
(ADR-0012) done 2026-10-21, matching the Madelung constants, PySCF's
`energy_nuc` in three and two dimensions, and independent of the
range-separation parameter. The k axis on `BlockStructure` and
`BlockMatrix` with half-zone storage done 2026-10-22 (ADR-0015): the
property tests pass for all eight classes with k-points and the half-zone
and full-zone storage agree. The Bloch sums of the overlap and kinetic
integrals done 2026-10-22 (ADR-0016), with the images chosen from the
basis at a precision: the large-box test reproduces the molecular
matrices in one to three dimensions, the band-folding test holds for the
kinetic bands, and the matrices match PySCF's `pbc_intor`. M4 is
complete.

**M5 Periodic Hartree–Fock.** First item: an ADR choosing the treatment
of the exchange divergence (probe-charge Ewald, truncated Coulomb or
Wigner–Seitz truncation, see §5.9), decided with the range-separation
independence test in hand, since the choice fixes the long-range kernel
of the density fitting. Then range-separated Gaussian density fitting
with momentum transfer, nuclear attraction by the same split, k-point J
and K with that correction, occupations across k, all eight classes from
the M3 driver; three dimensions first, then the two- and one-dimensional
reciprocal-space kernels. Exit: the band-folding and
half-zone tests of §8 pass to `ENERGY_TOL`, the range-separation and
large-box tests pass to their stated tolerances, and KRHF/KUHF/KGHF agree
with PySCF's `pbc` module within the recorded fitting residual. Status:
the probe-charge Ewald correction chosen (ADR-0017), with
`madelung_constant` matching PySCF, and the regularized nuclear attraction
of three-dimensional cells done 2026-10-22 (ADR-0018): native short-range
kernel with rigorous screening, independent of the range separation,
converging to the molecular matrix in a large box with the expected power
law, folding across supercells and matching PySCF's density-fitted
potential. The density fitting with momentum transfer done 2026-10-22
(ADR-0019): the metric per transfer and the tensor over every k-point
pair from the screened Bloch three-centre kernel and the reciprocal-space
sums at `q + G`, independent of the range separation, reproducing the
molecular fitting in a large box with the Madelung terms, and matching
PySCF's fitted integrals. The k-point Hartree–Fock done 2026-10-22
(ADR-0020): Coulomb and exchange matrices over the full k set with the
exchange correction, occupations across k, spin expectation values per
cell, the periodic Hamiltonian and the eight classes over k; KRHF, KUHF
and KGHF match PySCF. The two- and one-dimensional reciprocal-space
kernels done 2026-10-22 (ADR-0021): the perpendicular components of the
three-dimensional transform integrated by quadrature with the finite part
of ADR-0012 on the `G_∥ = 0` line, the fitting metric signed along the
charge, the nuclear attraction, fitted integrals and energies of slabs and
chains independent of the range separation, folding, reproducing the
analytic Ewald sums for point charges and a box with growing vacuum, and
KRHF of a slab matching PySCF's two-dimensional treatment. M5 is complete.
Known limitations: the universal JKFIT basis
can be near-linearly dependent in a small cell (LiH in 4 Bohr), where the
fitted integrals agree with PySCF only to the fitting residual of 1e-6
and the reference tests choose their tolerance from the metric's
conditioning; fractional occupations across k-points (smearing) are not
part of version 1. The real-space three-centre sum rebuilt 2026-10-24
(ADR-0025): image lists by the exponents with the neglected tail bounded,
screening tabulated once per step, accumulation per Born–von Kármán class
and each pair of images evaluated once, which builds the fitted tensor of
a silicon cell (def2-SVP, 2×2×2) in 19 s instead of 129 s. ADR-0028
(2026-10-24) folds the step phases as well, so the kernel accumulates
real integrals per class of the image and of the step and applies both
transforms once, and fits the Coulomb term alone for pure functionals
from the blocks of zero momentum transfer; the remaining cost of the full
tensor is the primitive quartets and the reciprocal-space pair
transforms, one pass per momentum transfer, and fitting pairs of diffuse
functions in reciprocal space stays on the later list.

**M6 Spatial symmetry.** Molecular point groups: detection and
orientation, Abelian irreps, SALCs, blocked SCF, per-irrep occupations.
Crystal space groups through spglib: the irreducible wedge with weights
and rotated partner blocks. Exit: symmetry-constrained results equal
unconstrained results for symmetric molecules and crystals; irreps of
orbitals agree with PySCF; wedge-reduced results equal full-zone results.
Status: part one done 2026-10-22 (ADR-0022): the Abelian groups as sign
operations with parity irreps, detection of the full point group with
the standard frame of the Abelian subgroup following PySCF's frame
rules, orientation as an explicit step, the signed-permutation
representation on the basis with SALCs, irrep blocks, orbital labels and
the symmetrization of degenerate sets; full groups, frames, SALC counts
and labels match PySCF for twenty molecules. Part two done 2026-10-22
(ADR-0023): the irrep sectors multiply the spin pattern of
`BlockStructure`, the dense form stays in the atomic basis, the SCF is
blocked over the sectors with the two-electron terms built in the atomic
basis and the one-electron matrices projected onto the sectors,
occupations are shared over the sectors of a spin block or fixed per
irrep, and `SymmetrySpec.point_group` asks for it; all eight classes
reproduce the unconstrained results for symmetric molecules, and
energies, orbital energies, labels and electrons per irrep match PySCF's
symmetry-adapted classes. Part three done 2026-10-23 (ADR-0024): space
groups through spglib, now a dependency, the Wigner matrices of
real solid harmonics from their polynomial table, the action of the
group on the Bloch basis with lattice phases, the point operations as an
attribute of the k axis so that every class stores the irreducible wedge
and derives the rotated points, projection onto the wedge as the group
average, and `SymmetrySpec.with_space_group`; all eight classes
reproduce the full-zone results for a non-symmorphic crystal, a
hexagonal slab and a chain, and energies, wedge sizes, weights and
orbital energies match PySCF's symmetric KRHF. Later: the fitted tensor
over wedge × full set and the exchange and the one-electron integrals at
the wedge alone, which the projection does not yet save; little-group
blocking at high-symmetry points.

**M7 DFT.** Grids and basis-on-grid in Rust, the functionals of libxc,
RKS and UKS with LDA/GGA/meta-GGA/hybrid/range-separated functionals,
then periodic KS with lattice-image grids. Exit: matches PySCF to
grid-limited precision on a functional matrix, molecular and periodic.
Status: part one done 2026-10-24 (ADR-0026): grids from the
Treutler–Ahlrichs mapping, the Lebedev–Laikov rules, NWChem pruning and
Becke's partition with PySCF's size tables, which reproduce PySCF's grids
point by point; basis values with first and second derivatives on grid
points from the compiled layer; the functionals through the pure-Rust
`xcx` crate, which reproduces libxc to 1e-13 at the points tried, named
as libxc names them with a registry of common names; the
exchange–correlation term of the Hamiltonian for the restricted and
unrestricted classes with point-group sectors, hybrids and
range-separated hybrids through the exact and fitted builders with the
long-range kernel; RKS and UKS energies match PySCF to 1e-10 Hartree for
SVWN, PBE, BLYP, B3LYP, PBE0, TPSS, r²SCAN, M06-L, M06-2X and the ωB97
functionals without VV10, with exact and fitted integrals. Part two done
2026-10-24 (ADR-0027): a cell's grid partitions space among the atoms and
their lattice images with the Stratmann–Scuseria–Frisch step, which makes
the periodic partition of unity local, and takes a larger Lebedev rule on
the shells shared with the neighbours; Bloch sums of the basis functions
on the grid from the compiled layer; the exchange–correlation term over
the full k set, reduced onto the stored points as the Coulomb matrix is,
for RKS and UKS with the fitted exchange of hybrids; periodic energies
match PySCF to 1e-12 on one and the same grid, and the grid of ADR-0027
integrates the electron count to 1e-5 where PySCF's truncated grids
reach 1e-3. Part three done 2026-10-25 (ADR-0029): the generalized classes take the
multi-collinear form of the functional, the closed form `f(ρ, |m|)` for
an LDA and a quadrature over spin directions with the second derivatives
of `xcx` for GGA and meta-GGA, on product rules aligned with the
magnetization of each point and refined to a precision, strongly
polarized points in the locally collinear branch with its exact
derivatives; the spinor
density on the grid and the potential `v_ρ + v_m·σ` in the
exchange–correlation term for molecules and cells; the kernel reproduces
PySCF's `mcfun` to rounding, and the generalized classes reproduce the
restricted and unrestricted energies in the collinear limit. Later:
range-separated hybrids for cells (fitted integrals with the long-range
kernel on a lattice), VV10 and dispersion corrections, functionals
outside `xcx`'s list (SCAN, HSE, CAM-B3LYP, Laplacian meta-GGAs),
caching the basis values across SCF iterations, a smooth blend of the
two noncollinear branches and the second derivatives of the
multi-collinear functional for response theory.

**M8 Correlated methods.** DF MO transformation, DF-MP2, `BlockTensor`
with structured einsum over irreps and crystal momentum, equation
generator, CCSD, (T); k-point MP2 and CCSD from the same code. Exit:
matches PySCF UCCSD/GCCSD/RCCSD and `pbc` KMP2/KCCSD energies. Status:
part one done 2026-10-28 (ADR-0030): `BlockTensor` over the blocks of a
`BlockStructure` with the conservation of spin and irrep charges, derived
blocks by the simultaneous partner map, the dense form and a two-operand
contraction in einsum notation; the orbital spaces of a converged state,
the fitted tensor in the orbital basis with the auxiliary basis
symmetry-adapted, and the orbital energy denominators; DF-MP2 written
once in spin-orbital form for all eight classes with point-group sectors,
matching PySCF's DF-MP2 to 1e-11 Hartree. Part two done 2026-10-28
(ADR-0031): the sectors of a tensor axis over a k axis are the blocks at
every point of the full set, the crystal momentum an exact integer charge
on the k axis, the auxiliary index of a cell the momentum transfers in
the eigenbasis of the metric without derivation, the symmetries of the
state (the spin flip and time reversal) generators of the derived blocks
with one block stored per orbit, and the orbital-basis integrals in the
supercell normalization; k-point DF-MP2 from the molecular code for all
eight classes, folding exactly onto the supercell at Γ and matching
PySCF's KMP2. Part three done 2026-10-30 (ADR-0032): an in-house Wick
generator derives the spin-orbital CCSD equations, canonicalizes and
merges the terms, groups the doubles into permutation orbits and writes
them as a committed program of pairwise contractions over block tensors,
regenerated by a test; the normal-ordered Hamiltonian of the reference
with the fitted tensor between every pair of orbital spaces assembling
the antisymmetrized integrals of any block, shared with MP2; the
amplitude iteration with DIIS; CCSD for all eight classes with
point-group sectors and over k-points from the same program, equal to
FCI for two electrons and matching PySCF's DF-CCSD and k-point CCSD in
all three classes to 1e-10 Hartree. Part four done 2026-10-31 (ADR-0033):
the generator derives the connected and the disconnected triples of (T)
and writes them as a second program that takes windows of the occupied
axis, tensor axes have windows that keep the symmetries of the state,
and the correction is summed over triples of windows under a memory
limit; (T) for all eight classes with point-group sectors and over
k-points, equal across the classes, zero for two electrons, matching
PySCF's CCSD(T) for RHF, UHF and GHF and its k-point KRCCSD(T) and
KGCCSD(T) to 1e-10 Hartree. Part five done 2026-10-31 (ADR-0034): the
generator derives the Λ equations as the linked terms of
``<Φ| (1 + Λ) (H_N e^T)_C |Φ_μ>`` and the blocks of the one-particle
density as ``<Φ| (1 + Λ) ({p† q} e^T)_C |Φ>``, iterated like the
amplitudes and transformed to the atomic basis over the structure of
the reference; for all eight classes with point-group sectors and over
k-points, the density is the derivative of the energy with respect to
the orbital energies, equals the full-CI density for two electrons, and
its natural occupations match PySCF's for RHF, UHF and GHF to 1e-10.
Next: M9; later within M8 the two-particle density for gradients, the
DF-driven ladder with windows over the virtual axes, the point
operations of a space group as further generators over the wedge.

**M9 JAX backend and AD.** Pytree registration, `jax.ffi` bindings with
`custom_vjp` from derivative integrals, libxc `custom_jvp`, implicit
differentiation of the SCF, AD nuclear gradients. Exit: AD gradients match
analytic gradients and finite differences.

**M10 GPU.** CuPy backend validation, DF-HF and DFT on GPU, GPU
basis-on-grid kernel, thread and memory policies. Exit: GPU results equal
CPU results; benchmark suite reports speedups.

**M11 Properties and tools.** Analytic gradients, stability analysis and
symmetry-breaking workflows following the JCTC 2011 paper (dissociation
curves across Fukutome classes as a worked example and regression test),
band structures along k-point paths, Molden/cube output, checkpoints.

**M12 Relativistic Hamiltonians.** First item: an ADR deciding the
nuclear charge model default, the two-electron spin–orbit correction and
whether ECP integrals enter version 1 (§12). Then the kinetic-balance
integrals with all nine Cartesian components in Rust, sharing the
derivative mechanism of M11, with per-nucleus attenuation for Gaussian
nuclei; the uncontracted basis and the contraction matrix on `BasisSet`;
the spin-free decoupling and its Hamiltonian in all eight classes; the
spin–orbit decoupling as a TSCW one-body operator; the atomic decoupling;
the two-electron spin–orbit correction; the picture-change transformation
of the multipole operators of M11; a bundled relativistic basis family;
cells with the atomic decoupling at every k. Its only dependencies are M3
and the derivative kernels, so it can be scheduled before M9–M11 if heavy
elements become pressing. Exit: the relativistic tests of §8 pass: the
non-relativistic limit and the shift invariance to `ENERGY_TOL`, spin-free
and two-component energies against PySCF for a set of heavy-element atoms
and molecules, the hydrogen-like convergence, and orientation invariance
with spin–orbit coupling.

Later: AO-direct J/K, non-Abelian point groups, quaternion storage, EOM-CC, ECPs (pending §12) and pseudopotentials,
four-component Hamiltonians and the Breit interaction, the full per-k
decoupling and analytic gradients of the full decoupling (§5.10),
FFT-based density fitting, explicit Q-vector coupling between k-points.

## 11. Relationship to PySCF

PySCF is the reference for correctness and a source of algorithms, not a
design template. Where this plan differs deliberately:

| PySCF | This library |
|---|---|
| Mutable `Mole` and method objects; `kernel()` mutates state | Frozen specs and results; functions return new objects |
| One class per symmetry case (RHF, UHF, GHF, ROHF, ...) | One SCF over structured matrices; symmetry is data |
| NumPy only; GPU via a separate fork (gpu4pyscf); AD via a separate project (PySCFAD) | Backend-agnostic from the start |
| Separate `pbc` package with its own `KRHF`, `KUHF`, `KGHF`, ... classes | k-points are sectors of the same structured matrices; one SCF |
| Low-dimensional cells padded with vacuum in a 3×3 lattice | Only the periodic vectors are stored; reduced-dimension kernels |
| Units and conventions implicit in call signatures | Stated in `CONVENTIONS.md` and docstrings |
| Loose typing | Strict typing |
| Relativity as separate `x2c` wrapper classes over a spinor basis, plus `DHF` | The two-component Hamiltonian is a one-body operator with a block structure; the generalized classes of the one SCF are the spinor case |

PySCF is never a runtime dependency; its `pbc` module is the reference
for periodic results.

## 12. Open decisions needing your input

1. libxc binding: `pylibxc` versus an own binding through the Rust crate.
   Decided 2026-10-24 (ADR-0026): neither; the pure-Rust `xcx` crate,
   which reproduces libxc's functionals with libxc's ids and conventions
   and differentiates them automatically, since `pylibxc` has no wheels
   for Python 3.10 and later and a C build of libxc would add a C
   toolchain to the Rust-only build. A libxc build behind the same
   interface stays the fallback if a functional outside `xcx`'s list is
   needed.
2. Whether JAX is a development target from M1 (recommended: honor its
   constraints from M1, ship the backend in M9).
3. Space-group detection: spglib (recommended; the standard library for
   crystal symmetry, C with Python bindings, BSD-3-Clause) as an optional
   dependency, versus an own implementation. Molecular point groups stay
   in-house either way, since spglib does not treat finite systems.
   Decided 2026-10-23 (ADR-0024): spglib, as a required dependency
   rather than an extra, since it is a small wheel and the periodic
   systems it serves are a primary target; chains and slabs are padded
   for it.
4. The nuclear charge model for relativistic calculations (§5.10): point
   nuclei (PySCF's default, the simplest oracle comparison) versus the
   Gaussian model of Visscher and Dyall (DIRAC's default, what the
   relativistic basis families are optimized with; recommended).
   Non-relativistic calculations keep point nuclei either way.
5. The two-electron spin–orbit correction on top of the one-electron
   operator: Boettger's screened-nuclear spin–orbit scaling (no new
   integrals; recommended first), its modified form, or atomic mean-field
   integrals (atomic two-electron spin–orbit integrals; the more accurate
   option, as a later addition).
6. Effective core potentials in version 1: native semi-local ECP integrals
   are a substantial kernel of their own, and libecpint would add a C++
   toolchain to a build that is Rust only (ADR-0002); the recommendation
   is all-electron relativistic basis sets only, with ECP data in a basis
   file rejected by a clear error as today, and ECPs kept on the later
   list.

Decided so far: no libcint adapter, the native engine being complete
for M2 and libcint the test oracle only (2026-10-21, ADR-0014); `Molecule`
and `Cell` as siblings under `AtomicSystem` (2026-10-21, ADR-0013);
Apache-2.0 license (2026-10-21, ADR-0001); Rust for the
compiled layer (2026-10-21, ADR-0002); the package name `hafnium`
(2026-10-21, ADR-0003), with the PyPI name still to be obtained, see
ADR-0003 and the risk table; Python 3.11+, NumPy 2, uv, ruff and pyright
strict as the toolchain baseline (2026-10-21, ADR-0004); spherical
functions ordered m = −l..+l with BSE normalization conventions and CODATA
2018 units (2026-10-21, ADR-0006); periodic systems as k-point sectors,
with `Cell` as an `AtomicSystem` plus a `Lattice` of only the periodic vectors
(a sibling of `Molecule`, ADR-0013),
Bloch and k-point conventions, and range-separated density fitting for
the periodic Coulomb terms (2026-10-21, ADR-0007); McMurchie–Davidson for
every integral type in the native engine, with Rys quadrature held in
reserve for four-centre integrals (2026-10-21, ADR-0008); the
functionals from the pure-Rust `xcx` crate instead of a libxc binding,
and DFT grids built from PySCF's tables (2026-10-24, ADR-0026).

## 13. Risks

| Risk | Mitigation |
|---|---|
| The native integral engine is the largest single effort | Done for M2: every kernel native and validated against libcint as the oracle (ADR-0014); derivative and periodic kernels follow the same pattern |
| Periodic Coulomb terms (divergences, exchange correction, reduced-dimension kernels) are a second large effort | Range-separated DF with the consistency tests of §8 (range-separation independence, band folding, large box); three dimensions first, slabs and chains after |
| Structured generality grows into a research project | Version-1 scope in §5.8; generality lives in metadata and tests, not in extra code paths |
| Generic spin-orbital CC is slower than spin-adapted RCCSD | Accept for version 1; add a fast path only when measured |
| JAX constraints conflict with iterative algorithms | Pure step functions and implicit differentiation from the start |
| Array API support in JAX still partly experimental | All backend-specific code isolated in `backend/`, with a lint test for the allowed subset |
| Thread oversubscription between rayon and BLAS | Single thread setting applied to both |
| Spin–orbit coupling is the first operator that ties spin to space, so an error in the Kramers structure is invisible in spin-free calculations | The orientation-invariance, embedding (TSCW versus TSW) and non-relativistic-limit tests of §8 with spin–orbit coupling, and PySCF's two-component energies |
| The PyPI name `hafnium` is held by an empty placeholder package | Contact the owner and file a PEP 541 request before M0; publishing is far off, so wait, with `hafnio` as the fallback if the name is not obtained by the first public release |

## References

- C. A. Jiménez-Hoyos, T. M. Henderson, G. E. Scuseria, J. Chem. Theory
  Comput. 7, 2667 (2011): generalized HF and molecular dissociation.
- T. M. Henderson, C. A. Jiménez-Hoyos, G. E. Scuseria, arXiv:1710.00932,
  Table I: the Fukutome classification with orbital-coefficient structures.
- H. Fukutome, Int. J. Quantum Chem. 20, 955 (1981): original classification.
- H. J. Monkhorst, J. D. Pack, Phys. Rev. B 13, 5188 (1976): k-point meshes.
- L. E. McMurchie, E. R. Davidson, J. Comput. Phys. 26, 218 (1978): the
  Hermite Gaussian integral scheme.
- R. Ahlrichs, Phys. Chem. Chem. Phys. 8, 3072 (2006): three-centre
  integrals in the Hermite form for density fitting.
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure
  Theory*, Wiley (2000), chapter 9: integral evaluation.
- Q. Sun, T. C. Berkelbach, J. D. McClain, G. K.-L. Chan, J. Chem. Phys.
  147, 164119 (2017): Gaussian and mixed density fitting for periodic
  systems, the exchange divergence and low-dimensional kernels in PySCF.
- H.-Z. Ye, T. C. Berkelbach, J. Chem. Phys. 154, 131104 (2021):
  range-separated periodic Gaussian density fitting.
- A. Togo, K. Shinohara, I. Tanaka, arXiv:1808.01590: spglib.
- libcint: https://github.com/sunqm/libcint (BSD-2-Clause).
- Basis Set Exchange: https://www.basissetexchange.org and the
  `basis_set_exchange` package.
- JAX FFI: https://docs.jax.dev/en/latest/ffi.html
- Python array API standard and `array-api-compat`.
- REST (Rust electronic structure toolkit) and its `rest_libcint` crate as
  prior art for Rust in quantum chemistry.
- W. Kutzelnigg, W. Liu, J. Chem. Phys. 123, 241102 (2005); M. Iliaš,
  T. Saue, J. Chem. Phys. 126, 064102 (2007); W. Liu, D. Peng, J. Chem.
  Phys. 131, 031104 (2009): the exact two-component Hamiltonian.
- T. Saue, ChemPhysChem 12, 3077 (2011): review of relativistic
  Hamiltonians for molecular calculations.
- K. G. Dyall, K. Fægri, *Introduction to Relativistic Quantum
  Chemistry*, Oxford University Press (2007).
- D. Peng, M. Reiher, J. Chem. Phys. 136, 244108 (2012): local (atomic)
  exact decoupling.
- L. Cheng, J. Gauss, J. Chem. Phys. 135, 084114 (2011): analytic
  gradients of the exact two-component Hamiltonian.
- L. Visscher, K. G. Dyall, At. Data Nucl. Data Tables 67, 207 (1997):
  the Gaussian nuclear charge model.
- J. C. Boettger, Phys. Rev. B 62, 7809 (2000); M. Filatov, W. Zou,
  D. Cremer, J. Chem. Phys. 139, 014106 (2013): screened-nuclear
  spin–orbit scaling and its modified form.
- B. A. Heß, C. M. Marian, U. Wahlgren, O. Gropen, Chem. Phys. Lett. 251,
  365 (1996); J. Liu, L. Cheng, J. Chem. Phys. 148, 144108 (2018): atomic
  mean-field spin–orbit integrals, also within X2C.
- P. Pollak, F. Weigend, J. Chem. Theory Comput. 13, 3696 (2017): the
  x2c-SVPall and x2c-TZVPall basis sets.
- R. A. Shaw, J. G. Hill, J. Chem. Phys. 147, 074108 (2017): libecpint.
