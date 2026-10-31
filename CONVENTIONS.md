# Coding conventions

Status: draft, 2026-10-21. This file is the single source of truth for
naming, notation, layout and style. Changes go through pull request review
like code. Where a rule is enforced by a tool, the tool's configuration is
authoritative and this file explains the intent.

## 1. Language and tooling

- Python ≥ 3.11, NumPy ≥ 2.0. Rust stable, edition 2021 or later.
- Python: `ruff` formats and lints (line length 88, rule set in
  `pyproject.toml`); `pyright` in strict mode type-checks (ADR-0004);
  `pytest` runs tests. All three gate merges.
- Function size is limited by ruff: at most 30 statements, 10 branches,
  cyclomatic complexity 10 and 7 parameters (raised from 6 for the
  Kohn–Sham entry point, which is the Hartree–Fock one plus a functional).
  A function that trips a limit is split, not exempted.
- Rust: `rustfmt` default style; `cargo clippy -- -D warnings` with the
  `pedantic` group enabled; `cargo nextest` runs tests. All gate merges.
- `pre-commit` runs the formatters and linters locally.

## 2. Naming

### 2.1 General

- Python: modules `lowercase`, classes `PascalCase`, functions and variables
  `snake_case`, constants `UPPER_SNAKE_CASE`. Rust: standard Rust naming.
- Names say what a thing is, not how it is stored: `density`, not `dm`;
  `overlap`, not `S` (exception in 2.3).
- Abbreviations are allowed only from the glossary in 2.2. Everything else
  is spelled out (`coefficients`, not `coeffs`; `auxiliary`, not `aux`,
  except as an index letter).
- Counts carry the prefix `n_`: `n_ao`, `n_mo`, `n_occ`, `n_vir`, `n_aux`,
  `n_electrons`, `n_alpha`, `n_beta`, `n_atoms`, `n_grid`, `n_kpt`.
- Functions that compute something are verbs: `build_fock`,
  `compute_energy`, `transform_to_mo`. Predicates start with `is_` or
  `has_`. Constructors that parse input are `from_<source>` class methods
  (`Molecule.from_xyz`). Conversions are `to_<target>`.
- Boolean flags are positive (`use_density_fitting`, never
  `no_density_fitting`).
- Private names use one leading underscore. There is no `__all__`
  gymnastics: public API is what `api/` re-exports.

### 2.2 Glossary of permitted abbreviations

| Abbreviation | Meaning |
|---|---|
| `ao` | atomic orbital (basis function) |
| `mo` | molecular orbital |
| `scf` | self-consistent field |
| `hf` | Hartree–Fock |
| `ks` | Kohn–Sham |
| `dft` | density functional theory |
| `xc` | exchange–correlation |
| `eri` | electron repulsion integral |
| `df` | density fitting |
| `mp2` | second-order Møller–Plesset |
| `cc`, `ccsd` | coupled cluster (singles and doubles) |
| `ccsd_pt`, `CcsdPt` | CCSD(T), with the perturbative triples in parentheses (never `ccsd_t`, which reads as CCSDT) |
| `diis` | direct inversion in the iterative subspace |
| `salc` | symmetry-adapted linear combination |
| `occ`, `vir` | occupied, virtual (only in `n_occ`, `n_vir`, slice names) |
| `rhf`, `uhf`, `ghf` | restricted, unrestricted, generalized HF (only in user-facing names) |
| `kpt` | k-point (as an axis name and in `n_kpt`) |
| `x2c` | exact two-component relativistic Hamiltonian |

### 2.3 Mathematical names

Public signatures use descriptive names:

| Quantity | Name | Shape |
|---|---|---|
| overlap matrix | `overlap` | `[ao, ao]` |
| kinetic energy | `kinetic` | `[ao, ao]` |
| nuclear attraction | `nuclear_attraction` | `[ao, ao]` |
| core Hamiltonian | `core_hamiltonian` | `[ao, ao]` |
| Fock / KS matrix | `fock` | `[ao, ao]` |
| density matrix | `density` | `[ao, ao]` |
| MO coefficients | `mo_coefficients` | `[ao, mo]` |
| MO energies | `mo_energies` | `[mo]` |
| MO occupations | `mo_occupations` | `[mo]` |
| DF three-index tensor | `df_tensor` | `[aux, ao, ao]` |
| four-centre ERIs (chemists') | `eri` | `[ao, ao, ao, ao]` |
| antisymmetrized MO ERIs (physicists') | `eri_antisym` | `[mo, mo, mo, mo]` |
| singles / doubles amplitudes | `t1`, `t2` | `[occ, vir]`, `[occ, occ, vir, vir]` |
| lattice vectors (rows) | `lattice.vectors` | `[d, 3]`, Bohr |
| k-points | `k_points.fractional` | `[kpt, d]`, reciprocal-basis fractions |
| k-point weights | `k_points.weights` | `[kpt]`, summing to one |

Periodic quantities carry a leading k-point axis in front of the molecular
shape: `fock[kpt, ao, ao]`, `mo_coefficients[kpt, ao, mo]`,
`mo_energies[kpt, mo]`. The DF tensor carries the two k-points of its AO
pair, `df_tensor[kpt, kpt, aux, ao, ao]`, the first AO index at the first
k-point and the second at the second; their difference is the momentum
transfer. The axis is named `kpt`, never `k`, because `k` is an occupied
orbital in §2.4.

Inside a short numerical function (under roughly 30 lines) single-letter
mathematical names (`S`, `F`, `D`, `C`, `h`) are allowed when the
docstring or a comment states the mapping once.

Structured objects use the same names; the shapes above then describe each
stored block.

### 2.4 Index letters (einsum subscripts, loop variables, docstrings)

| Letters | Meaning |
|---|---|
| `u v w x` | atomic orbitals (μ ν λ σ in papers) |
| `p q r s` | general molecular orbitals |
| `i j k l` | occupied molecular orbitals |
| `a b c d` | virtual molecular orbitals |
| `P Q R` | auxiliary (DF) functions |
| `g` | grid points |
| `A B` | atoms |
| `X Y` | spin components, only inside kernels that handle spin explicitly |
| `K L M` | k-point sectors (batch axes); a momentum transfer such as `L − K` is written out in prose |
| `T` | lattice translations (real-space images) |
| `G` | reciprocal lattice vectors (plane-wave components); lowercase `g` stays grid points |
| `tx ty tz` | Hermite expansion orders along each axis, only inside integral kernels (ADR-0008) |

Spin never appears as an explicit index in method code; it is part of the
`BlockStructure`.

## 3. Physical and mathematical conventions

- Units: atomic units everywhere inside the library (Bohr, Hartree,
  electron mass, ħ = 1). Conversions happen only at the boundary, in
  `molecule/units.py` and in `io/`. A function argument or attribute in
  other units carries the unit in its name (`length_angstrom`,
  `energy_ev`).
- Ordering of spin blocks in generalized structures: α first, β second.
- Density matrix: `D[u, v] = Σ_i C[u, i] · conj(C[v, i])` over occupied
  `i`, so the first index is unconjugated. The weighted trace of a
  structured density with the overlap equals the electron count; stored
  blocks hold per-spin quantities and weights (multiplicities) are applied
  by `trace`, never baked into the data.
- Inner products conjugate the first factor. `adjoint` means conjugate
  transpose. Hermitian matrices are stored in full, not as triangles.
- Fock matrix: `F = h + J − K` with `J[u, v] = Σ_wx (uv|wx) D_total[w, x]`
  and `K[u, v] = Σ_wx (uw|xv) D[w, x]` per spin block; `D_total` is the
  spin trace. Energy `E = E_nuc + tr(D (h + F)) / 2` with the structured
  trace.
- Kohn–Sham (ADR-0026): functionals are named by libxc's names and ids.
  The densities on a grid follow libxc's layout, `rho [2, g]` for the
  spins, `sigma [3, g]` for `∇ρ_α·∇ρ_α, ∇ρ_α·∇ρ_β, ∇ρ_β·∇ρ_β` and
  `tau [2, g]` for `½ Σ_i |∇ψ_iσ|²`, and the energy per particle `ε_xc`
  and the derivatives of the energy density `ρ ε_xc` come back in the
  same layout. The exact exchange of a hybrid is `a K + b K_lr(ω)` in
  libxc's range-separation convention `a + b erf(ω r)`, so
  `Functional.exact_exchange` is `a` and `long_range_exchange` is `b`.
- Noncollinear Kohn–Sham (ADR-0029): the spin-density matrix on the grid
  is `ρ_στ(r) = Σ_uv φ_v D[vσ, uτ] conj(φ_u)` with the spin blocks of the
  generalized structure, the density `ρ = ρ_αα + ρ_ββ` and the
  magnetization `m = tr(σ ρ) = (2 Re ρ_αβ, −2 Im ρ_αβ, ρ_αα − ρ_ββ)` with
  the Pauli matrices; the gradients and kinetic energy densities follow
  the same combinations, `grad_m [3, 3, g]` with the spin direction before
  the spatial one. The potential enters the spinor Fock matrix as
  `v_ρ + v_m·σ`.
  Basis values on grid points are `[g, component, ao]`, their Bloch sums
  `[kpt, g, component, ao]` complex; the density of a cell on the grid is
  `Σ_k w_k Σ_uv D[v, u](k) conj(φ_u^k) φ_v^k`, the index convention of the
  Coulomb matrix, and the potential `V[u, v](k) = ∫ conj(φ_u^k) v_xc φ_v^k`
  is formed over the full k set and reduced onto the stored points
  (ADR-0027).
  Basis values on grid points are `[g, component, ao]` with the
  components value, `x, y, z`, `xx, xy, xz, yy, yz, zz`. Grid sizes by
  level follow PySCF's tables so that results compare directly.
- Two-electron integrals in the AO basis use chemists' notation
  `(uv|wx)` stored as `eri[u, v, w, x]`. Correlated-method code uses
  antisymmetrized physicists' integrals `<pq||rs>` stored as
  `eri_antisym[p, q, r, s]`. The name says which.
- Block tensors (ADR-0030): every axis of a `BlockTensor` runs over the
  blocks of a `BlockStructure`, stored and derived, with a size per
  sector and a flag marking a conjugated (bra) index; a block is stored
  when the spin charges of its sectors, `+1` alpha, `−1` beta, counted
  negatively on conjugated axes, sum to zero and the characters of its
  irreps multiply to the totally symmetric irrep. `(ia|jb)` conjugates
  `i` and `j`; `<ij||ab>` conjugates `i` and `j`; the amplitudes
  `t2[i, j, a, b] = <ab||ij> / (ε_i + ε_j − ε_a − ε_b)` conjugate `a` and
  `b`, the index character of the excitation `a†_a a†_b a_j a_i`. A
  contraction sums a conjugated against an unconjugated index, so the
  conjugate of an operand is taken explicitly, and the auxiliary index
  of the fitted tensor, itself conjugated, is paired as
  `Σ_P B[P, i, a] conj(B[P, b, j])`. Over a k axis (ADR-0031) a sector is
  a block at a point of the full set, the points of a block adjacent;
  the crystal momentum is an integer charge over twice the mesh size per
  direction, conserved modulo that denominator; the auxiliary index of a
  cell runs over the momentum transfers and carries the transfer
  `k2 − k1` of its pair; the fitted tensor enters the orbital basis
  divided by `√N_k`, the supercell normalization, and correlation
  energies per cell are those of the supercell divided by `N_k`. The
  symmetries of the state derive blocks: the spin flip of a restricted
  state copies, time reversal conjugates and maps every point to `−k`.
  Coupled-cluster code (ADR-0032) reads the normal-ordered Hamiltonian
  as blocks keyed by the `Space` of each slot, `f[OCC, VIR]` for
  `f[i, a]` with `i` conjugated and `v[OCC, OCC, VIR, VIR]` for
  `<ij||ab>` with `i` and `j` conjugated; the amplitudes are `t1[i, a]` and `t2[i, j, a, b]` with the
  virtual indices conjugated, and the residuals `<Φ_i^a| H_N e^T |Φ>` and
  `<Φ_ij^ab| H_N e^T |Φ>` share their structure. The de-excitation
  amplitudes of the Λ equations (ADR-0034) are `l1[i, a]` and
  `l2[i, j, a, b]` with the occupied indices conjugated, the structure
  of the conjugate amplitudes, which the residuals
  `<Φ| (1 + Λ) H_N e^T |Φ_i^a>` and `<Φ| (1 + Λ) H_N e^T |Φ_ij^ab>` share;
  the CCSD density is `D[p, q] = <a†_q a_p>` like the mean-field density,
  its generated blocks `<Φ| (1 + Λ) {p† q} e^T |Φ>` pairing with
  `f[p, q]` in a trace.
- DF tensor: `B[P, u, v]` with the metric already absorbed, so that
  `(uv|wx) ≈ Σ_P B[P, u, v] B[P, w, x]`. Over k-points the tensor lives in
  the eigenbasis of the metric of each momentum transfer and carries the
  `signature` of its eigenvalues, `(uv|wx) ≈ Σ_P s_P B[P, u, v]
  conj(B[P, x, w])`, since the metric of a slab or a chain at zero
  transfer has one negative eigenvalue along the charge (ADR-0021).
- Integral screening: a kernel receives a threshold and leaves every shell
  block whose Schwarz bound `Q_uv Q_wx` lies below it at zero, where
  `Q_uv = max sqrt((uv|uv))` over the functions of the shell pair, so every
  neglected integral is smaller than the threshold. The thresholds
  themselves (`DEFAULT_SCHWARZ_THRESHOLD`) are set in Python, never in the
  compiled layer.
- Spherical harmonics: real solid harmonics ordered `m = −l, …, +l`,
  except p shells, which are ordered `x, y, z` (Molden, PySCF and libcint
  ordering, ADR-0009); phase convention of Helgaker, Jørgensen and Olsen
  equation 6.4.47, implemented in the compiled layer and tested against
  libcint. Cartesian functions are supported for input conversion only
  (decision pending).
- Spin specification on `Molecule` is `n_alpha − n_beta`, never the
  multiplicity 2S+1, to avoid off-by-one errors.
- Complex derivatives: for a real-valued function `E` of complex
  parameters `z`, the gradient returned is `∂E/∂conj(z)` scaled so that
  the real case reduces to the ordinary gradient. Tested against finite
  differences in both real and imaginary directions.
- Point groups (ADR-0022): version 1 imposes D2h and its subgroups. In
  the standard frame their operations are diagonal sign matrices, named
  by Cotton's symbols with reflections named by their plane (`σ(xy)` is
  `z → −z`); the principal axis is `z`, the mirror plane of Cs is `xy`,
  the planes of C2v are `xz` and `yz`. Irreps are listed in Cotton's
  order with Mulliken labels (`A1`, `B2`, `Ag`, ...) and represented by
  parity vectors under the three coordinate reflections, so characters
  are products of signs and direct products are sums of parities modulo
  two. A molecule is oriented into the standard frame explicitly
  (`PointGroupDetection.orient`); the frame rules follow PySCF so that
  labels agree, and symmetric results refer to that frame.
- Irrep sectors (ADR-0023): a `BlockStructure` with `irreps` repeats
  its spin pattern once per irrep the basis spans, in the group's order,
  each block acting on the SALCs of its irrep, `[salc, mo]`; sectors and
  a k axis are exclusive. The dense form of a matrix over sectors is in
  the atomic basis, operators transformed on both sides and orbital
  coefficients on the rows (`to_dense(orbitals=True)`), with the entries
  of a spin block running over its sectors in order. One-electron
  matrices and the Coulomb and exchange matrices enter the sectors by
  projection, which drops couplings bounded by the symmetry tolerance.
  Electrons fixed per irrep follow PySCF's `irrep_nelec`: a total is
  split equally by restricted and paired classes and with the odd
  electron in alpha by unrestricted ones.
- Space groups (ADR-0024): a `SpaceGroup` holds the operations found by
  spglib, each as the integer matrix `W` and fractional translation `w`
  of the lattice basis (`x' = x Wᵀ + w` for row vectors) and as the
  Cartesian `R` and `t` (`r' = R r + t`); k-points map as `k' = k W⁻¹` in
  the reciprocal basis. The Wigner matrix of a shell is
  `S_lm(R⁻¹ r) = Σ_m' D^l[m', m] S_lm'(r)` in the storage order of the
  shell, so `D¹ = R` for p shells. The operation carries the Bloch
  functions at `k` to `Rk` by `U_g(k) = Π_g diag(exp(−i (Rk)·T_A))`, with
  `T_A = R r_A + t − r_{g(A)}` the lattice vector the atom picks up;
  operators map as `U X U†`, coefficients as `U C`. On a k axis with
  operations the real and quaternionic rings store one representative
  per orbit of the operations with time reversal, the complex ring one
  per orbit of the operations alone, each derived point by one
  operation and conjugated only when no operation reaches it; the
  projection onto the stored points is the group average.
- Periodic systems (ADR-0007): `Lattice.vectors[i]` is the i-th
  translation vector, shape `[d, 3]` in Bohr with `d` in {1, 2, 3} and no
  vacuum padding of open directions. Reciprocal vectors obey
  `b_i · a_j = 2π δ_ij` (with the 2π). k-points are fractional coordinates
  in the reciprocal basis wrapped into `[0, 1)`; the time-reversal partner
  of `k` is `(1 − k) mod 1`, and `k` is time-reversal invariant when the
  two coincide. Uniform meshes are Γ-centred by default,
  `k_i = (m_i + s_i) / n_i` with `m_i = 0, ..., n_i − 1` and shift `s_i`
  in {0, 1/2}; the half shift reproduces the original Monkhorst–Pack mesh
  for even `n_i` (for odd `n_i` the two coincide), and the name
  "Monkhorst–Pack" is reserved for that original definition. k-point
  weights sum to one, so traces and energies are per cell; a stored
  half-zone point carries the weight of its time-reversal pair.
  Structured matrices with a k axis (ADR-0015) store blocks as
  `[kpt, ao, ao]` over the stored k-points: blocks over the real and
  quaternionic rings store the lower-index member of every `{k, −k}` pair
  in ascending index order and derive the partner, blocks over the
  complex ring store every point. A real block is `float64` only when
  every point of the set is time-reversal invariant.
- Bloch sums: `φ_μ^k(r) = Σ_T exp(i k·T) φ_μ(r − T)` without a `1/√N_k`
  factor, so `S(k) = Σ_T exp(i k·T) S_T` where `S_T` is the overlap
  between the cell at the origin and the cell at `T`, and
  `S(−k) = conj(S(k))`. Energies, charge and spin are per unit cell; a
  charged cell carries a neutralizing uniform background. The images of
  a lattice sum are chosen from the exponents of the functions so that
  the images left out contribute less than the lattice-sum precision to
  any matrix element (ADR-0025); the kernels screen the terms within the
  kept images against the precision divided by their number (ADR-0018).
- Periodic Coulomb terms use range separation: the short-range part is a
  real-space lattice sum of erfc-attenuated integrals, the long-range part
  is evaluated in reciprocal space with the `G = 0` term replaced by the
  finite part of ADR-0012, over the reciprocal lattice of the periodic
  directions and a quadrature over the open directions of a slab or a
  chain (ADR-0021); the exchange energy carries the probe-charge Ewald
  (Madelung) correction.
  Fourier transforms use the kernel `exp(−i G·r)` in the forward
  direction; reciprocal vectors are `[G, 3]` arrays in inverse Bohr, the
  transform of a function is `[G, ao]` and that of a pair
  `[image, G, ao, ao]`, both complex.
  The range-separation parameter is named `omega` everywhere and results
  must not depend on it beyond the stated tolerance.
- Relativistic Hamiltonians (PLAN §5.10): the speed of light is
  `SPEED_OF_LIGHT = 137.035999084` (the CODATA 2018 inverse
  fine-structure constant) in `molecule/units.py`; every relativistic
  routine takes it as an explicit `speed_of_light` argument with that
  default and never reads it from global state, so that tests can pass an
  oracle's value and a very large value gives the non-relativistic limit.
  The kinetic-balance matrices are `W[u, v] = Σ_i <∂_i χ_u| V |∂_i χ_v>`
  for the scalar part and `W_so[k, u, v] = Σ_ij ε_ijk <∂_i χ_u| V |∂_j χ_v>`
  for the spin–orbit part, so that `(σ·p) V (σ·p) = W + i Σ_k σ_k W_so[k]`
  on spinors ordered α, β. A Gaussian nucleus is the erf-attenuated
  nuclear attraction with `omega = sqrt(zeta)` per nucleus,
  `zeta = 3 / (2 r²)` and `r = 0.836 A^(1/3) + 0.570 fm`, with `A` the
  mass number of the most abundant isotope as tabulated by Visscher and
  Dyall. Point nuclei are the default for non-relativistic calculations;
  the default for relativistic ones is pending (PLAN §12).

## 4. Data model

- Inputs (`Molecule`, `BasisSet`, method specifications, `BlockStructure`) are
  frozen dataclasses (or `attrs` classes with `frozen=True`; decide in an
  ADR and use one everywhere). They validate in `__post_init__` and raise
  on bad input.
- Results are frozen too. A method is a callable that takes a spec and
  returns a result; nothing mutates the spec. Re-running returns a new
  result.
- No module-level mutable state. Configuration (backend, threads, cache
  directory, log level) is an explicit `Config` object passed down or set
  once at the API entry point.
- Arrays inside structured objects are never modified in place. Use
  functional updates (`x.replace(block=...)`).
- `BlockMatrix` and `BlockTensor` are pytrees: arrays are leaves,
  everything else is static metadata. A structure without a k axis holds
  `[ao, mo]` blocks, one with a k axis `[kpt, ao, mo]` blocks; operations
  are written for either rank and never branch on the presence of the
  axis.

## 5. Backend rules

- `import numpy` is allowed only in `hafnium/backend/` and under `tests/`.
  Method code obtains the namespace with `xp = array_namespace(x)` or from
  the `Backend` object and calls `xp.<function>`.
- Only the operations listed in `backend/allowed.py` may be used; a test
  greps for violations.
- Dtype is explicit: `float64` and `complex128` only. No silent promotion
  of real structures to complex; conversions go through `embed`.
- Device placement is handled by the backend; method code never calls
  `.get()`, `.cuda()`, `device_put` or similar.

## 6. Errors and logging

- Exception hierarchy rooted at `HafniumError`. Subclasses include
  `InputError`, `ConvergenceError` (carries the last state), `BasisSetError`,
  `BackendError`, `SymmetryError`.
- Never fall back silently. If a requested feature is unavailable on the
  active backend, raise `BackendError` with the alternative named.
- Logging through `logging.getLogger(__name__)`. No `print` in library
  code. Iteration tables are logged at INFO; per-iteration internals at
  DEBUG.
- Warnings use `warnings.warn` with a library-specific category and only
  for conditions the user can act on.

## 7. Typing and documentation

- Every public function and method is fully annotated. Array arguments use
  the `Array` alias from `backend/types.py`; shapes are documented in the
  docstring with the index names of §2.4, for example
  `mo_coefficients : Array[ao, mo]`.
- Docstrings follow the NumPy style. Physical quantities state their units
  if not atomic, and any convention from §3 that the function relies on is
  named (for example "chemists' notation").
- Theory notes in `docs/theory/` use the same symbols and index letters as
  the code, so a derivation can be matched line by line to an `einsum`.

## 8. Tests

- Layout: `tests/unit/` (fast, no external references), `tests/property/`
  (invariants, may be slower), `tests/reference/` (comparisons against
  PySCF, libcint or stored reference values; marker `reference`),
  `tests/gpu/` (marker `gpu`).
- Test names: `test_<unit>_<behaviour>`, for example
  `test_block_matrix_trace_applies_weights`.
- Tolerances are named constants in `tests/tolerances.py`
  (`ENERGY_TOL = 1e-10`, `INTEGRAL_TOL = 1e-12`, `GRID_TOL = 1e-6`, ...)
  and never inline literals.
- Reference values are stored with provenance (program, version, settings,
  date) in `tests/reference/data/*.json`.
- Every bug fix adds a test that fails before the fix.

## 9. Rust

- Crate layout mirrors the Python package names (`integrals`, `grid`,
  `basis`).
- `#![deny(unsafe_code)]` crate-wide; `unsafe` only in the `ffi` module,
  each block with a `// SAFETY:` comment.
- Public items are documented; `#![warn(missing_docs)]`.
- `f64` for all floating-point data; `usize` for sizes and indices;
  shell tables in structure-of-arrays form.
- Errors are `Result<T, Error>` with `thiserror`; panics never cross the
  FFI boundary. Input validation happens on the Python side and is
  re-checked cheaply in Rust.
- Parallelism via `rayon`; the pool size is set once from Python.
- Unit tests live next to the code; integral accuracy tests compare
  against stored libcint values; `criterion` benchmarks under `benches/`.
- Same glossary and index letters as §2 in identifiers and comments.

## 10. Git and process

- Trunk-based development: short-lived branches, pull requests, at least
  one review, CI green before merge.
- Conventional commit messages (`feat:`, `fix:`, `refactor:`, `docs:`,
  `test:`, `perf:`, `build:`), imperative mood, body explains why.
- Architecture decisions are recorded in `docs/adr/NNNN-title.md` with
  the sections Context, Decision, Consequences. An ADR is never edited
  after acceptance; it is superseded by a new one.
- Semantic versioning; `CHANGELOG.md` updated in the same pull request as
  the change.
- Generated code (CC equations) is committed together with its generator
  and regenerated in CI to detect drift.
