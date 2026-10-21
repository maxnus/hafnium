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
  cyclomatic complexity 10 and 6 parameters. A function that trips a limit
  is split, not exempted.
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
| `diis` | direct inversion in the iterative subspace |
| `salc` | symmetry-adapted linear combination |
| `occ`, `vir` | occupied, virtual (only in `n_occ`, `n_vir`, slice names) |
| `rhf`, `uhf`, `ghf` | restricted, unrestricted, generalized HF (only in user-facing names) |
| `kpt` | k-point (as an axis name and in `n_kpt`) |

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
- Two-electron integrals in the AO basis use chemists' notation
  `(uv|wx)` stored as `eri[u, v, w, x]`. Correlated-method code uses
  antisymmetrized physicists' integrals `<pq||rs>` stored as
  `eri_antisym[p, q, r, s]`. The name says which.
- DF tensor: `B[P, u, v]` with the metric already absorbed, so that
  `(uv|wx) ≈ Σ_P B[P, u, v] B[P, w, x]`.
- Spherical harmonics: real solid harmonics ordered `m = −l, …, +l`
  (Molden and PySCF ordering); phase convention documented in
  `basis/spherical.py` with a test against tabulated values. Cartesian
  functions are supported for input conversion only (decision pending).
- Spin specification on `Molecule` is `n_alpha − n_beta`, never the
  multiplicity 2S+1, to avoid off-by-one errors.
- Complex derivatives: for a real-valued function `E` of complex
  parameters `z`, the gradient returned is `∂E/∂conj(z)` scaled so that
  the real case reduces to the ordinary gradient. Tested against finite
  differences in both real and imaginary directions.
- Point-group orientation: the standard orientation puts the principal
  axis along z and follows the Cotton ordering of irreps; irrep labels use
  Mulliken symbols (`A1`, `B2`, `Ag`, ...).
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
- Bloch sums: `φ_μ^k(r) = Σ_T exp(i k·T) φ_μ(r − T)` without a `1/√N_k`
  factor, so `S(k) = Σ_T exp(i k·T) S_T` where `S_T` is the overlap
  between the cell at the origin and the cell at `T`, and
  `S(−k) = conj(S(k))`. Energies, charge and spin are per unit cell; a
  charged cell carries a neutralizing uniform background.
- Periodic Coulomb terms use range separation: the short-range part is a
  real-space lattice sum of erfc-attenuated integrals, the long-range part
  is evaluated in reciprocal space with the `G = 0` term removed; the
  exchange energy carries the probe-charge Ewald (Madelung) correction.
  The range-separation parameter is named `omega` everywhere and results
  must not depend on it beyond the stated tolerance.

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
  everything else is static metadata.

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
