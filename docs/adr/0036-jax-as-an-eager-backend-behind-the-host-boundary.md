# 0036 JAX runs the methods eagerly as the active backend, behind a host boundary around the kernels

Status: accepted 2026-11-01

## Context

PLAN.md §6 designs the method code for JAX from the start: the array API
subset of `backend/allowed.py`, immutable data, static block shapes, and
pure step functions. M9 part two makes JAX a backend in fact. Three
things had to be settled: how a calculation chooses its array library
when the arrays that start it, the integrals, come out of the compiled
kernels as NumPy arrays; what happens to the code around the kernels,
which builds shell tables, grids, lattice images and symmetry data with
NumPy-only idioms such as item assignment and spglib calls; and how the
structured types take part in JAX's transformations, which see only
pytrees.

## Decision

- **A calculation runs in the active backend, which the caller selects.**
  `hafnium.backend.use_backend("jax")` makes JAX the active backend inside
  a `with` block, through a context variable, and `set_backend` for the
  rest of a program. `default_namespace()` is the active backend's array
  namespace, so every array that is not derived from an input, the
  integrals included, is created there, and everything downstream follows
  by `array_namespace` of its operands. The backend of an array library is
  loaded on first use: JAX is switched to 64-bit arrays, which the
  `float64`/`complex128` rule of CONVENTIONS.md §5 requires, and the
  structured types are registered as pytrees. No module imports JAX at
  import time, so the dependency stays optional.
- **The kernels and their setup stay on the host; results cross at the
  public helpers.** Shell tables, grids, lattice images, Ewald sums and
  symmetry detection work in `host_namespace()`, NumPy, whatever the
  active backend: they feed the compiled kernels, which read host
  buffers, and they use NumPy idioms that JAX does not support. The
  molecular and periodic integral helpers (`overlap`, `core_hamiltonian`,
  `DensityFitting.build`, `eri`, `overlap_k`, `density_fitting_k` and their
  relatives) convert their results with `from_host` before returning them,
  and the grid values and functional derivatives convert where they meet
  the density. Metadata of the structures that enters arithmetic, the
  SALC coefficients of the irrep sectors and the unitaries of the k-point
  operations, stays on the host and is pulled into the namespace of the
  block it acts on with `xp.asarray`, so a structure does not belong to a
  backend. `array_namespace` rejects mixed inputs, which turns any
  remaining boundary error into a `TypeError` at the operation instead of
  a silent conversion.
- **The block types are pytrees with their structure as static metadata,
  unflattened without validation.** `BlockMatrix`, `BlockVector` and
  `BlockTensor` flatten to their stored arrays with the structure as the
  auxiliary data, as PLAN.md §5 foresees. Unflattening bypasses the
  constructor, as JAX requires of pytree types, since a transformation may
  rebuild a type around placeholder leaves; validation remains in the
  constructors the method code calls.
- **Eager execution first.** Part two runs the methods under JAX as it
  runs them under NumPy, with the Python loops of the drivers and host
  reads of energies and convergence measures. Tracing the SCF with
  `lax.while_loop`, the implicit differentiation of its fixed point and
  the custom derivative rules of the kernels are parts three and four,
  which build on the backend without changing it.

## Consequences

- Every method, Hartree–Fock and Kohn–Sham in the eight classes with
  irrep sectors and over k-points, DF-MP2, CCSD, (T), the Λ equations and
  the nuclear gradient, gives the same energies under NumPy and JAX; the
  parity tests pin the agreement to the convergence tolerances.
- JAX dispatches every array operation separately when it runs eagerly
  on small arrays, so the backend is slower than NumPy on the test
  molecules by an order of magnitude. That is the expected cost before
  tracing; the backend exists for its transformations, not for its eager
  speed.
- The host boundary is explicit in the code: a helper that returns an
  array of the active backend says so, and one that works on host arrays
  takes `host_namespace()`. Code that mixes the two fails loudly.
- A structure's metadata on the host means a traced function sees the
  SALC coefficients and k-point unitaries as constants, which is right for
  static symmetry data.
- JAX joins the development dependencies so the parity tests run in CI;
  the `jax` extra remains for users.
