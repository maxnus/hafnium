# ADR-0014: No libcint adapter

Status: accepted, 2026-10-21. Amends the libcint clauses of ADR-0002 and
ADR-0008.

## Context

ADR-0002 planned the integral path as "libcint behind an interface, then
a native Rust engine", and ADR-0008 described each native kernel replacing
the adapter for its integral type. The adapter's purpose was to unblock
the Hartree–Fock milestone before native integrals existed. The native
engine reached every kernel of M2 first (PR #5 to #8, #11), validated
against libcint through PySCF at `INTEGRAL_TOL`, and M3 was built on it.
Building the adapter now would mean compiling libcint inside the Rust
build on every platform for a code path with no user.

## Decision

- No `LibcintEngine` is built. `RustEngine` is the only implementation of
  `IntegralEngine`; the protocol stays, so another engine can be added if
  one is ever needed.
- libcint, through PySCF, remains the test oracle for every integral type
  it provides, as ADR-0008 states; PySCF stays a development dependency
  only (PLAN.md §11).

## Consequences

- The plan's component list, M2 description, open decisions and risk
  table no longer mention the adapter.
- Derivative integrals (M11) and the periodic kernels (M4, M5) are native
  from the start and are validated against libcint or PySCF's `pbc`
  module where those provide a counterpart, and against finite
  differences or numerical quadrature otherwise.
