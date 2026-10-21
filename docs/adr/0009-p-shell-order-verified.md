# ADR-0009: Spherical function conventions verified against libcint

Status: accepted, 2026-10-21. Amends the function-order clause of ADR-0006.

## Context

ADR-0006 fixed the order of spherical functions as `m = -l, ..., +l` with
the phase of Helgaker, Jørgensen and Olsen section 6.4, and deferred the
verification against libcint to the integral milestone. The first native
kernels (overlap, kinetic energy, nuclear attraction) now exist and were
compared with libcint through PySCF for basis sets up to g functions.

## Decision

- **Phase and normalization** agree with libcint without change: the real
  solid harmonics of Helgaker equation 6.4.47, Racah-normalized and scaled
  by `sqrt((2l + 1) / 4π)`, applied to bare Cartesian monomials with the
  ADR-0006 primitive coefficients, reproduce libcint's spherical functions
  to `INTEGRAL_TOL` for `l ≤ 4`.
- **p shells are ordered `x, y, z`**, that is `m = +1, -1, 0`, as libcint,
  PySCF and Molden order them. Every other shell keeps `m = -l, ..., +l`.
  `FunctionLabel.m` reports the actual `m` of each function, so labels of a
  p shell read `+1, -1, 0`.
- The transformation is implemented once, in the compiled layer
  (`spherical_transformation`), and exposed to Python for tests.

## Consequences

- Molden output and PySCF comparisons need no permutation of functions.
- Code that enumerates the functions of a shell must use the order of
  `FunctionLabel`, never `range(-l, l + 1)` directly.
- ADR-0006 stays in force for everything else; its status line points here.
