# ADR-0006: Basis function and physical conventions

Status: accepted, 2026-10-21 (ordering and phase to be verified against
libcint when the integral engine lands; revisit then if they differ).

## Context

The molecule and basis layers fix conventions that every later component
depends on: units, the order and phase of basis functions, and how
contraction coefficients are normalized. PLAN.md §12 left the spherical
harmonic convention open.

## Decision

- **Units.** Atomic units inside the library; CODATA 2018 constants
  (`BOHR_IN_ANGSTROM = 0.529177210903`, `HARTREE_IN_EV = 27.211386245988`)
  live in `hafnium.molecule.units` and nowhere else. Input constructors take
  Ångström by default and say so.
- **Spherical functions only.** Every shell of angular momentum `l` spans
  the `2l + 1` real solid harmonics `S_lm`, ordered `m = -l, ..., +l`, as in
  Molden and PySCF. Cartesian basis functions are not supported as a
  computational basis; if ever needed they are converted on input.
- **Phase.** The real solid harmonics follow Helgaker, Jørgensen and
  Olsen, *Molecular Electronic-Structure Theory*, section 6.4, which is
  also the convention of libcint. The integral milestone adds a test that
  compares against libcint and settles any sign differences.
- **Function order.** Shells are ordered by atom, then as listed in the
  basis set data; within a shell, contraction-major and then `m`.
- **Normalization.** Coefficients from the Basis Set Exchange (and the
  Gaussian94 format) multiply *normalized* primitives. `Shell.coefficients`
  stores them unchanged; `Shell.primitive_coefficients()` returns
  coefficients of bare primitives `r^l S_lm exp(-α r²)` rescaled so that
  every contracted function has unit norm, which is what integral kernels
  consume.
- **Data source.** Basis sets come from the Basis Set Exchange JSON API,
  cached per element in the user cache directory. Four sets for hydrogen
  through argon (STO-3G, def2-SVP, cc-pVDZ, def2-universal-JKFIT) are
  bundled for offline tests and examples under the Basis Set Exchange's
  BSD-3-Clause license, with the citation request kept in the data README
  and NOTICE.

## Consequences

- Reference comparisons with PySCF must pass coordinates in Bohr, since
  PySCF converts Ångström with an older CODATA value.
- General contractions keep one `Shell` with several coefficient rows;
  integral engines may regroup them.
- Effective core potentials and Cartesian shells raise `BasisSetError`
  until supported, and ghost atoms raise `InputError` until counterpoise
  corrections are implemented.
- Cache entries record the basis set name, version and revision date they
  came from, so a later change can detect molecules that mix revisions.
