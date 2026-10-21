# ADR-0011: General contractions are evaluated once per shell

Status: accepted, 2026-10-21. Amends the shell-table clause of ADR-0008.

## Context

The first native kernels (ADR-0008) took a shell table with one entry per
contraction: a general contraction of a `Shell` (ADR-0006) was split by
the Python layer into consecutive entries sharing the exponents. That kept
the kernels simple and the function order unchanged, and the small test
molecules did not show the cost.

Benchmarks of the finished Hartree–Fock code (PR #9, #10) against PySCF on
benzene and naphthalene did: in the segmented def2-SVP basis the
four-centre kernel costs the same CPU time per integral as libcint, in the
generally contracted cc-pVDZ basis fifteen times more. With the shells
split, every quartet of carbon s functions recomputes the same primitive
quartet, Boys function included, for each of the `3⁴` combinations of
contractions, because the McMurchie–Davidson intermediates depend only on
the exponents and the contraction coefficients enter at the very end.

## Decision

- A `ShellTable` entry is one shell with all its contractions: the
  coefficients are stored shell by shell as `[contraction][primitive]`
  and `n_contractions` gives the row count of each entry. The function
  order stays contraction-major, then `m`, so nothing changes for the
  Python layer beyond `ShellTable.from_basis` no longer splitting shells.
- The pair layer builds the Hermite expansion of every primitive pair
  once and carries the `k_a × k_b` products of contraction coefficients
  per pair. Every kernel evaluates the primitive quantity without
  coefficients and accumulates it into all contraction combinations
  (`PrimitivePairs::accumulate` for the two-index kernels and the Fourier
  transforms, `accumulate_quartet` for the four-index Coulomb kernel).
  Cartesian blocks are laid out `[k_a, cart_a, k_b, cart_b, …]`, and the
  spherical transformation acts on the Cartesian axes of that layout.
- Schwarz bounds and the permutational symmetries of the four-centre
  kernel act on whole shells, so a shell pair's bound is the largest over
  its contractions. The bound is coarser than per-contraction bounds but
  identical to what libcint uses.
- The scatter of four-centre blocks into the output tensor runs in
  parallel over ranges of the first index; each task scans the blocks of a
  chunk and writes the elements whose first index it owns, which keeps the
  write-each-element-once property without any unsafe code.

## Consequences

- The per-integral CPU cost on benzene in cc-pVDZ drops eight-fold, to
  within a factor of two of libcint's, where the segmented def2-SVP basis
  already was; the exact in-core path becomes usable for benchmarks at a
  few hundred functions, although density fitting remains the production
  route. The remaining factor and the thread efficiency of the four-centre
  kernel (about 20 of 32 threads against libcint's 31) are open.
- Hand-built tables pass `n_contractions`; omitting it means one
  contraction per entry, so the test tables of the earlier milestones
  are unchanged.
- The Rust tests compare every kernel on a generally contracted table
  against the same table with its contractions split, which is the
  invariant the earlier representation enforced by construction.
