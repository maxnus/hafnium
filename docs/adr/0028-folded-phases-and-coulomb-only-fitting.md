# ADR-0028: The Bloch three-centre kernel folds both phases, and pure functionals fit the Coulomb term alone

Status: accepted, 2026-10-24. Refines ADR-0019 and ADR-0025.

## Context

The fitted tensor of silicon in def2-SVP with the universal JKFIT basis
on a 3×3×3 mesh took 78 s on sixteen threads, where PySCF builds the
same tensor in 2 s, and the build grew faster than the number of
k-points: 10.5 s at Γ, 20 s at 2×2×2, 75 s at 3×3×3 on the same image
lists. The profile of the 3×3×3 build split it into the three-centre
integrals themselves, about 9 s and independent of the mesh; the scatter
of every value into the k-point rows of the kernel, about 27 s and
proportional to the number of k-points; one pass of pair Fourier
transforms per momentum transfer, 20 s; the phase contraction of those
transforms in NumPy, 13 s and quadratic in the k-points; and the
stacking of the blocks and the metric, 3 s.

ADR-0025 accumulated the real-space sum per Born–von Kármán class of the
bra image, so that the transfer phase `exp(i q·T_i)` is applied once per
class, but it still multiplied every value by the step phase
`exp(i k2·T_s)` for every k-point. That phase too depends only on the
class of the step, the lattice coordinates modulo the mesh. And the
Coulomb matrix `J(k) = Σ_P s_P B[k, k, P] Σ_k' w_k' ⟨B[k', k', P], D(k')⟩`
reads only the blocks of zero momentum transfer, one per k-point, while
the exchange matrix reads every pair; a pure functional, LDA, GGA or
meta-GGA, never asks for exchange, yet the whole tensor was built for it.

## Decision

- **The kernel accumulates real integrals per class of the image and
  class of the step** and applies both transforms once at the end,
  auxiliary shell by auxiliary shell: first over the step classes into
  `A[c_i][k2]`, then over the image classes with the transfer phase into
  the `[k1, k2]` blocks. The per-value work of the image loop is one real
  addition; the slabs hold `n_k²` real blocks, half the memory of the
  former `n_k` complex slabs over `k2`. `BlochSum` and `BlochImages`
  carry the class of every step, validated as the image classes are. The
  completion of the sum by its conjugate transpose, which supplies the
  pairs evaluated once, runs in place, one pair of blocks at a time.
- **Pure functionals fit the Coulomb term alone.** `Transfers::Zero`
  (`coulomb_only`) collapses the image classes and returns one block
  `(P^0 | u k v k)` per k-point; `coulomb_fitting_k` adds the
  reciprocal-space part at zero transfer only and returns a
  `PeriodicCoulombFitting` with the metric at zero transfer;
  `KPointDensityFittedCoulomb` provides the Coulomb contraction and
  refuses the exchange one. A cell's Hamiltonian takes it when the
  functional has neither full-range nor long-range exact exchange;
  Hartree–Fock and hybrids keep the full fitting.
- **The reciprocal-space part is added into the kernel's tensor in
  place**, block by block, instead of being stacked into a second copy.

## Consequences

- Silicon def2-SVP, 3×3×3, sixteen threads, precision 1e-12: the
  Coulomb-only build takes 9.7 s instead of 78 s, 8 s of it the
  integrals, so an LDA calculation runs in 24 s against PySCF's 5.5 s;
  the full tensor for a hybrid or Hartree–Fock builds in 38 s, 20 s of
  it the pair Fourier transforms and 11 s the kernel. The energies are
  unchanged to 1e-10 Hartree.
- The Coulomb-only tensor holds `n_k` blocks, 140 MB for the silicon
  case against 3.9 GB, and the peak memory of the full build falls from
  about three copies of the tensor to the tensor plus its real slabs.
- What remains of the full build is the one pass of pair Fourier
  transforms per momentum transfer and the integrals themselves, ADR-0025's
  list: a contraction order that reuses the auxiliary side across the
  pair, a tighter envelope for high angular momenta and a per-pair
  spectator bound; the precision of the lattice sums, 1e-12 against
  PySCF's 1e-8, is a separate decision.
