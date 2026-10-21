# ADR-0012: One Ewald regularization in every dimension

Status: accepted, 2026-10-21.

## Context

The nuclear repulsion of a cell is a lattice sum that diverges on its own;
only the sum of the nuclear, electron–nuclear and electron–electron terms
of a neutral cell converges (PLAN.md §5.9). ADR-0007 fixed the approach:
one range separation of the Coulomb kernel, the short-range part summed
in real space, the long-range part in reciprocal space of the periodic
directions only, so chains and slabs carry no vacuum padding, and a
charged cell neutralized by a uniform background. The nuclear repulsion
is the first consumer and had to settle three things: how the `G = 0`
term is regularized in one and two dimensions, where the cutoff policy
lives, and how the reduced-dimension kernels are evaluated.

PySCF implements Ewald sums in three dimensions (uniform background) and
two dimensions (the formulas of Sundararaman and Arias, PRB 87, 165122,
2013) and raises for chains. Its splitting parameter is called `eta`.

## Decision

- The Ewald split is the range separation of CONVENTIONS.md §3 and uses
  its name: `1/r = erfc(ωr)/r + erf(ωr)/r` with the parameter `omega`.
- In every dimension `d` the long-range part at a reciprocal vector `G`
  of the periodic subspace is the Fourier transform of `erf(ωr)/r` along
  the periodic directions, a function of `|G|` and of the perpendicular
  separation of the pair: `4π exp(−G²/4ω²)/G²` for `d = 3`;
  `(π/G) [e^{Gz} erfc(G/2ω + ωz) + e^{−Gz} erfc(G/2ω − ωz)]` for a slab
  with height difference `z`; `∫_{1/4ω²}^∞ exp(−G²u − ρ²/4u) du/u` for a
  chain with axial distance `ρ`, which is `E_1(G²/4ω²)` on the axis.
- The `G = 0` term is defined in every dimension by the same rule: write
  the kernel as `(2/√π) ∫_0^ω dλ (π/λ²)^{d/2} exp(−λ² r_⊥²)`, remove the
  divergence of the lower limit, which does not depend on `ω`, and keep
  the rest. This gives `−π/ω²` in three dimensions (the uniform
  background), `−2π [z erf(ωz) + exp(−ω²z²)/(ω√π)]` for a slab, and
  `−γ − ln ρ² − E_1(ω²ρ²)` for a chain (`2 ln ω` on the axis). The total
  is independent of `omega`; the absolute energy of a charged cell is a
  convention, the same convention in every dimension.
- The electron–nuclear and electron–electron long-range terms of M5 use
  exactly these `G ≠ 0` kernels and `G = 0` finite parts, so that the
  conventions cancel in the total energy of a neutral cell.
- The cutoffs are chosen in Python from a precision bound on the first
  neglected term of each sum (`hafnium.molecule.ewald`), with `omega`
  defaulting to `√π / V^{1/d}`; the compiled layer receives the image and
  reciprocal-vector lists and runs the loops (PLAN.md §3.4).
- The chain kernel is evaluated by composite Gauss–Legendre quadrature in
  the variable `t = ln(4ω²u)`, where the integrand is entire; the special
  functions (`erfcx`, `E_1`, the quadrature) live in `hafnium_core::special`.

## Consequences

- `Cell.nuclear_repulsion_energy` reproduces the Madelung constants of
  NaCl, CsCl, the square lattice and the alternating chain, the energy of
  a point charge in a cubic box with background, and PySCF's `energy_nuc`
  in three and two dimensions, independently of `omega`.
- A slab or a chain is described by its periodic vectors alone; results
  do not depend on a vacuum size, and there is none to converge.
- The chain kernel costs a quadrature per pair and reciprocal vector,
  which is negligible for nuclei; the electronic kernels of M5 will need
  a vectorized evaluation over Gaussian pair densities.
