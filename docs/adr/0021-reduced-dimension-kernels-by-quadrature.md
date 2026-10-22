# ADR-0021: Reduced-dimension reciprocal-space kernels by quadrature

Status: accepted, 2026-10-22.

## Context

ADR-0012 fixed the long-range kernels of slabs and chains for point
charges: the Fourier transform of `erf(ωr)/r` along the periodic
directions as a function of the reciprocal vector and of the perpendicular
separation of the pair, with a finite part at `G = 0` chosen so that the
total is independent of `ω` and the absolute energy of a charged cell
follows one convention in every dimension. The electronic terms of M5
(ADR-0018, ADR-0019) were written for three dimensions, where the
long-range term of two distributions is a reciprocal-lattice sum
`(1/V) Σ_G ṽ(G) conj(ρ̂_A(G)) ρ̂_B(G)` of the three-dimensional kernel
`ṽ(G) = 4π exp(−G²/4ω²)/G²` against the Fourier transforms the engine
evaluates at any vector, and chains and slabs raised. The question was how
to evaluate the reduced-dimension kernels for Gaussian pair densities,
whose profile along the open directions is neither a point nor analytic
in closed form once the angular momenta grow.

Two routes were open: analytic perpendicular integrals of the Hermite
Gaussians against the slab and chain kernels, which means a new family of
kernels in the compiled layer with their own recurrences and, for chains,
a quadrature anyway; or a quadrature over the perpendicular components of
the three-dimensional transform, which reuses the existing transforms and
keeps the kernel in Python. PySCF takes a third route, the truncated
Coulomb kernel of Ismail-Beigi on a padded cell, which needs a vacuum and
has no one-dimensional form.

## Decision

- **The long-range term of a slab or a chain is the three-dimensional sum
  with the perpendicular components integrated by quadrature.** With `d`
  periodic and `n = 3 − d` open directions,
  `(1/V_d) Σ_{G_∥} ∫ dⁿG_⊥/(2π)ⁿ ṽ(G_∥ + G_⊥) conj(ρ̂_A) ρ̂_B`, which by
  Parseval along the open directions equals the sum over the kernels of
  ADR-0012. `RangeSeparation.reciprocal_sum` returns the vectors
  `q + G_∥ + G_⊥` and their weights, the kernel times the quadrature
  weight, as a `ReciprocalSum`; in three dimensions the vectors are the
  reciprocal lattice and the weights the kernel with the finite part at
  the zero vector, as before. The nuclear attraction and the density
  fitting contract these weights with the same transforms in every
  dimension.
- **The finite part of ADR-0012 is the Hadamard finite part on the line
  `q + G_∥ = 0`, with its constant.** The integrand `ṽ(G_⊥) F(G_⊥)` of
  that line, `F = conj(ρ̂_A) ρ̂_B`, is not integrable at `G_⊥ = 0`; the
  regularized value is `∫ dⁿG_⊥/(2π)ⁿ ṽ(G_⊥) [F(G_⊥) − F(0)] + v₀ F(0)`
  with `v₀` the kernel of ADR-0012 at `G = 0` for coinciding centres:
  `−π/ω²` in three dimensions (no integral), `−2√π/ω` for a slab and
  `2 ln ω` for a chain. For a point pair this reproduces the slab and
  chain kernels of ADR-0012 exactly, so the three Coulomb terms of a cell
  share one convention and the total energy of a neutral cell is
  convention-free. In the sum the zero vector carries the weight
  `v₀ − Σ_j w_j ṽ(G_⊥,j)`, so that one contraction evaluates the
  regularized integral; the line only occurs at zero momentum transfer.
- **The quadrature is composite Gauss–Legendre along the open directions,
  with a trapezoid rule in the polar angle of a chain.** The integrand of
  a line `q + G_∥ ≠ 0` has poles at `G_⊥ = ±i|q + G_∥|`, a Gaussian
  `exp(−βG²)` from the kernel and two functions of the smallest exponent,
  and the
  oscillation `exp(iG·Δ)` of the perpendicular separation `Δ` of two
  centres; sixteen nodes per panel integrate it to below 1e-15 of its size
  when the panel is no wider than the smallest `|q + G_∥|`, than
  `2/√β` and than `4/Δ`, which a `PerpendicularProfile` of the cell and
  the bases supplies (its extent, smallest exponent and polynomial
  degree). Panels cover `(0, G_c]` symmetrically for a slab and radially
  for a chain, with the number of angles the degree plus the order at
  which `J_m(G_c Δ)` falls below 1e-14. The cutoff `G_c` is the
  three-dimensional bound of `reciprocal_cutoff`, since the quadrature
  approximates the same continuum estimate. No node lies on the periodic
  subspace, so the zero vector is the only point where the kernel is
  singular.
- **The fitting metric is signed.** The slab and chain kernels give a
  charged distribution a self-energy unbounded below (the plane-averaged
  potential of a sheet is `−2π|z|`), so the metric `(P^q | Q^q)` at
  `q = 0` has exactly one negative eigenvalue, along the charge of the
  auxiliary functions, and is positive on neutral combinations and at every
  other transfer; dropping that eigenvalue would lose the charge of every
  pair density. The periodic fitted tensor is therefore
  `B = |Λ^q|^{-1/2} (U^q)† b` in the eigenbasis of the metric with the
  `signature` of its eigenvalues riding along, and every contraction is
  `Σ_P s_P B_P ... conj(B_P)`: `PeriodicDensityFitting.signature`,
  `KPointDensityFittedCoulombExchange.signature`, and
  `signed_inverse_square_root`. In three dimensions every sign is positive
  and the fitted integrals are unchanged; the molecular tensor keeps its
  symmetric `M^{-1/2}` form.

## Consequences

- The nuclear attraction, the metric, the three-centre integrals and the
  Hartree–Fock energy of a slab and of a chain are independent of `ω` to
  1e-12, fold across supercells, and the reciprocal sum reproduces the
  analytic Ewald energy of charged point charges in every dimension to
  1e-10. A slab in a three-dimensional box with vacuum `L` reproduces the
  slab's nuclear attraction up to the Madelung constants' difference and
  the background's `(2π/(AL)) (z − z_A)²` term, with the residual times
  `L` constant to 1e-9 between 20 and 40 Bohr; for a chain the residual
  after the Madelung term falls off as `1/L²`. KRHF of a helium slab
  matches PySCF's analytic two-dimensional treatment to 1e-12 in the
  energy and the orbital energies; PySCF has no exact chain kernel, so
  chains rest on the internal tests.
- A slab costs about ten times the transforms of a comparable
  three-dimensional cell, a chain several tens of times more, since the
  perpendicular quadrature replaces one or two lattice directions by a
  few hundred or a few thousand nodes per line; the cost grows with `ω`
  through the cutoff and with the k-mesh through the panel width, which
  follows the smallest `|q + G_∥|`. The transforms run in batches within
  the memory budget, as before.
- `RangeSeparation.reciprocal_vectors` and `long_range_kernel` are replaced
  by `reciprocal_sum`; `reciprocal_cutoff` takes the dimension of its
  bound; `integer_coordinates` returns three coordinates in every
  dimension; `Lattice.perpendicular_vectors` spans the open directions.
- The constants of PySCF's two-dimensional kernels (the truncated kernel's
  `πL_z/A` per unit charge pair) differ from those of ADR-0012, so matrices
  of charged distributions compare only up to multiples of the overlap;
  energies of neutral cells compare directly, and the reference tests
  compare those.
