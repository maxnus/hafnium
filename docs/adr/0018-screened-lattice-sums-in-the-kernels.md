# ADR-0018: Screened lattice sums in the kernels and the regularized nuclear attraction

Status: accepted, 2026-10-22.

## Context

The lattice sums of M4 (ADR-0016) take every image from a list that
Python chooses from the basis at a precision, and the kernels evaluate
every term. The nuclear attraction of a cell adds a second lattice sum,
over the images of the nuclei inside every ket image: for STO-3G lithium
in a 4 Bohr cell this is seven thousand ket images times ten thousand
nuclear images, hours per matrix without screening, while PySCF's
density-fitted potential takes a second. The engine also had only the
long-range `erf(ωr)/r` kernel of the range separation, so the short-range
`erfc(ωr)/r` part was reachable only as the difference of two unscreened
sums.

Options for the screening: per-image selection in Python, which cannot
see the primitive pairs whose Gaussian prefactors decide what is
negligible; Schwarz bounds, which are blind to the distance between a
pair and a point charge and diverge for the charge's self-interaction;
or rigorous bounds on every term inside the kernel, with the threshold
chosen in Python. Options for the reciprocal-space part: the engine's
pair transforms at every reciprocal vector, which for thousands of images
exceed memory, or a kernel that contracts them with weights.

## Decision

- **The short-range kernel is native.** `CoulombKernel` of the Rust
  kernels, and the Python `CoulombKernel(omega, short_range)` with
  `PointCharges` on the engine protocol, select `1/r`, `erf(ωr)/r` or
  `erfc(ωr)/r`; the short-range Hermite integrals are the full ones minus
  the attenuated ones in one pass over the charges.
- **Kernels screen terms against a threshold with rigorous bounds, and
  the threshold is policy chosen in Python.** The nuclear attraction and
  the pair Fourier transform take a `threshold`; every term whose bound
  lies below it is left out, so every neglected term is smaller than the
  threshold, the promise of CONVENTIONS.md §3 for Schwarz screening. A
  Cartesian primitive pair with Hermite coefficients `E_tuv`, total
  exponent `p` and a charge `Z` at distance `R` from its centre is bounded
  by `|Z| (2π/p) (1/√p) (s_hi − s_lo) exp(−(1 − ε) s_lo² R²) M` with
  `M = max_ab Π_axis Σ_t |E_t| s_hi^t h_t(ε)`, from
  `R_000 = (1/√p) ∫_{s_lo}^{s_hi} exp(−s² R²) ds` over the exponent range
  of the kernel (`[0, √p]`, `[0, √α]` and `[√α, √p]` with
  `α = p ω² / (p + ω²)`) and the Hermite envelopes
  `|H_t(y)| exp(−ε y²) ≤ h_t(ε)`, `ε = 0.1`. The pair transform is bounded
  by `∫ |φ_a φ_b| dr ≤ (π/p)^{3/2} max_ab Π_axis Σ_t |E_t| p^{t/2} sqrt(2^t t!)`.
  The largest contraction coefficient and the row norms of the spherical
  transformations carry the bounds to the matrix elements. Whole pairs
  are skipped by every kernel; charges beyond the radius where the bound
  falls below the threshold are skipped by the short-range kernel.
- **The nuclear attraction of a cell is the regularized periodic
  potential.** `nuclear_attraction_k` and `core_hamiltonian_k` sum the
  short-range part over the nuclear images within the atom radii of
  ADR-0016 plus the distance beyond which `erfc(ωr)` is below the
  precision, with the screened kernel, and the long-range part in
  reciprocal space from the pair transforms at the reciprocal vectors
  within the Ewald cutoff of ADR-0012 for the nuclear charge against a
  pair transform of magnitude at most one, with the finite part at
  `G = 0` carried by the zero vector of the list. Every element of a Bloch
  sum collects one term per image, so each image is converged to the
  precision divided by the number of images. The transforms are evaluated in batches of images
  within a memory budget and contracted in Python, which keeps the
  contraction on the array backend. Only three dimensions are
  implemented; chains and slabs raise until their reciprocal-space
  kernels exist, planned as a quadrature over the perpendicular
  components of the three-dimensional transform with the finite part of
  ADR-0012 subtracted on the `G_∥ = 0` line.
- **One-electron kernels run their images in parallel** over the
  translations with rayon.

## Consequences

- The periodic nuclear attraction is independent of `ω` to the integral
  tolerance, converges to the molecular matrix in a large box with the
  Madelung term and the expected inverse-cube residual, folds across
  supercells, and matches PySCF's density-fitted potential to 1e-13 for
  helium in cc-pVDZ and 2e-11 for LiH in STO-3G, PySCF's own precision;
  LiH takes about a second at the default precision. Loosening the
  precision raises the error in proportion.
- The bounds are loose by a factor that grows with the angular momenta,
  about 30 for s–d and 2e4 for d–d pairs, which enters the screening
  radius logarithmically; a tighter envelope is a local change.
- The three-centre and two-centre Coulomb kernels adopt the same
  `CoulombKernel`, threshold and bounds when the density fitting of M5
  needs their short-range parts over images.
- The engine protocol's `nuclear_attraction` changed its arguments
  (`PointCharges`, `kernel`); the molecular convenience functions are
  unchanged.
