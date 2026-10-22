# ADR-0019: Density fitting with momentum transfer

Status: accepted, 2026-10-22.

## Context

ADR-0007 decided that the two-electron terms of a cell go through
Gaussian density fitting with one range separation of the Coulomb kernel,
and ADR-0012, ADR-0017 and ADR-0018 fixed the kernels, the `G = 0`
convention, the exchange correction and the screened short-range sums.
What remained was the shape of the fitted tensor over k-points, how its
two lattice sums are organized, and how the kernels of the compiled layer
expose the short-range three-centre sum, which for diffuse bases runs over
millions of image pairs.

The pair density `conj(φ_u^{k1}) φ_v^{k2}` of two Bloch functions has the
character of the momentum transfer `q = k2 − k1`, so it is fitted in the
Bloch sums `χ_P^q` of the auxiliary functions and the metric depends on
`q` alone. The three-centre integral `(P^q | u k1 v k2)` is, with the
auxiliary function fixed in the home cell, the sum over the images `T_i`
of `u` and `T_j` of `v` with the phases `exp(−i k1·T_i) exp(i k2·T_j)`,
while its long-range part is a reciprocal-space sum over `q + G`. The
engine's three-centre kernel returned every image pair as its own block,
which for the image pairs of STO-3G lithium in a 4 Bohr cell would be
gigabytes before any phase is applied; PySCF accumulates the phases inside
its C driver for the same reason.

## Decision

- **The fitted tensor is `df_tensor[kpt, kpt, aux, ao, ao]`** over every
  pair of k-points, the first orbital index at the first k-point and the
  second at the second (CONVENTIONS.md §2.3), with the first function of
  every pair conjugated: `(u k1 v k2 | w k3 x k4) ≈ Σ_P B[k1, k2, P, u, v]
  conj(B[k4, k3, P, x, w])` for `k2 − k1 = k3 − k4`, and
  `B = (M^q)^{-1/2} b` per transfer. The metric `M^q = (P^q | Q^q)` is
  stored per transfer, `metric[q, aux, aux]`; the transfers of a uniform
  mesh form the Γ-centred mesh of the same sizes whatever the shift, and
  `KPoints.momentum_transfers` returns that set with the table of the
  transfer of every pair. Explicit k-point lists that are not uniform
  meshes are not fitted.
- **The short-range three-centre sum runs in the kernel over image
  pairs found by lattice coordinates, with the phases accumulated inside.**
  `coulomb_three_center_bloch` takes the image list of the orbital
  functions with the integer lattice coordinates of every image, the
  steps at which a pair of functions still overlaps (the pair-image list
  of ADR-0016) with their coordinates, and one phase table per side; the
  ket image of a pair is the bra image displaced by a step, looked up by
  its coordinates, so no pair list crosses the boundary. It returns the
  complex `[kpt, kpt, aux, ao, ao]` sum with the auxiliary function in the
  home cell, accumulating the ket phases per bra image and the bra phases
  once per image, in parallel over bra images. Before any integral, a
  shell triple is skipped when the product of the Schwarz bounds of the
  pair at its step, tabulated once per step, and of the auxiliary shell is
  below the threshold, which is what makes the sum affordable for diffuse
  functions, whose image pairs run into the tens of millions. Python
  chooses the images that reach the home-cell auxiliary functions within
  the short-range radius (`reaching_images`). The metric's short-range
  part is the Bloch sum of the two-centre kernel over the images of one
  auxiliary function.
- **The long-range parts are reciprocal-space sums at `q + G`** of the
  kernel of ADR-0012 with the Fourier transforms of the auxiliary
  functions and of the Bloch-summed pair densities, the finite part at
  `q + G = 0` included; the pair transforms are evaluated in batches of
  images and Bloch-summed for every `k2` at once. The reciprocal cutoff is
  that of ADR-0012 for the charge product of the two distributions, with
  `∫ |χ_P|` bounded from the shell data (`function_absolute_integral`)
  and the pair transforms bounded by one, extended by `|q|`.
- **Every Coulomb kernel takes `CoulombKernel` and a threshold.** The
  two-, three- and four-centre kernels and the Schwarz bounds select the
  full, long-range or short-range kernel through one enum, and the two-
  and three-centre kernels screen primitive quartets by the bound of
  ADR-0018 generalized to two Hermite expansions,
  `(2π^{5/2}/(pq√(p+q))) (1/√α) (s_hi − s_lo) exp(−(1 − ε) s_lo² R²) M`
  with `M` the envelope over the combined orders `t + τ`. The four-centre
  kernel keeps its Schwarz screening of shell quartets and evaluates a kept
  quartet in full. Each image's sums use the precision divided by the
  number of image pairs (or auxiliary images), since every element
  collects one term per pair.

## Consequences

- The fitted integrals of helium in cc-pVDZ with the universal JKFIT
  basis on a shifted 2×2×2 mesh agree with PySCF's `GDF.get_eri` to 3e-13
  in every k-point quadruple tested, are independent of `ω` to 1e-13, and
  converge in proportion to the precision; the metric is Hermitian and
  positive definite at every transfer, the tensor Hermitian across the
  swap of its k-points and functions, and a large box at Γ reproduces the
  molecular fitting with the Madelung terms of the charged auxiliary
  functions and pair densities and an inverse-cube residual.
- The agreement with PySCF is limited by the conditioning of the metric,
  not by the integrals: the universal JKFIT basis of LiH in a 4 Bohr cell
  has metric eigenvalues at 1e-14 of the largest, the fitted integrals
  move by 1e-5 with the eigenvalue cutoff, and the two codes agree to
  1e-6, the fitting residual of a near-linearly dependent auxiliary
  basis. The reference test chooses its tolerance from the metric's
  conditioning and records the case; a fitting basis built for the cell
  would remove the dependence.
- Version 1 holds the whole tensor in memory, `n_kpt² × n_aux × n_ao²`
  complex numbers; the pair `(k2, k1)` is the conjugate transpose of
  `(k1, k2)`, which halves the kernel work and the storage once the
  consumers can address it. The image pairs of diffuse functions dominate
  the cost; the remedy PySCF applies, fitting the pairs of diffuse
  functions in reciprocal space alone, is the extension to consider when
  all-electron bases with diffuse functions matter.
- The k-point Coulomb and exchange builders of M5 contract this tensor;
  the Coulomb matrix needs the `q = 0` slice alone.
- Chains and slabs wait for the reduced-dimension reciprocal-space kernels
  (ADR-0018), as the nuclear attraction does.
