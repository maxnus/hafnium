# ADR-0025: Lattice sums reach by exponent, and the Bloch three-centre kernel accumulates by class

Status: accepted, 2026-10-24.

## Context

The density fitting of a cell (ADR-0019) spends almost all of its time
in the short-range three-centre lattice sum. For silicon in def2-SVP
with the universal JKFIT basis on a 2×2×2 mesh the fitted tensor took
two minutes on sixteen threads where PySCF builds the same tensor and
converges the SCF in five seconds. Profiling the kernel showed where the
time went, and it was not the integrals: of the processor time inside
the image loop a tenth evaluated primitive quartets, while the rest
re-derived the Hermite envelope bound of every quartet at every image,
allocated and transformed the spherical blocks, zeroed a buffer of the
whole `[k, aux, ao, ao]` block per image, and folded that buffer into a
`[k, k, aux, ao, ao]` accumulator per image; half the available time was
lost to the per-split identity of the parallel fold and to load
imbalance. The image lists Python passed were generous too: an orbital
image was kept while it lay within the sum of the auxiliary function's
radius, the orbital function's radius and the distance beyond which
`erfc(ωr)` is small, three thresholds applied in sequence, which kept
three thousand images of which the kernel found work on a third.

Two things were open: how Python should choose the images of a lattice
sum, and how the kernel should organize the sum so that the integrals,
not the bookkeeping, set the cost.

## Decision

- **Image lists are chosen by the exponents, with the neglected tail
  bounded.** `pair_reach(first, second, lattice, precision)` returns the
  distance per pair of atoms beyond which
  `∫ |a b|`, bounded by `Σ_ij A_i A_j (π/(α'_i + α'_j))^{3/2} exp(−μ'_ij X²)`
  over the primitives, summed over every image of the second atom
  farther than that distance, is below the precision. `coulomb_reach`
  does the same for the short-range kernel between two functions, from
  the closed form of the short-range integral of two s Gaussians,
  `π³ [erf(√ρ D) − erf(√θ D)] / (αβ √(α+β) √ρ D)` with
  `1/θ = 1/α + 1/β + 1/ω²`, and takes a `spectator` bound for the third
  function of a three-centre integral, `periodic_amplitude`, the lattice
  sum of a function's envelope, which is at most its value on a lattice
  point. The polynomial of a function is absorbed by a tenth of its
  exponent, `r^l exp(−δ α r²) ≤ (l/(2eδα))^{l/2}`, so that the Gaussian
  that sets a reach keeps `(1 − δ) α`; the tail of a sum over the lattice
  beyond a radius is the continuum estimate with a margin of two. The
  one-electron sums, the metric and the three-centre sum take their
  images from these reaches (`lattice_images`, `reaching_images` with a
  reach per pair of atoms); the nuclear attraction keeps the function
  radii of ADR-0016 for the nuclear images. The precision of a lattice
  sum now bounds what the images left out contribute to any matrix
  element, and the per-term threshold inside the kernels stays the
  precision divided by the number of steps.
- **The kernel tabulates the screening once per step.** For every step
  the shell pairs above the Schwarz threshold are kept with their
  primitive pairs, and for every shell triple and primitive quartet the
  squared reach, the distance at which the bound of ADR-0018 falls below
  the threshold, per type of auxiliary shell (shells with the same
  exponents and coefficients share their bounds). The image loop then
  compares squared distances with the tables, the shell triple's from the
  auxiliary centre to the segment between the pair's centres, the
  quartet's from the product centre, and evaluates nothing else; the
  spherical transformation runs in a reused buffer.
- **The sum is accumulated per Born–von Kármán class.** The phases of a
  pair of images factor as `exp(i q·T_i) exp(i k2·T_s)` with the
  momentum transfer `q = k2 − k1` and the step `T_s`; on a uniform mesh
  `exp(i q·T)` is the same for every image of a class, the lattice
  coordinates modulo the mesh. The kernel takes the k-points and the
  class of every image (`BlochImages.classes`, `BlochImages.kpoints`
  instead of phase tables), verifies that the transfer phases agree
  within every class, scatters each spherical block with the step phases
  into one slab per class and auxiliary shell under a lock, and applies
  the class phases once at the end. No buffer is zeroed per image, no
  accumulator is folded, and the parallel loop over images balances
  itself.
- **Each pair of images is evaluated once.** When the bra and the ket
  tables coincide and the steps are closed under negation, the integral
  `(P | μ(T_i) ν(T_j))` also serves as `(P | ν(T_j) μ(T_i))`: the pairs
  with `j < i` are skipped, a shell with itself on one image enters with
  half its weight, and the sum of the half is completed by its conjugate
  transpose over k-points and functions, `out[k1, k2, P, μ, ν] +=
  conj(out[k2, k1, P, ν, μ])`, one pass over the output.

## Consequences

- The fitted tensor of the silicon cell builds in 19 s instead of 129 s
  at the default precision (sixteen threads), with the RHF energy
  unchanged to 2e-11 Hartree; at a precision of 1e-8 it takes 8 s
  instead of 72 s. The three-centre image list shrinks from 3025 to 1265
  images and the step list from 875 to 513. Of the 19 s, 12 s are the
  kernel and 6 s the reciprocal-space pair transforms, which are now the
  next target. The fitted integrals of the test cells, their independence
  of the range separation and their agreement with PySCF are unchanged,
  and the one-electron sums keep their precision with fewer images.
- The reaches are rigorous bounds up to the continuum estimate of a
  lattice tail and apply the precision once to the whole term, where the
  old lists applied it to each factor; a looser precision now buys a
  proportionally smaller list. The reach of a diffuse function is set by
  `1/ω² + 1/α + 1/β`, so a larger `ω` shortens the real-space sum at the
  expense of the reciprocal one, as for the Ewald sums.
- The kernel's cost is now the primitive quartets it evaluates and the
  scatter of every value into the `n_k` entries of its row; the next
  steps, when they matter, are a tighter envelope for high angular
  momenta (ADR-0018 notes the slack), a spectator bound per pair of
  functions instead of the largest function of the basis, a contraction
  order that reuses the auxiliary side across the pair, and the
  reciprocal-space fitting of diffuse pairs that ADR-0019 deferred.
- The slabs per class and auxiliary shell hold `n_k` times the
  `[aux, ao, ao]` block per class, the size of the output; a system whose
  output does not fit in memory needs the batching over auxiliary shells
  that version 1 does not have.
- `BlochImages` changed its fields and the engine protocol's
  `coulomb_three_center_bloch` its arguments; `reaching_images` takes a
  reach per pair instead of radii per set.
