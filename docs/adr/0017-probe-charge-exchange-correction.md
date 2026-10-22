# ADR-0017: Probe-charge Ewald correction of the exchange divergence

Status: accepted, 2026-10-22.

## Context

The exchange energy of a cell sampled on a k-point mesh contains the
Coulomb kernel at zero momentum transfer. The singularity of `4π/|q + G|²`
at `q + G = 0` is integrable over the Brillouin zone, but a discrete sum
over k misses the integrable part: with the `G = 0` term removed the
exchange energy converges only as `N_k^{-1/3}` in three dimensions. The
Coulomb and the nuclear terms have no such problem, since their `G = 0`
parts cancel for a neutral cell under the regularization of ADR-0012.

ADR-0007 named the probe-charge Ewald correction (Gygi and Baldereschi;
Paier, Hirschl, Marsman and Kresse, J. Chem. Phys. 122, 234102, 2005),
PySCF's default, without weighing it against the two truncated kernels:
the spherically truncated Coulomb kernel of Spencer and Alavi (Phys. Rev.
B 77, 193110, 2008), `4π (1 − cos(G R_c)) / G²` with `R_c` the radius of
a sphere of the supercell's volume, and the Wigner–Seitz truncation of
Sundararaman and Arias (Phys. Rev. B 87, 165122, 2013), the Fourier
transform of `1/r` cut off at the Wigner–Seitz cell of the supercell,
evaluated numerically. Both converge faster with the mesh for insulators.
The choice had to be made first because it fixes the long-range kernel of
the density fitting: a truncated kernel would have to be fitted, not added
afterwards.

## Decision

- **The exchange divergence is treated by the probe-charge Ewald
  correction.** The exchange matrix of every spin block at every stored
  k-point gains `ξ S(k) D(k) S(k)`, and the exchange energy of a
  determinant gains `−ξ N / 2` for `N` electrons, where the Madelung
  constant `ξ` is `−2 E` for the Ewald energy `E` (ADR-0012) of a unit
  point charge on the supercell lattice `n_i a_i` of the mesh, with its
  neutralizing background. `madelung_constant(lattice, kpoints)` computes
  it in every dimension from the same Ewald sum; it requires a uniform
  mesh, which `KPoints.mesh_size` recovers from the points.
- **The electronic long-range kernels keep the `G = 0` convention of
  ADR-0012.** The reciprocal-space part of every electronic term uses the
  transform of `erf(ωr)/r` with the finite part `−π/ω²` at `G = 0` in three
  dimensions, so that the total kernel's `G = 0` term vanishes, each
  matrix is independent of `ω` on its own, and the exchange evaluated with
  these kernels is the "`G = 0` removed" exchange that the correction
  completes; PySCF's `exxdiv="ewald"` is the same construction.
- **Truncated kernels are not adopted**, for three reasons. The
  range-separated Gaussian density fitting of ADR-0007 relies on a
  long-range kernel that is smooth in `G`: `4π exp(−G²/4ω²)/G²` is summed
  to a cutoff set by `ω` alone. A truncated kernel adds `cos(G R_c)/G²`,
  which does not decay, so the reciprocal sum would converge only through
  the Gaussian decay of the pair transforms, with a cutoff set by the
  tightest primitive of an all-electron basis, hundreds of inverse Bohr;
  the Wigner–Seitz kernel has no closed form at all. Plane-wave and grid
  codes afford both because their densities live on grids. Second, one
  kernel serves the Coulomb and the exchange terms, so one fitted tensor
  serves both. Third, the correction is an additive term that costs two
  matrix products per k-point and compares directly with PySCF's Gaussian
  density fitting, which supports no other treatment.

## Consequences

- The leading `N_k^{-1/3}` error of the exchange energy is removed; the
  remaining error falls off faster and is extrapolated in the mesh as
  usual. The band-folding test holds by construction: a supercell at Γ and
  the primitive cell with the matching mesh share the supercell lattice and
  hence `ξ`.
- A k-point list that is not a uniform mesh cannot carry the correction;
  `madelung_constant` raises, as PySCF's does.
- `ξ` of a chain or a slab follows from the Ewald sum of its dimension;
  whether that is the right constant for the reduced-dimension exchange
  is checked by the mesh convergence of the exchange energy when the
  reduced-dimension electronic kernels exist (M5).
- A faster-converging exchange, such as the truncated kernels, can be
  added later behind the exchange builder's interface; it would need a
  second long-range fitted tensor with its own kernel. Revisit for metals,
  where the integrand of the exchange is not smooth and the probe-charge
  correction loses its justification.
