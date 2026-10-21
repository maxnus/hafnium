# ADR-0007: Periodic systems as k-point sectors

Status: accepted, 2026-10-21.

## Context

Periodic boundary conditions in one, two and three dimensions with k-point
sampling were added to the goals on 2026-10-21, after the molecule and
basis layers (ADR-0006) and the block-structured data model (PLAN.md §5)
were in place. Three things had to be decided before the integral milestone
starts: how a periodic system is represented, how k-points enter the
symmetry-as-data model, and which conventions fix the Bloch functions so
that integrals, SCF and correlated code agree.

Options for the system: a `Cell` holding a `Molecule` (composition), or
`Cell` as a subclass of `Molecule`. For chains and slabs: PySCF's three
lattice vectors with vacuum in the open directions, or only the periodic
vectors with analytic reduced-dimension Coulomb kernels. For k-points in
the data model: a Python list of per-k matrices, or a batch axis on every
block. For the periodic Coulomb terms: FFT-based density fitting on a
uniform grid, or Gaussian density fitting with a range-separated kernel.

## Decision

- `Cell` is a `Molecule` with a `Lattice`. Every molecular function
  (electron counts, symbols, positions) applies to a cell unchanged;
  `nuclear_repulsion_energy` becomes an Ewald sum, and `to_xyz` writes
  the lattice as the extended-XYZ `Lattice="..."` field on the comment
  line, which `Cell.from_xyz` reads back. A molecule is a system without
  a lattice, not a cell with a large box.
- `Lattice` stores exactly the periodic translation vectors,
  `vectors[i]` for `i < dimension` with `dimension` in {1, 2, 3}, shape
  `[d, 3]` in Bohr. Open directions carry no vacuum padding; the
  reciprocal-space Coulomb kernel is the analytic kernel of the reduced
  dimension. Reciprocal vectors satisfy `b_i · a_j = 2π δ_ij`.
- Basis functions of a cell are the Bloch sums
  `φ_μ^k(r) = Σ_T exp(i k·T) φ_μ(r − T)` without a `1/√N_k` factor.
  Matrices, energies, charge and spin are per unit cell, and a charged
  cell is neutralized by a uniform background.
- k-points are fractional coordinates in the reciprocal basis, wrapped
  into `[0, 1)`, with weights. Uniform meshes are Γ-centred by default,
  `k_i = (m_i + s_i) / n_i`, `m_i = 0, ..., n_i − 1`, shift `s_i` in
  {0, 1/2}; the half shift gives the original Monkhorst–Pack mesh for even
  `n_i`. The time-reversal partner of `k` is `(1 − k) mod 1`; a k-point
  equal to its partner is a time-reversal invariant momentum.
- A k-point is a sector label of a `Block`, like the spin sector. All k
  blocks of a sector form one batched array with a leading k axis
  (`[kpt, ao, ao]`); the k-point set with its weights and partner map is
  static metadata of the `BlockStructure`. The antiunitary operators a
  Fukutome class preserves act on k-space as `k → −k` combined with their
  molecular action. The real classes store a half Brillouin zone with the
  block at `−k` the complex conjugate of the block at `k`; paired GHF
  stores a half zone with the block at `−k` the Kramers map of the block
  at `k`; paired UHF stores the alpha block at every k and derives
  `beta(k) = conj(alpha(−k))`; the complex classes store every k. At
  time-reversal invariant momenta the derived block coincides with the
  molecular partner rule (real, quaternionic, or `beta = conj(alpha)`).
  k-point weights sum to one over the full zone, a stored half-zone point
  carrying the weight of both members of its pair, so that traces and
  energies are per unit cell. Blocks are `complex128` unless every stored
  k-point is time-reversal invariant, in which case the real classes store
  `float64` blocks, so a Γ-only cell is identical to the molecular case.
- Two-electron terms conserve crystal momentum modulo a reciprocal lattice
  vector; the DF tensor is labelled by the momentum transfer `q`. The
  periodic Coulomb terms use one range separation of the kernel: the
  short-range part is a real-space lattice sum of erfc-attenuated
  integrals, the long-range part is evaluated in reciprocal space from the
  analytic Fourier transform of Gaussian pair densities, the `G = 0` term
  is removed by the background, and the exchange divergence is handled by
  the probe-charge Ewald (Madelung) correction. The two-electron terms go
  through Gaussian density fitting in that split; the nuclear attraction
  uses the split directly, without an auxiliary basis. FFT-based density fitting is not planned: it needs
  pseudopotentials to be affordable with all-electron Gaussian bases and
  does not extend cleanly to chains and slabs.
- Translational symmetry breaking with a wavevector `Q` is described by a
  commensurate supercell; no explicit coupling between `k` and `k + Q`.

## Consequences

- The integral engine (M2) is designed for shell pairs at explicit
  translation vectors, range-separated Coulomb kernels and the Fourier
  transform of basis-function pairs from the start, although its first
  consumer is molecular. This favours McMurchie–Davidson for the native
  engine, decided in ADR-0008.
- `BlockMatrix` operations become batched over k, `trace` applies the
  k weights, `eigh` runs per k, and occupation policies fill across k.
  The molecular code path is the one-k-point case and must not grow a
  separate branch.
- Space-group reduction of the k set (irreducible wedge, rotated partner
  blocks) is a later extension of the same partner mechanism (M6), with
  detection delegated to spglib if that open decision is confirmed.
- Every periodic feature must pass the consistency tests of PLAN.md §8:
  the large-box test against the molecular result, the band-folding test
  between a supercell at Γ and the primitive cell with the matching mesh,
  half-zone versus full-zone storage, and independence of the
  range-separation parameter.
- Reference comparisons use PySCF's `pbc` module with the same auxiliary
  basis and the Ewald exchange correction; the residual due to the
  fitting scheme is recorded with the reference data.
- Revisit if an application needs explicit `Q` coupling, plane-wave or
  pseudopotential treatment, or if little-group blocking at high-symmetry
  k-points (M6) makes block shapes differ across k, which batched storage
  cannot express.
