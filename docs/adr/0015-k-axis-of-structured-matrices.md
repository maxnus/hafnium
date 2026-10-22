# ADR-0015: The k axis of structured matrices

Status: accepted, 2026-10-22. Implements the k-point clauses of ADR-0007.

## Context

ADR-0007 decided that a k-point is a sector label of every block, that the
blocks of a sector are one batched array with a leading k axis, that the
real and paired Fukutome classes store a half Brillouin zone with the
block at `−k` derived from the block at `k`, and that k-point weights
enter traces so that energies are per cell. Implementing it raised
questions ADR-0007 left open.

- Is a molecule a structure with one Γ point, or a structure without a k
  axis? The former gives every molecular block a leading axis of length
  one; the latter risks a separate molecular code path.
- `hafnium.symmetry` imports `hafnium.structured` for the Fukutome
  structures, so the k-point set with its geometry cannot be the metadata
  the structure holds without an import cycle.
- Which member of a pair `{k, −k}` is stored, in which order, and how the
  dtype rule of ADR-0007 (`float64` only when every stored point is
  invariant) is applied.
- How traces, diagonalization and the Θ-symmetry checks of the molecular
  quaternionic block treat the points that are stored and the points that
  are derived.
- A quaternionic block at `−k` is `Θ X Θ⁻¹` for an operator but `Θ C` for
  a coefficient matrix, and a square coefficient matrix cannot be told
  from an operator by shape.

## Decision

- `BlockStructure.k_axis` is an optional, keyword-only field. Without it
  the structure is molecular and blocks are `[ao, mo]`; with it blocks are
  `[kpt, ao, mo]` over the k-points the block stores. Every operation of
  `BlockMatrix` and `BlockVector` is written for either rank, batched
  over the leading axis, so method code never branches on the presence of
  the axis. A cell at Γ alone has a one-point axis with `float64` blocks
  of shape `[1, n, n]` and reproduces the molecular numbers.
- `KAxis(partners, weights)` lives in `structured` and holds only what the
  structure needs: an involutive partner map and weights that are
  positive, sum to one and are equal within a pair. `KPoints.axis()` in
  `symmetry` builds it from the fractional coordinates, so `symmetry`
  depends on `structured` and not the reverse; `SymmetrySpec.structure`
  takes an optional `KPoints` and attaches the axis.
- Storage follows the ring of a block. A real or quaternionic block stores
  the representatives, the lower index of every pair, in ascending order;
  a complex block stores every point. The derived point is the complex
  conjugate for a real block, the Kramers map `Θ X Θ⁻¹` for a square
  quaternionic block and `Θ C` for a rectangular one; the derived beta
  block of paired UHF is `beta(k) = conj(alpha(−k))`.
- A real block is stored as `float64` only when every point of the axis
  is time-reversal invariant (Γ alone, an unshifted mesh of even sizes);
  otherwise it is `complex128`, with the conjugate relation between `k`
  and `−k` as its constraint. The constructor checks the dtype as before.
- `trace` and `trace_product` weight each stored point with its own weight
  and add the conjugate of its value with the weight of the derived
  partner, since `tr(conj(X)) = conj(tr(X))` and
  `tr(Θ X Θ⁻¹) = conj(tr(X))`; derived spin blocks add the total of their
  source, conjugated for a conjugate partner. No derived point is
  materialized for an energy.
- `eigh` diagonalizes each stored point on its own. Θ-symmetry of a
  quaternionic block and Kramers pairing of its eigenvectors, and the
  reality of a real block's density, are checked and imposed at the
  time-reversal invariant points only; elsewhere the partner at `−k`
  carries the symmetry.
- `embed` and `project` require equal k axes. A structural embedding
  expands a half-zone source block to whatever the target block stores,
  so real restricted embedded into complex restricted materializes every
  k. The dense form is the full set, `[kpt, 2n, m]`.
- The square-versus-rectangular rule for quaternionic blocks is a known
  limitation: the derived `−k` block of a square coefficient matrix taken
  through the dense form is wrong. The library never materializes
  coefficient matrices: `eigh`, matrix products and `density_matrix` act
  on the stored points. Separate row and column structures (PLAN.md §5.3)
  are the eventual fix.

## Consequences

- Occupations across k-points (M5) act on `[kpt, mo]` vectors with the
  weights of the axis; until then the occupation policies reject
  structures with a k axis.
- A `KPoints` set must be closed under time reversal to form an axis;
  every mesh is, an explicit list must include the partners.
- The space-group reduction of M6 adds a rotated partner rule to the same
  mechanism, and the dedicated quaternion storage of PLAN.md §5.3 would
  halve the stored half-zone blocks once more.
- Revisit if little-group blocking at high-symmetry points makes block
  shapes differ across k, which the batched storage cannot express.
