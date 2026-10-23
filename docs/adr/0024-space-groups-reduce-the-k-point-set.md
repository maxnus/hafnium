# ADR-0024: Space groups reduce the k-point set

Status: accepted, 2026-10-23.

## Context

The third part of M6 brings spatial symmetry to crystals. PLAN.md §5.9
asks that the point operations of the space group shrink the stored
k-point set to the irreducible wedge of the Brillouin zone with weights,
the blocks at the other points following by a unitary rotation of the
Bloch basis, a third partner rule beside copying and conjugation;
ADR-0007 and ADR-0015 reserved the mechanism, ADR-0020 noted that the
orthogonalizer at a rotated point is the rotated one, and §12 left the
detection to spglib if confirmed. Open were:

- whether to detect the space group with spglib or in-house, and how a
  chain or a slab, which spglib does not treat, is handled;
- how a rotation acts on the basis in a general frame, where the
  operations are not the sign operations of ADR-0022 but mix the
  functions of a shell, and how the Bloch phases of the atoms that an
  operation carries into another cell enter;
- how the operations enter `KAxis`, which points each ring stores once
  rotations are combined with time reversal, and how the derived points
  and the weights are defined;
- whether the one- and two-electron matrices at the stored points are
  verified to have the symmetry or projected onto it, as ADR-0023 decided
  for the irrep sectors;
- what happens when a k-point set is not invariant under the group, and
  what becomes of the constraints of the little group of a stored point.

## Decision

- **spglib detects the space group, as an optional dependency.**
  `detect_space_group(cell, tolerance=1e-5)` passes the lattice, the
  fractional positions and the atomic numbers to spglib and returns a
  `SpaceGroup` with the Hermann–Mauguin symbol, the number, the crystal
  class and the operations, each holding the integer matrix and the
  fractional translation in the lattice basis and the Cartesian
  rotation and shift; the Cartesian matrix is taken from the polar
  decomposition of `Aᵀ W (Aᵀ)⁻¹`, so a lattice symmetric within the
  tolerance gives exactly orthogonal operations. A chain or a slab is
  padded along its open directions with vectors `1.37` times its longest
  periodic vector, and only the operations that keep the open directions
  apart from the periodic ones, without translating along them, are
  kept; the names are those of the padded cell. Without spglib
  (`hafnium[symmetry]`) detection raises; molecular point groups stay
  in-house (ADR-0022).
- **The group acts on the Bloch basis by rotated functions and lattice
  phases.** A Wigner matrix of the real solid harmonics,
  `S_lm(R⁻¹ r) = Σ_m' D^l[m', m] S_lm'(r)`, is computed exactly from the
  polynomial table of the harmonics (`solid_harmonic_rotation`), so that
  improper operations and the `x, y, z` order of p shells need no
  special case. `SpaceGroupAdaptation.from_basis(basis, cell, group,
  kpoints)` maps every atom onto its image `g(A)`, records the lattice
  vector `T_A = R r_A + t − r_{g(A)}`, assembles `Π_g [ao, ao]` from the
  Wigner matrices of the shells, and forms the phases
  `exp(−i (Rk)·T_A)` of every function at every point. The operation
  carries the Bloch functions at `k` to those at `Rk` by
  `U_g(k) = Π_g diag(phases)`, so an operator commuting with the group
  satisfies `X(Rk) = U_g(k) X(k) U_g(k)†`, orbital coefficients
  `C(Rk) = U_g(k) C(k)`, and a spinor block is rotated within each spin;
  these are PySCF's `transform_dm` and `transform_mo_coeff` conventions.
  Atoms the group relates must carry equal shells.
- **The operations are an attribute of the k axis.** `KPointOperations`
  in `structured` holds the name of the group, the images `[op][kpt]`
  of the points, the matrices `Π_g` and the phases `[op, kpt, ao]`;
  `KAxis(partners, weights, operations=...)` builds two tables of
  derivations, one per storage rule. Blocks over the real and
  quaternionic rings store the lowest-index representative of every
  orbit of the operations combined with time reversal, blocks over the
  complex ring those of the operations alone; without operations these
  are the half zone and the full set of ADR-0015. Every other point is
  derived from its representative by one operation, followed by the
  antiunitary map of the ring only when no operation alone reaches it,
  so a crystal with inversion never conjugates. The weight of a stored
  point is the total weight of the points derived from it, split into
  the part derived without and with conjugation, whose traces are the
  value and its conjugate, so traces, occupations across k and the
  dense form are unchanged in form. `embed` and `project` between axes
  over the same points with different operations go through the dense
  form and are verified.
- **Projection onto the wedge is the group average.** Reducing a block
  given at every point to its stored points averages, for each stored
  point, over all elements of the group of the axis, each member mapped
  back by `U† X U` and the antiunitary map; elements fixing the point
  symmetrize its block, which at a time-reversal invariant point is the
  real part or the Kramers symmetrization of ADR-0015. With the check
  (`from_dense`, `from_spatial`) the derived points must reproduce the
  input; without it the Hamiltonian projects: the overlap, the core
  Hamiltonian, the orthogonalizer, the atomic guess, and the Coulomb and
  exchange matrices, built over the full set from the derived density
  and averaged back, as ADR-0023 does for the sectors. For a cell
  symmetric to rounding this changes nothing; for a cell symmetric to
  the detection tolerance it is the symmetrized Hamiltonian, and the
  stored Fock matrix is the exact gradient of the energy of the derived
  density. The little group of a stored point is not blocked: its
  constraint is carried by the Fock matrix built from the symmetric
  density, as the reality at time-reversal invariant points was.
- **A k-point set must be invariant; the subgroup is explicit.**
  `SpaceGroup.kpoint_images` raises for a set an operation maps outside
  itself, and `SpaceGroup.restricted_to(kpoints)` returns the subgroup
  that maps the set onto itself, which the user passes on. The
  specification names the group, `SymmetrySpec.with_space_group`,
  beside its k-points; `Hamiltonian.from_cell(..., space_group=)` builds
  the adaptation and `Hamiltonian.structure(spec)` attaches the
  operations to the axis, so
  `hartree_fock(engine, basis, cell, symmetry=spec.with_kpoints(mesh).with_space_group(group), auxiliary=aux)`
  runs over the wedge. A space group and a point group are exclusive.

## Consequences

- All eight Fukutome classes over the wedge reproduce the energies and
  densities of the full-zone calculations for the diamond structure
  (non-symmorphic, three stars of the 2×2×2 mesh), a hexagonal slab
  (three stars of the 3×3 mesh, where the half zone alone keeps five)
  and a chain; the wedge density embedded into the full-zone structure
  is stationary there, and the full-zone density projects onto it. The
  energies, wedge sizes, weights and orbital energies agree with PySCF's
  symmetric KRHF. The Bloch overlap and core Hamiltonian at a rotated
  point equal the rotated matrices to `1e-14`, with d functions.
- The two-electron terms are still evaluated over the full set and the
  exchange matrix at every point before projection, as the one-electron
  integrals are, so the reduction saves the diagonalization, the DIIS
  algebra and the storage but not the contractions. Building the fitted
  tensor over wedge × full and the exchange at the wedge alone, and the
  one-electron integrals at the wedge, is the later optimization.
- The point operations hold arrays on the structure, like the SALCs of
  ADR-0023; equality compares them and the hash does not. The matrices
  and phases must represent the group, `U_g(hk) U_h(k) = U_{gh}(k)` and
  `U_g(−k) = conj(U_g(k))`, for the group average to be a projection;
  the closure of the images is checked, the rest holds by construction.
- The elementwise conjugate and transpose of a matrix over a wedge are
  not members of its structure, since the conjugate at `Rk` carries the
  phases of `−k`; `BlockMatrix.conj` and `transpose` raise there, while
  the adjoint, products, sums and traces are block-wise as before.
- The operations of a chain or a slab are those of a padded cell, so
  their names are three-dimensional symbols; PySCF's own search omits
  the operations that flip the normal of a slab, which do not change the
  wedge.
- The Wigner matrices lift the standard-frame restriction of ADR-0022 in
  principle; the molecular point groups keep it until a need arises.
- Revisit for little-group blocking at high-symmetry points, for the
  symmetrization of a cell given to the detection tolerance, and when
  `BlockTensor` needs the operations on the two-electron tensor.
