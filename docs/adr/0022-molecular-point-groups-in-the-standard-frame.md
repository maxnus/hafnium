# ADR-0022: Molecular point groups in the standard frame

Status: accepted, 2026-10-22.

## Context

M6 adds spatial symmetry. Version 1 imposes the Abelian point groups,
D2h and its subgroups, and detects the full point group of a molecule
for reporting and for the orientation (PLAN.md §5.8). The first part of
the milestone had to settle how the group is detected, how its frame is
chosen, how it acts on the basis functions, and how orbitals are labelled.

- The frame fixes the labels. Which axis is `z` and which is `x`
  distinguishes `B1` from `B2` in C2v and `B1`, `B2`, `B3` in D2 and D2h.
  Mulliken's recommendations cover the principal axis and planar
  molecules but leave the descent from a higher group and the non-planar
  cases open, and the exit criterion of M6 asks for irrep labels that
  agree with PySCF, whose `pyscf.symm.geom` makes its own choices.
- A point operation in a general frame acts on a shell of real solid
  harmonics by a Wigner rotation matrix. In the standard frame of an
  Abelian group every operation is a diagonal matrix of signs, under which
  each real solid harmonic is an eigenfunction.
- PySCF reorients the molecule internally and expresses its results in
  the input frame through those rotation matrices. The alternative is to
  make the orientation an explicit step the user takes.
- Orbitals of an unconstrained calculation that are degenerate across
  irreps come out as arbitrary mixtures, which a per-orbital label cannot
  describe.

## Decision

- **The Abelian groups are tables of sign operations and parity irreps.**
  `Operation` holds the eight operations of D2h as diagonals of signs,
  named by Cotton's symbols, with reflections named by their plane
  (`σ(xy)` is `z → −z`). `PointGroup.named` builds D2h and its seven
  subgroups with the operations in Cotton's order; each irrep carries a
  Mulliken label in Cotton's order and a parity vector, and its character
  under an operation is the product of the operation's signs raised to the
  parities, so that the direct product of two irreps is the irrep of the
  sum of their parities modulo two. The mirror plane of Cs is `xy`, the
  twofold axis of C2, C2h and C2v is `z`, and the planes of C2v are `xz`
  and `yz`. These agree with PySCF's tables up to the order of the
  operations.
- **The full group is found from its symmetry elements.** Atoms are
  equivalent when they are the same element; the basis is checked later.
  Candidate directions are the axes of the charge tensor
  `Σ_A Z_A r_A r_Aᵀ`, the directions of the atoms, the sums, differences
  and cross products of pairs of equivalent atoms, and the normals of the
  planes through three equivalent atoms. Each candidate is tested as a
  rotation axis of every order the angles between equivalent atoms allow,
  as a mirror normal, and, if it is an axis of order `n`, as an `S_2n`
  axis; the inversion is tested once. Linear molecules and single atoms
  are recognized first. The standard flowchart then names the group:
  cubic and icosahedral groups from several axes of order three or more,
  the axial groups from the principal axis, its perpendicular twofold
  axes, horizontal and vertical mirrors and improper axis, and Cs, Ci, C1
  from what remains. A test passes when every atom lands within the
  tolerance, `1e-5` Bohr by default, of an atom of the same element.
- **The frame rules follow PySCF, and the user orients the molecule.** The
  principal axis is `z`. For the asymmetric Abelian groups a twofold axis
  or mirror normal becomes `z` by a cyclic permutation of the ascending
  axes of the charge tensor; a planar C2v molecule is turned into the
  `yz` plane; a planar D2h molecule lies in the `yz` plane with `z` the
  in-plane axis more atoms are displaced along, a non-planar one has its
  axes in ascending extent; the axes of a D2 or a cubic frame are the
  symmetry axes closest to the input axes. When the full group is axial
  of order `n ≥ 3`, `x` is the perpendicular twofold axis through the
  first atom that lies on one, else through the first pair's midpoint, or
  for C_nv the normal of the vertical mirror closest to the axis it will
  become, or else the first input axis off `z`; for odd `n` the group
  descends by making that perpendicular axis or normal `z` (D_nh to C2v,
  D_nd to C2h, D_n to C2, C_nv to Cs), for even `n` the principal axis
  stays `z` (D_nh to D2h, D_nd and D_n to D2, C_nh to C2h, C_nv to C2v,
  C_n and S_4k to C2); C_nh of odd `n` gives Cs, S_4k+2 gives Ci. The
  signs of the axes do not affect labels and are chosen to make the
  rotation into the frame proper and closest to the identity, so that an
  oriented molecule is left where it is. A C1 molecule keeps its frame
  and origin. `PointGroupDetection.orient` applies the frame to a
  molecule; nothing in the library reorients a molecule on its own, and
  `SymmetryAdaptation.from_basis` refuses a system that is not in the
  standard frame of its group about the origin.
- **The group acts on the basis by signed permutations.** In the standard
  frame an operation maps a function on atom `A` to the same function on
  `g(A)` times the character of its parity, `solid_harmonic_parity`,
  which is `(m mod 2, 0, (l − m) mod 2)` for the cosine-type harmonic of
  order `m`, `((m + 1) mod 2, 1, (l − m) mod 2)` for the sine-type one and
  `(0, 0, l mod 2)` for `m = 0`; the `x, y, z` order of p shells (ADR-0009)
  needs no special case. Atoms related by the group must carry equal
  shells. The SALCs of an irrep are the normalized projections of the
  functions of one atom per orbit, orthonormal in the coefficient space
  rather than the overlap metric, so every operator commuting with the
  group is block-diagonal over the irreps after the orthogonal
  transformation and the overlap of each block still needs its own
  orthogonalizer.
- **Orbitals are labelled by irrep weights, degenerate sets by a
  projector.** The weight of an irrep in an orbital is the norm of its
  component in that irrep's SALCs in the overlap metric; an orbital is
  labelled by the irrep that carries all but `1e-8` of its weight and
  rejected otherwise. Within a set of degenerate orbitals the operator
  `Σ_Γ Γ P_Γ` of the irrep projectors is diagonalized; its eigenvectors
  are the pure orbitals and its eigenvalues their irrep indices.

How the irreps enter `BlockStructure` and the SCF, and how space groups
reduce the k-point set, are decided in the following parts of M6.

## Consequences

- Full groups, Abelian subgroups, frames, per-irrep SALC counts and
  orbital labels agree with PySCF for the twenty test molecules, which
  cover every Abelian group and the descent from C3v, Td, D6h, D2d, D∞h,
  C∞v, Oh, D3h, D3d, C4v and the full rotation group. The frames differ
  where PySCF's own choice is not a rule: for cubic groups it swaps `x`
  and `y` instead of flipping a sign when its candidate triple is
  left-handed, for C4v either class of vertical mirrors may become `xz`,
  and for linear molecules `x` and `y` are arbitrary; each of these only
  permutes the members of degenerate sets.
- Results of a symmetric calculation refer to the standard frame, since
  the user orients the molecule first; input coordinates are never
  modified silently. Wigner rotation matrices for a general frame are
  needed for crystals, whose lattice fixes the frame, and would also lift
  this restriction for molecules.
- Detection costs a number of candidate directions quadratic in the size
  of the largest equivalence class (capped at twenty-four atoms per class
  for the pair and triple candidates) times a position match linear in
  the number of atoms, which is immediate for molecules of a few hundred
  atoms.
- Non-Abelian groups are reported by name only; their degenerate irreps
  split into Abelian ones, as in PySCF. Non-Abelian blocking stays on the
  later list of PLAN.md.
