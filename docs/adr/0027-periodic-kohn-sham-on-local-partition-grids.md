# ADR-0027: Periodic Kohn–Sham integrates on atom-centred grids with a local partition

Status: accepted, 2026-10-24. Completes M7 for the restricted and
unrestricted classes of a cell.

## Context

The Kohn–Sham term of ADR-0026 served molecules. A cell needs three more
things: a quadrature that integrates a cell-periodic function over one
cell, the basis functions on the grid as Bloch sums over the lattice
images, and the densities and potentials of the k-point density matrices.

For the grid there were three candidates. PySCF's uniform grids are out:
an all-electron basis on silicon asks for a mesh of 2263³ points. PySCF's
periodic Becke grids keep the atomic grids of every atom whose grid
reaches into the unit cell, truncate them to the cell, and partition the
kept points among those atoms with Becke's step and no atomic size
adjustment. The integrand is then discontinuous at the cell faces, and
the quadrature converges slowly: on silicon in def2-SVP the electron
count is off by 1.5e-3 at level 3, 3.3e-4 at level 5 and 7.4e-5 at
level 7, and the energy moves by 1.5e-4 and 3.5e-5 Hartree between the
levels. The third candidate is a periodic partition of unity: with cell
functions ``p_{A,T}`` of every atom ``A`` and image ``T`` summing to one
everywhere, ``∫_cell f = Σ_{A ∈ cell} ∫ p_A f`` over all space, one
atomic quadrature per atom of the cell, exactly as for a molecule.

Two facts decided how to build that partition. First, Becke's step never
reaches one, so a partition over the centres within a finite radius of an
atom is not the same partition for the neighbouring atom, and the
partition of unity breaks between them: with the centres within 25 Bohr
the volume of the silicon cell comes out 0.1 Bohr³ too large, and the
error falls only slowly with the radius. The step of Stratmann, Scuseria
and Frisch (Chem. Phys. Lett. 257, 213 (1996)) is exactly one for
``ν ≤ −0.64`` and exactly zero for ``ν ≥ 0.64``; with it the partition
converges with the radius, is the same at 30 and 40 Bohr as at 20, and
reproduces the volume of the cell to 1e-7 relative. Second, in a crystal
the radial shells near the nearest-neighbour distance pass through the
neighbours' cells, where an atom's weight is shared with several
neighbours and fragments into ridges between the regions it loses
outright. The standard Lebedev rules resolve those ridges poorly: on
silicon the 434-point rule of level 3 integrates the shells between 0.7
and 1.2 of the nearest-neighbour distance with an angular error of
0.22 Bohr³ out of 135, which the radial quadrature does not share (its
error is 1e-5). The error falls to 8.8e-3 with 1202 points, 9.3e-4 with
2030 and below 1e-5 with 5810, while the shells inside and outside that
window are integrated to 1e-6 by the standard rules. The electron count
of the plain level-3 grid is off by 3.5e-3, and raising the level helps
as slowly as it does for PySCF (6.7e-4 at level 5, 1.4e-4 at level 7).

## Decision

- **A cell's grid partitions space among the atoms and their lattice
  images with the Stratmann step.** `build_grid` of a `Cell` keeps the
  atomic grids of ADR-0026 for the atoms of the cell and weights each by
  its normalized cell function among the atoms and images within
  `GridSpec.partition_radius` (25 Bohr) of it, with Treutler's atomic
  size adjustment. Points whose weight is exactly zero are dropped, which
  removes most of every atomic grid's far region. The step is a choice of
  `GridSpec.partition`: molecules keep Becke's, the construction of
  ADR-0026 that reproduces PySCF's grids point by point, and may take the
  Stratmann step; cells take the Stratmann step unless Becke's is asked
  for, which is then a truncated partition, slower by the number of
  centres squared.
- **The shared shells take a larger angular rule.** From 0.6 of an atom's
  nearest-neighbour distance outwards (`SHARED_SHELL_START`) the shells
  take the Lebedev rule of the level's table, 1202 points at levels 0 to
  2, 2030 at level 3, 2702, 3470 and 4334 at levels 4 to 6, 5810 from
  level 7, or `GridSpec.shared_angular_points`. On silicon at level 3 this
  gives 67 000 points instead of 28 000, the electron count to 1.6e-5 and
  the exchange–correlation energy to 3e-6 Hartree of the converged value,
  against 3.5e-3 and 6e-4 without the rule and 1.5e-3 and 1.5e-4 for
  PySCF's grid of the same level.
- **The partition weights of many centres come from the compiled layer.**
  `partition_weights` accumulates the cell functions from the nearest
  centre outwards, so that with the Stratmann step a far centre is
  settled by its first factor; the array-code `becke_partition` stays the
  molecular path and the reference, and the one the JAX backend of M9 can
  differentiate for the weight derivatives of M11.
- **Bloch sums of the basis functions come from the compiled layer.**
  `bloch_basis_values` evaluates ``Σ_T exp(i k·T) φ(r − T)`` and its
  derivatives as ``[kpt, point, component, ao]``, image by image with the
  per-shell cutoff radii of ADR-0026 applied to every image. Python
  chooses the images, every lattice vector at which some shell's image
  reaches some grid point (`reaching_images`, now bounded in memory by
  taking its candidates in chunks), and supplies the phases, so that one
  image list serves every batch; a batch of a cell holds `batch_size`
  points divided by the number of k-points.
- **The exchange–correlation term serves cells through the full k set.**
  With ``M = conj(Φ D)`` at every k-point, ``ρ = Σ_k w_k Re Σ_u Φ_u M_u``
  and likewise for the gradients and kinetic energy densities, which for a
  molecule reduce to the real formulas of ADR-0026; the potential
  ``V(k) = Φ† A + A† Φ`` is formed at every k-point and reduced onto the
  stored points of the structure, half zone or irreducible wedge, as the
  Coulomb matrix is (ADR-0020, ADR-0024). The index convention is that of
  the Coulomb matrix: the coefficient of ``conj(φ_u^k) φ_v^k`` in the
  density is ``D[v, u](k)``. `Hamiltonian.from_cell` takes the term, and
  `kohn_sham` takes a cell with the k-points and the auxiliary basis of
  `hartree_fock`. The fitted exchange of a hybrid carries the Madelung
  correction of ADR-0017; the exchange of a pure functional is skipped.
  Range-separated hybrids are refused for cells until the fitted
  integrals with the long-range kernel exist for a lattice.

## Consequences

- On one and the same grid Hafnium and PySCF agree to 5e-13 Hartree for
  LDA, PBE, TPSS and B3LYP in the restricted class and PBE in the
  unrestricted one, on three-dimensional cells and a slab, so the
  reference tests run at the energy tolerance of the molecular ones with
  PySCF's grid points and weights passed in. On its own grid Hafnium
  differs from PySCF's level-3 energies by 1e-5 Hartree on the helium
  cells, PySCF's truncation error, and its electron counts are off by
  1e-6 where PySCF's are off by 7e-5.
- The quadrature of a cell costs more than a molecule's: 2.4 times the
  points for the shared shells, and the Bloch sums over about a hundred
  images within reach of every point, which dominate the time of the term
  for silicon.
- The energy per cell at a 2×1×1 mesh equals that of the doubled cell at
  Γ to 1e-15 Hartree, and the irreducible wedge of a space group gives the
  full-set energy to the same precision; the partition is
  translation-covariant and the k-point reduction exact.
- Remaining for cells: range-separated hybrids, VV10, the noncollinear
  evaluation for the generalized classes, caching the Bloch sums across
  SCF iterations, and the derivatives of the grid weights.
