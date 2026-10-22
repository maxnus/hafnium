# ADR-0023: Irrep sectors of the structured matrices

Status: accepted, 2026-10-22.

## Context

The second part of M6 makes the SCF use the point group: the state is
blocked over the irreps of the Abelian group, electrons are occupied per
irrep, and the exit criterion asks that symmetry-constrained results equal
unconstrained results for symmetric molecules and that the irreps of the
orbitals agree with PySCF. ADR-0022 provides the group in its standard
frame and the SALCs of each irrep as an orthogonal change of basis under
which every operator that commutes with the group is block-diagonal.

PLAN.md §5.2 says how this composes with the Fukutome classes: each spin
block splits into one block per irrep in the SALC basis, with no change
of ring or weight. §5.3 adds that a structure knows the mapping from the
atomic basis to the sector basis, and §5.4 that the Coulomb and exchange
matrices are built in the atomic basis and transformed. Open were:

- how the sectors enter `BlockStructure`, whose blocks so far are the
  spin pattern over an optional k axis;
- where the SALC coefficients live, since a structure is static metadata
  while the coefficients are arrays;
- what the dense form of a matrix over sectors is, and how embedding and
  projection cross the boundary between a structure with sectors and one
  without;
- whether the one- and two-electron matrices are verified to commute with
  the group or projected onto the sectors;
- how the electrons of a spin block are shared over its sectors, and how
  a configuration is fixed per irrep;
- how a calculation asks for the blocking, given that the sectors depend
  on the basis while `SymmetrySpec` is declarative.

## Decision

- **Irrep sectors multiply the spin pattern.** `IrrepSectors` in the
  `structured` package holds the name of the group, the Mulliken labels
  of the irreps the basis spans and the orthonormal SALC coefficients
  `U_Γ [ao, salc]` of each; irreps without SALCs have no sector. A
  `BlockStructure` with `irreps` repeats its spin pattern once per
  sector, in the order of the irreps, each block carrying the index of
  its sector as `Block.irrep` and acting on that sector's SALCs; rings,
  weights and partner rules are unchanged, and the constructor checks
  that every sector repeats the pattern of the first. `sector_structure`
  returns the pattern, `BlockMatrix.sectors` and `from_sectors` split and
  join the per-sector matrices, and most operations are block-wise as
  before. Sectors and a k axis are exclusive: the point group of a
  crystal reduces the k-point set instead (part three of M6). Two sector
  sets are equal when their group, labels and coefficients are, so
  structures built from equal adaptations compare equal; the hash uses
  the labels and dimensions.
- **The dense form stays in the atomic basis.** The spin-orbital form of
  a matrix over sectors is that of the same matrix without them: the
  blocks of an operator are assembled as `Σ_Γ U_Γ X_Γ U_Γᵀ` and the blocks
  of orbital coefficients placed side by side as `U_Γ C_Γ`, sector by
  sector within each spin block, which is also the order of the entries
  of a `BlockVector`. A square coefficient block cannot be told from an
  operator, so `to_dense`, `materialized` and `spin_blocks` take
  `orbitals=True` for coefficients. From the dense form only square
  operators enter a structure with sectors, by projection or, with the
  check, after verifying that the matrix couples no two irreps; the
  columns of orbital coefficients carry no irrep to assign them by.
  Embedding and projection between the Fukutome classes with the same
  sectors are block-wise; across different sectors they go through the
  dense form, for operators.
- **The one- and two-electron matrices are projected onto the sectors.**
  `BlockMatrix.from_spatial(..., irreps=...)` keeps the blocks
  `U_Γᵀ X U_Γ` and drops the couplings between irreps. The Hamiltonian
  builds the overlap, the core Hamiltonian and the orthogonalizer of each
  sector that way, the Coulomb matrix from the atomic-basis total density
  `Σ_Γ U_Γ D_Γ U_Γᵀ`, and the exchange matrix of each spin block from its
  atomic-basis matrix, projected back; the atomic density guess is
  projected too, which loses nothing for a symmetric molecule. The
  dropped couplings vanish for a molecule exactly in the standard frame
  and are bounded by the geometry tolerance of the symmetry adaptation
  otherwise (`1e-5` Bohr, ADR-0022), so the energy of a slightly
  asymmetric input is that of the symmetrized Hamiltonian, second order
  in the asymmetry. PySCF behaves the same way: its symmetry-adapted
  classes transform the Fock matrix with the SALCs and never check that
  it commutes with the group.
- **Electrons fill the sectors of a spin block together, or as fixed per
  irrep.** `electrons_per_block` gives the count of each spin block of
  the pattern; the aufbau policy fills the orbitals of all sectors of a
  spin block in order of energy, so a degenerate shell that spans several
  irreps is filled by energy as before. `AufbauOccupation.irrep_electrons`
  fixes the electrons of named irreps as PySCF's `irrep_nelec` does: a
  total, which a restricted or paired class splits equally and an
  unrestricted class with the odd electron in alpha, or `(n_alpha,
  n_beta)` for a collinear class; a generalized state takes totals only.
  The named sectors fill their own lowest orbitals, and the remaining
  electrons of each spin share the other sectors.
- **The specification names the group, the Hamiltonian carries the
  adaptation.** `SymmetrySpec.point_group` (`from_name(...,
  point_group=)`, `with_point_group`) imposes an Abelian group;
  `Hamiltonian.from_molecule(..., point_group=)` builds the
  `SymmetryAdaptation`, which refuses a molecule outside the group's
  standard frame, and `Hamiltonian.structure(spec)` multiplies the
  spec's structure by `adaptation.irreps()`. `solve_scf` and
  `hartree_fock` use that structure, so
  `hartree_fock(engine, basis, molecule, symmetry=SymmetrySpec.from_name("UHF", point_group="C2v"))`
  runs the blocked calculation of an oriented molecule, and a lower group
  in the same frame (`C2` for water) is accepted. The orbitals of a block
  carry their irrep by construction, `BlockStructure.irrep_labels` names
  them, and `ScfResult.electrons_per_irrep` reports the occupation of
  every sector.

## Consequences

- All eight Fukutome classes blocked over C2v reproduce the energy and
  density of the unconstrained calculation for water; the energies,
  electrons per irrep, orbital energies and orbital labels agree with
  PySCF's symmetry-adapted RHF, UHF and GHF for eleven cases, among them
  closed- and open-shell configurations fixed per irrep, the degenerate
  shells of methane, benzene and nitrogen, and the descent of ammonia to
  Cs. A blocked density embedded into the plain structure is stationary
  there, and a plain density of a symmetric state enters the blocked
  structure losslessly, which is what stability analysis and
  symmetry-breaking searches over spatial symmetry use.
- The Coulomb and exchange contractions stay in the atomic basis, so the
  blocking saves the diagonalization and the DIIS algebra per sector and
  yields the labels for free; the integral-level reduction (petite lists,
  direct products of the SALCs) stays on the later list of PLAN.md.
- The structure, static metadata of the pytrees to come, now holds
  arrays. Equality compares them and the hash does not; the JAX milestone
  decides whether the coefficients move to a side table keyed by that
  hash.
- The `orbitals` flag exists because one structure serves rows and
  columns alike. The `BlockMatrix(row_structure, column_structure,
  blocks)` of PLAN.md §5.3 would remove it, and is the natural extension
  when `BlockTensor` arrives.
- Irreps the basis does not span have no sector, so the labels of a
  structure are a subset of the group's; a configuration fixing electrons
  in such an irrep is an input error.
- Non-Abelian groups remain reported by name only (ADR-0022).
