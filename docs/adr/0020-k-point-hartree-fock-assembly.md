# ADR-0020: k-point Hartree–Fock through the structured matrices

Status: accepted, 2026-10-22.

## Context

With the Bloch one-electron matrices (ADR-0016, ADR-0018), the fitted
tensor over k-point pairs (ADR-0019) and the exchange correction
(ADR-0017) in place, the self-consistent field of a cell needed four
things the molecular driver of ADR-0010 did not have: Coulomb and exchange
matrices at every k-point from a density stored at the half zone of a
real class, the one-electron matrices and the orthogonalizer of the atomic
basis in the structure of the class, occupations that fill bands across
k-points with their weights, and spin expectation values per cell. The
molecular code path had to stay the one-k-point case of the same code
(ADR-0007), and the eight Fukutome classes had to keep working over k.

The two-electron terms couple k-points: the Coulomb matrix sums the
density over every k-point and the exchange matrix at `k` needs the
density at every `k'`, both over the full set, while the real and paired
classes store a half zone and derive the partner at `−k` by conjugation
or the Kramers map (ADR-0015). The canonical orthogonalizer at `−k` is not
the conjugate of the one at `k`, since the eigenvectors of `S(k)` and
`S(−k)` carry independent phases.

## Decision

- **The two-body builder works on full-set arrays.** `CoulombExchange`
  implementations take `[ao, ao]` for a molecule and `[kpt, ao, ao]` over
  the full k set for a cell; `coulomb_exchange_matrices` materializes the
  derived partners of the density (the spin trace for the Coulomb matrix,
  each stored block for the exchange matrix), calls the builder, and
  stores the results back at the stored k-points in the ring of each
  block. `KPointDensityFittedCoulombExchange` contracts the fitted tensor
  with the k-point weights,
  `J(k) = Σ_P B[k, k, P] Σ_k' w_k' ⟨B[k', k', P], D(k')⟩` and
  `K(k) = Σ_k' w_k' Σ_P B[k, k', P] D(k') B[k, k', P]† + ξ S(k) D(k) S(k)`,
  the Madelung term of ADR-0017 included.
- **Spatial periodic matrices enter a structure through
  `BlockMatrix.from_spatial`**, which stores the half zone of a real ring
  after checking that the matrix at `−k` is the conjugate of the matrix at
  `k`, and casts to the real dtype where every point is invariant; the
  overlap, the core Hamiltonian, the Coulomb matrix and the atomic-density
  guess (the same density at every k-point) all go this way, so the
  Hamiltonian does not branch on the k axis.
- **The orthogonalizer is built at the stored points, over the block's
  storage ring.** The overlap enters through `from_spatial`, which makes
  it real where every point is invariant, so the eigendecomposition and
  the orthogonalizer are real there; the complex eigendecomposition of a
  real matrix carries an arbitrary phase per eigenvector, and its real
  part is not an orthogonalizer. The conjugate of the orthogonalizer at
  `k`, which a real structure derives at `−k`, orthogonalizes
  `S(−k) = conj(S(k))` just as well, so no check is made;
  `canonical_orthogonalizer` works per k-point and keeps the smallest
  retained count any k-point allows, so that one block shape serves
  every k-point.
- **Occupations fill levels across k-points by weight.** The levels of
  every stored k-point are ordered by energy and filled with the weight
  of the point, a stored half-zone point carrying its partner's weight
  too, until the block's count per cell is reached. The aufbau policy
  requires the count to land on a whole level and raises otherwise; the
  averaged policy shares electrons equally within degenerate groups,
  which span the k-points. Fractional occupation of a band crossing the
  Fermi level (smearing) is not part of version 1.
- **The k-point set lives on the symmetry specification.** The
  translational sectors are symmetry data like the spin structure
  (PLAN.md §5.9), so `SymmetrySpec` carries `kpoints`, `with_kpoints`
  attaches them and `structure()` builds the k axis from them; `solve_scf`
  checks them against the Hamiltonian's, and `hartree_fock` serves a cell
  with the same signature as a molecule.
- **Spin expectation values are per cell, weighted over the full k set**;
  the two-body term of `⟨S²⟩` has no cross terms between k-points because
  orbitals at different k-points are orthogonal. PySCF's k-point classes
  report the supercell of the mesh instead,
  `n_kpt (⟨S²⟩ − |⟨S⟩|²) + n_kpt² |⟨S⟩|²`, and take `cell.spin` as
  `n_α − n_β` over the whole mesh; the reference tests map between the
  conventions.

## Consequences

- The closed-shell energy of a cell is the same in all eight classes, the
  half-zone and full-zone storage agree, a supercell at Γ reproduces the
  primitive cell with the matching mesh per cell, and the energy is
  independent of the range separation; KRHF, KUHF and KGHF energies,
  orbital energies and spin expectation values match PySCF's density-fitted
  k-point SCF with the Ewald exchange correction.
- Every Fock build expands the density to the full k set and reduces the
  results, which costs the dense form of the density per iteration; for
  the systems of version 1 that is negligible next to the contractions.
- A metallic band crossing the Fermi level at the mesh makes the aufbau
  policy raise; smearing and fractional occupations across k are the
  later extension named in PLAN.md.
- Revisit the orthogonalizer when the space-group reduction (M6) derives
  blocks at rotated k-points, which need the rotated orthogonalizer rather
  than a conjugate.
