# 0031 Correlated tensors over k-points conserve crystal momentum and store one block per orbit of time reversal

Status: accepted 2026-10-28

## Context

ADR-0030 built the block tensors of the correlated methods for molecules:
sectors labelled by spin and irrep, blocks selected by the cancellation of
their charges, derived blocks by the partner map of the structure, and
DF-MP2 written once over them. Part two of M8 extends them to the
k-points of a cell, where the orbital indices carry a crystal momentum
and the two-electron integrals are the fitted tensor with momentum
transfer of ADR-0019. Four things had to be settled: how the k-point of
an index enters the sectors and the conservation rule, including the
shifted meshes whose points are not closed under addition; how the
auxiliary index, which lives in the eigenbasis of the metric of each
transfer, fits in; how the half-zone storage of the real and paired
classes (ADR-0015), which derives the blocks at ``−k`` by time reversal,
carries over to tensors; and in which normalization the orbital-basis
integrals and amplitudes are kept.

## Decision

- **A sector is a block at a point of the full set.** Over a k axis the
  sectors of a `TensorAxis` are the pairs of a block of the structure and
  a point of the full k set, the points of a block adjacent, with a size
  each; the orbitals at a derived point are the time-reversed orbitals of
  its stored partner, as the dense form of the coefficients builds them,
  so the derived blocks of a tensor are consistent with the orbitals they
  refer to.
- **The crystal momentum is an exact integer charge.** `KAxis` carries
  the momenta of a uniform mesh as integers over a common denominator,
  twice the mesh size per direction so that the half shift of a
  Monkhorst–Pack mesh is an integer; a block is allowed when the momenta
  of its ket sectors minus those of its bra sectors vanish modulo the
  denominator, on top of the spin and irrep rules. A set that is not a
  uniform mesh carries no momenta and admits no tensors, and the
  irreducible wedge of a space group (ADR-0024) is rejected for now: the
  point operations would be further symmetries deriving blocks, a later
  extension.
- **The auxiliary index runs over the momentum transfers without
  derivation.** The auxiliary axis of a cell is a complex spatial
  structure over the transfer mesh, one sector per transfer with the
  auxiliary functions in the eigenbasis of the metric at that transfer;
  it enters conjugated and carries the transfer ``q = k2 − k1`` of the
  pair it fits. The eigenvectors of the metric at ``q`` and ``−q`` carry
  independent phases, so the fitted tensor at ``−q`` is not the conjugate
  of the one at ``q`` and no block derives from another along this axis:
  the three-index tensor stores every transfer, and the time-reversal
  symmetry returns on the four-index tensors that have no auxiliary
  index. The signature of the metric eigenvalues multiplies the auxiliary
  index of the tensor between virtual and occupied orbitals
  (``signed=True``), so that ``Σ_P B_ov conj(B_vo)`` is a fitted integral
  as in ADR-0019. For a molecule the auxiliary axis stays a real spatial
  structure, now conjugated too.
- **Symmetries are generators, not one partner rule.** The single partner
  rule of ADR-0030 becomes a set of symmetries of the state that map
  every sector of every axis: the spin flip of a restricted structure (a
  copy) and time reversal (a conjugate), which maps a sector to its
  partner point ``−k`` in the block the structure derives there, a real
  or quaternionic block itself and a paired block its conjugate partner,
  and without a k axis reduces to the conjugate partner map of a paired
  structure. A symmetry applies to a tensor when every axis has an image
  under it, a complex axis without partners having none; the symmetries
  commute, the orbit of a key has up to four members, and the
  lexicographically smallest is stored. A restricted real state over a
  mesh with a partner pair stores one block in four.
- **Supercell normalization.** The fitted tensor enters the Bloch
  orbital basis divided by ``√N_k``, so that the orbital-basis integrals
  are those of the supercell of the mesh and the correlated equations of
  a molecule apply unchanged with the composite index of an orbital and
  its point; the energy of the supercell divides by ``N_k`` per cell. The
  amplitudes are kept in that normalization.
- **MP2 over k-points is the molecular code.** `mp2_from_periodic_fitting`
  transforms the periodic fitting twice, forms the same antisymmetrized
  integrals, denominators and amplitudes as the molecular path and scales
  the energy per cell; `df_mp2` takes a `Cell` with the k-points of the
  reference. A reference whose occupied orbitals differ between the
  points of a block, a metal, is rejected, since the orbital spaces hold
  one count per block.

## Consequences

- All eight classes over k-points share the molecular MP2 code; the
  energy per cell of a helium pair over three k-points equals that of the
  supercell at Γ to 1e-14 and PySCF's `KMP2` to 1e-14 at tight
  convergence.
- Storage: a three-index tensor holds every pair of points (``N_k²``
  blocks per spin), a four-index tensor one block per orbit of the
  symmetries among the ``N_k³`` momentum-conserving quadruples; the
  contraction's Python loop over block pairs grows as ``N_k³`` times the
  spin and irrep blocks.
- Later: the point operations of a space group as further generators
  over the wedge, batching and the path optimizer of ADR-0030, k-point
  CCSD from the same tensors.
