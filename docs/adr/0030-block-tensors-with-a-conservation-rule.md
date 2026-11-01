# 0030 Block tensors conserve spin and irrep charges, and DF-MP2 is written once over them

Status: accepted 2026-10-28

## Context

The correlated methods of M8 need quantities with several orbital indices:
the fitted integrals in the orbital basis, antisymmetrized integrals and
amplitudes. PLAN.md §5.5 asks for them as block-sparse tensors whose
blocks are labelled by tuples of sectors obeying a conservation rule, so
that the restricted, unrestricted and generalized cases, with point-group
sectors, are served by one spin-orbital formulation; the one-body
quantities of the mean field are already `BlockMatrix` objects, which are
block-diagonal and need no rule. DF-MP2 is the first consumer. The design
had to fix five things: how the sectors of a tensor index are labelled and
which blocks the symmetry allows; which blocks a tensor stores when its
structure derives blocks from others (the restricted and paired classes);
how the auxiliary index of density fitting fits in; how contractions are
expressed; and how MP2 is written once for every class.

## Decision

- **Sectors are the blocks of a `BlockStructure`.** A `TensorAxis` runs
  over the blocks of a structure, stored and derived alike, with a size
  per sector (the orbitals of that block in the space the axis covers)
  and a flag saying whether the index enters conjugated, as the bra
  index of an integral does. The sectors of an orbital index are the
  blocks of the state's structure; the auxiliary index runs over a
  spatial, real structure.
- **A block is allowed when its charges cancel.** A sector carries the
  spin charge `+1` (alpha), `−1` (beta) or `0` (spinor, spatial) and the
  characters of its irrep. A block of a tensor is allowed when the spin
  charges, counted negatively on conjugated axes, sum to zero and the
  characters of all its irreps multiply to the totally symmetric irrep.
  `IrrepSectors` therefore carries the characters of its irreps, which
  the symmetry adaptation takes from the group; characters rather than
  parity vectors or a fusion table, because they are unique per irrep in
  every Abelian group (several parities span one irrep of C2v) and the
  product of several irreps stays well defined when an intermediate
  product is an irrep the basis does not span. Blocks with an empty
  sector are never stored either.
- **Derived blocks follow from the simultaneous partner map.** Mapping
  every index of a block to its partner sector at once (alpha to beta and
  back) is a symmetry of the state, so the image of a block is its copy
  or its complex conjugate by the partner rule of the structure; the
  tensor stores the lexicographically smaller key of every pair. The map
  applied to one index at a time, which would make every spin block of a
  spin-free tensor equal, is not a symmetry of antisymmetrized tensors
  (`t2` of the restricted class differs between its same-spin and
  opposite-spin blocks), so only the simultaneous map is used. All axes
  of a tensor share one partner rule, and with the conjugate rule every
  axis without partners holds real blocks, since the rule conjugates
  them too.
- **The auxiliary basis is symmetry-adapted.** With a state over irrep
  sectors the auxiliary basis is adapted to the same group, so that the
  auxiliary index carries an irrep and the three-index tensor obeys the
  same rule as every other; an unsectored index that couples to every
  irrep would need a special case in the rule. Without sectors the
  auxiliary index is one sector.
- **A contraction pairs a conjugated with an unconjugated index.** The
  sum over a complete set of orbitals pairs a bra with a ket index, and
  then the charge rule of the result follows from those of the operands.
  `contract("Pia,Pbj->iajb", first, second)` takes two operands in
  einsum notation, sums the products of the blocks whose contracted
  sectors coincide with the dense einsum, and computes the stored blocks
  of the result only; chains are nested calls, and a path optimizer over
  several operands is left for the coupled-cluster programs. The
  auxiliary sum of `(ia|jb) = Σ_P B[P, i, a] B[P, j, b]` is written as
  `Σ_P B[P, i, a] conj(B'[P, b, j])` with the tensor `B'` between virtual
  and occupied orbitals, `B[P, j, b] = conj(B'[P, b, j])` for the real
  auxiliary functions of a molecule, which costs a second transformation
  of the fitted tensor and matches the form the periodic tensor takes,
  `Σ_P s_P B[P, u, v] conj(B[P, x, w])`.
- **Rings are explicit.** A `TensorStructure` names the ring its blocks
  are stored over, which must be complex when any axis holds complex
  blocks and is complex for the result of a contraction when either
  operand is; the scalar of a full contraction over complex tensors is
  complex until its real part is taken.
- **k-points come with part two.** An axis over a k-point set is
  rejected for now; the charge then gains the crystal momentum, the
  auxiliary index the momentum transfer, and the time-reversal partner
  at `−k` joins the partner map.
- **DF-MP2 is one spin-orbital formula.** `E₂ = ¼ Σ |<ij||ab>|² / Δ` with
  `<ij||ab> = (ia|jb) − (ib|ja)` and `Δ = ε_i + ε_j − ε_a − ε_b`, over the
  blocks the rule allows: for a restricted reference the blocks of one
  spin follow from the other, an unrestricted reference has the
  same-spin and opposite-spin blocks of UMP2, a generalized reference
  one spinor block. The amplitudes `t2[i, j, a, b] = <ab||ij> / Δ` of the
  first-order wavefunction are kept, with the virtual indices conjugated
  as in the excitation operator. The full `(ia|jb)` tensor is formed,
  `n_occ² n_vir²` elements; batching over occupied orbitals is a later
  fast path behind the same interface. The reference must occupy every
  orbital by zero or one electron, so an averaged occupation is rejected.
  `OrbitalSpaces` splits the canonical orbitals of an `ScfResult`, for
  which the spinor-block check of `BlockMatrix` is relaxed to the rows,
  since the occupied coefficients of a doublet have an odd column count.
- **Layout.** What the correlated methods share, the orbital spaces, the
  fitted integrals in the orbital basis and the energy denominators,
  lives in `hafnium/orbitals/`; MP2 in `hafnium/mp/` as PLAN.md §4
  foresees.

## Consequences

- The eight classes and the point-group sectors are served by one MP2
  implementation of a few lines over the tensors; the energies agree
  with PySCF's DF-MP2 to 1e-11 Hartree for restricted, unrestricted and
  generalized references converged to a gradient of 1e-9. The
  second-order energy is first order in the orbital error, so a reference
  converged to the default gradient of 1e-6 carries an uncertainty of
  about 1e-9 Hartree in it.
- The contraction loops over pairs of blocks in Python with one dense
  einsum each, which is cheap next to the arithmetic for the hundreds to
  thousands of blocks of a four-index tensor in D2h; it exploits the
  irrep sparsity of the result but not a path through several operands,
  and it materializes derived blocks on request rather than caching
  them.
- Later: k-point MP2 from the same code (part two), the same-spin and
  opposite-spin components for spin-component scaling, batching over
  occupied orbitals, natural orbitals from the amplitudes, and the
  equation generator with a path optimizer for CCSD.
