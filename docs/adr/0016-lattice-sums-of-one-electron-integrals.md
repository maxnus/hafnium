# ADR-0016: Lattice sums of one-electron integrals

Status: accepted, 2026-10-22.

## Context

ADR-0007 fixed the Bloch sums `φ_μ^k(r) = Σ_T exp(i k·T) φ_μ(r − T)` and
made every kernel of the integral engine take a list of ket translations,
so that the overlap and kinetic matrices of a cell at `k` are the lattice
sums `M(k) = Σ_T exp(i k·T) M_T` over the matrices the engine returns for
the images `T`. Three things had to be decided before the first periodic
matrices existed: which images enter the sum, what the periodic functions
return, and how they relate to the molecular entry points.

Options for the images: a fixed number of images per direction, a cutoff
radius chosen by the user, or a cutoff derived from the basis set at a
precision, as PySCF derives `rcut`. Options for the result: a
`BlockMatrix` carrying the k axis of a symmetry class (ADR-0015), with the
half zone stored and real blocks at the invariant points, or plain arrays
over the full k-point set. Options for the entry points: overloads of the
molecular `overlap` and `kinetic` accepting a `Cell`, or separate
functions.

## Decision

- **Images are chosen from the basis at a precision.** Every shell has a
  radius beyond which its functions are below the precision, from the
  bound `Y_l Σ_i |c_i| r^l exp(−α_i r²)` over its bare primitives with
  `Y_l = sqrt((2l + 1) / 4π)`; every atom has the largest radius of its
  shells; an image `T` is kept when some atom `A` of the cell and some
  atom `B` of the image satisfy `|A − B − T| ≤ r_A + r_B`. One list serves
  every shell pair of the cell, `−T` is kept with `T`, and the positions
  enter as given, without wrapping into the home cell. The single knob is
  the precision, `DEFAULT_LATTICE_SUM_PRECISION = 1e-12`; no cutoff radius
  or image count is exposed.
- **The periodic functions return plain complex arrays over the full
  set.** `overlap_k` and `kinetic_k` return `[kpt, ao, ao]` host arrays
  over every point of the `KPoints`, complex at every point, in the same
  form the molecular functions return their `[ao, ao]` matrices. They
  carry no symmetry: the half-zone storage and the real dtype at the
  invariant points are applied where the matrices enter a `BlockMatrix`,
  the one place that knows the symmetry class, as the molecular SCF
  driver embeds its real matrices into the structure of the class
  (ADR-0010).
- **`lattice_images` and `bloch_sum` are the reusable parts.**
  `lattice_images(basis, cell, precision=...)` gives the translations and
  `bloch_sum(images, translations, kpoints, lattice)` transforms any
  engine output with a leading image axis, so the short-range nuclear
  attraction and the fitted tensors of M5 run the same image list through
  the same transform.
- **Periodic entry points are separate functions on `Cell`.** Following
  ADR-0013, `overlap_k(engine, basis, cell, kpoints)` and `kinetic_k` are
  not overloads of `overlap(engine, basis, molecule)`: the molecular
  functions return one matrix, the periodic ones a batch, and the type
  checker keeps cells out of the molecular path.

## Consequences

- The lattice sums are converged to the precision up to the Gaussian
  prefactors of the integrals. Tightening the precision must not change a
  matrix beyond that, which the tests check together with the large-box
  test against the molecular matrices and the band-folding test of
  PLAN.md §8, and PySCF's `pbc_intor` at its tightest precision agrees to
  the integral tolerance in one, two and three dimensions.
- The image count grows with the diffuseness of the basis and falls with
  the cell size; for one-electron kernels it is cheap, but the same list
  will drive the three-centre sums of M5. Per-shell-pair image lists,
  screened by the engine's Schwarz bounds, are the extension to revisit
  there.
- Moving an atom by a lattice vector multiplies its rows and columns by a
  phase and leaves band energies unchanged; a consumer that needs a
  gauge must wrap positions itself.
- The periodic Hamiltonian of M5 builds the `BlockMatrix` with the k axis
  from these arrays, selecting the stored points and casting to the ring
  of each block, so the arrays stay independent of the symmetry class.
