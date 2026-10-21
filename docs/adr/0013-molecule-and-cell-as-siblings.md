# ADR-0013: `Molecule` and `Cell` are siblings under `AtomicSystem`

Status: accepted, 2026-10-21. Amends the `Cell` clause of ADR-0007.

## Context

ADR-0007 described a `Cell` as a `Molecule` with a `Lattice`, so that the
electron counts, symbols and positions of a molecule apply to a cell
unchanged, and the first implementation made `Cell` a subclass of
`Molecule`. Review of that implementation showed what the subtype
relation claims and what it costs: every function typed `Molecule`
accepts a cell, and most of them then compute something meaningless
without an error. `hartree_fock(engine, basis, cell)` type-checks, builds
the one-electron integrals of the home cell in vacuum and adds the Ewald
nuclear repulsion of the overridden method. The constructors differ too:
a cell cannot be built from atoms alone, so `from_atoms` could not be
overridden without a required parameter, which the type checker rejects
for a subclass. PySCF's `Cell(Mole)` is the same design and a known
source of the same mistakes (PLAN.md §11).

What the two classes genuinely share is the atom list and the electron
bookkeeping. What they do not share is their physics: nuclear repulsion,
integrals, Hamiltonians and results have a molecular and a periodic
meaning, and the molecular path being the one-k-point special case of
the periodic algorithms (PLAN.md §5.9) is a statement about the
algorithms, not about the data types.

## Decision

- `AtomicSystem` (`hafnium.molecule.system`) holds `atoms`, `charge`
  and `spin`, validates them, and provides the derived quantities:
  symbols, atomic numbers, nuclear charges, positions, electron counts
  and plain XYZ output.
- `Molecule(AtomicSystem)` and `Cell(AtomicSystem)` are siblings. A cell
  adds its `Lattice` as a keyword-only field, extended-XYZ input and
  output that carry the lattice, supercells and the Ewald nuclear
  repulsion; a molecule adds the molecular nuclear repulsion and the
  plain XYZ parser. Neither is an instance of the other.
- A function that needs only the atoms, such as `BasisSet.for_atoms`,
  takes `AtomicSystem`. A function with molecular meaning takes
  `Molecule`, a function with periodic meaning takes `Cell`, and pyright
  in strict mode, which gates every merge, rejects the wrong kind at the
  call site. No runtime `isinstance` guards are added for this purpose.
- The sentence "`Cell` is a `Molecule` with a `Lattice`" of ADR-0007 is
  replaced by "`Molecule` and `Cell` are the two kinds of
  `AtomicSystem`"; the rest of ADR-0007 stands.

## Consequences

- Passing a cell to a molecular entry point is a type error, not a wrong
  number. The periodic entry points of M5 take `Cell` and nothing else.
- `Cell.from_atoms(atoms, lattice, ...)` exists with its own signature,
  taking the lattice vectors in the same unit as the positions or a
  `Lattice` in Bohr, since no base signature constrains it.
- Code that only needs atoms must be typed on the base to accept both
  kinds; `BasisSet.for_molecule` was renamed `for_atoms` for that reason.
- A third kind of system, for example a cluster embedded in a field,
  would be another sibling rather than a flag on `Molecule`.
