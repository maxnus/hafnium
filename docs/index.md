# Hafnium

Hafnium is a quantum chemistry library in the planning and bootstrap stage:
Hartree–Fock in all Fukutome symmetry classes, Kohn–Sham DFT, density
fitting, MP2 and coupled cluster, for molecules and for periodic systems in
one to three dimensions with k-point sampling, with a Python front end, a
Rust compiled layer, and a configurable NumPy, CuPy or JAX array backend.

Nothing beyond the repository skeleton is implemented yet. These pages hold
the design:

- [Design plan](plan.md): goals, architecture, the symmetry design, the
  backend and automatic-differentiation strategy, roadmap, open decisions
  and risks.
- [Conventions](conventions.md): naming, notation, layout and style rules
  for all code in the repository.
- [Decisions](adr/README.md): architecture decision records.
