# Hafnium

Hafnium is a quantum chemistry library in the planning stage: Hartree–Fock
in all Fukutome symmetry classes, Kohn–Sham DFT, density fitting, MP2 and
coupled cluster, with a Python front end, a Rust compiled layer, and a
configurable NumPy, CuPy or JAX array backend.

Nothing is implemented yet. The repository currently holds the design:

- [PLAN.md](PLAN.md): goals, architecture, the symmetry design, roadmap,
  open decisions and risks.
- [CONVENTIONS.md](CONVENTIONS.md): naming, notation, layout and style
  rules for all code in this repository.
- [docs/adr/](docs/adr/): architecture decision records.

License: Apache-2.0 (see ADR-0001; the license file is added with the
repository bootstrap).
