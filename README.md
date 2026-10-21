# Hafnium

Hafnium is a quantum chemistry library in the planning stage: Hartree–Fock
in all Fukutome symmetry classes, Kohn–Sham DFT, density fitting, MP2 and
coupled cluster, with a Python front end, a Rust compiled layer, and a
configurable NumPy, CuPy or JAX array backend.

Nothing beyond the repository skeleton is implemented yet. The repository
holds the design and the bootstrap:

- [PLAN.md](PLAN.md): goals, architecture, the symmetry design, roadmap,
  open decisions and risks.
- [CONVENTIONS.md](CONVENTIONS.md): naming, notation, layout and style
  rules for all code in this repository.
- [docs/adr/](docs/adr/): architecture decision records.
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup and the checks
  every change must pass.

License: [Apache-2.0](LICENSE).
