# ADR-0026: DFT grids follow PySCF's tables, and the functionals come from `xcx`

Status: accepted, 2026-10-24. Resolves open decision 1 of PLAN.md §12.

## Context

Kohn–Sham DFT (M7) needs three things the library did not have: a
quadrature grid for the exchange–correlation energy, the basis functions
and their derivatives on its points, and the functionals themselves with
their first derivatives. The plan left one question open: whether to bind
libxc through its Python package `pylibxc` or through an own binding in
the Rust crate.

The facts changed since the plan was written. `pylibxc` has no wheels
for the Python versions this project supports: the PyPI package
`pylibxc2` stops at libxc 6.0.0 for CPython 3.5 to 3.9. Building libxc
from its C sources inside the wheel would need a C toolchain in a build
that is Rust only (ADR-0002), tens of megabytes of generated source in
the repository and minutes of compile time, for a library of which a few
dozen functionals are used. Linking a system libxc would make
`pip install hafnium` depend on a library the user installs separately.
Meanwhile a pure-Rust library appeared: `xcx` (nmrtist, 2026, MIT or
Apache-2.0 with the ported formulas under MPL-2.0) implements the
functionals of libxc with libxc's ids, names, array layouts and numerical
conventions, derives every first and second derivative by forward-mode
automatic differentiation, and checks itself against libxc to 1e-10. On
seventeen functionals across the LDA, GGA, meta-GGA, hybrid and
range-separated rungs its energies and derivatives agree with the libxc
7.0.0 that PySCF ships to 1e-13 relative at the points tried here.

For the grids the question was whether to follow a textbook scheme or
PySCF's choices. PySCF's grids are the Treutler–Ahlrichs radial mapping
with Treutler's element scales, Lebedev rules by level and period, NWChem
pruning near the nuclei, and Becke's partition with Treutler's atomic
size adjustment from the Bragg radii, all published schemes with
published tables. Following them makes every Kohn–Sham result directly
comparable with PySCF's at the same level, which is what the exit
criterion of M7 asks for.

## Decision

- **The functionals come from the `xcx` crate**, a dependency of
  `hafnium-core`. `hafnium.xc.Functional` is a linear combination of
  libxc functionals named by their libxc names (`gga_x_pbe`,
  `hyb_gga_xc_b3lyp`, ...) plus the exact exchange it asks for, with a
  registry of common names (`pbe`, `b3lyp`, `tpss`, `r2scan`,
  `wb97x-v`, `hf`, ...). The compiled layer evaluates the combination in
  parallel over the points and returns the energy per particle and the
  derivatives of the energy density in libxc's layout. A libxc build
  behind the same interface stays the fallback if a functional outside
  `xcx`'s list becomes necessary.
- **Exact exchange follows libxc's range-separation convention.** The
  exchange of a hybrid is `a K + b K_lr(ω)` with `K` the exact exchange
  of the full Coulomb kernel and `K_lr` that of `erf(ω r)/r`, the
  fractions `a` (`Functional.exact_exchange`) and `b`
  (`long_range_exchange`) of libxc's `a + b erf(ω r)`. The Hamiltonian
  carries the long-range integrals as a second two-electron
  representation (`Hamiltonian.long_range_two_body`), built by the same
  exact or density-fitted builders with the long-range kernel, and skips
  the exchange of a pure functional altogether; `coulomb_matrix` and
  `exchange_matrix` are therefore separate contractions.
- **Grids follow PySCF's construction and tables.** `GridSpec` fixes the
  radial count and the Lebedev degree per level (0 to 9) and period; the
  radial quadrature is the Treutler–Ahlrichs M4 mapping with the element
  scale, the angular rule a Lebedev–Laikov rule expanded from its orbit
  generators (`hafnium/grid/data/lebedev.json`, with provenance), the
  inner shells pruned in the NWChem fashion, and the atomic grids joined
  by Becke's partition with Treutler's adjustment from the Bragg radii.
  The grid of a molecule coincides with PySCF's point by point.
- **Basis values on grid points come from the compiled layer** as
  `[point, component, ao]` with the value, the gradient and the six
  second derivatives, each shell cut off at a radius Python derives from
  its exponents and a precision (`function_radius`, the policy staying
  in Python as for the lattice sums).
- **The exchange–correlation term is collinear and lives in the
  Hamiltonian.** `ExchangeCorrelation` evaluates the alpha and beta
  densities, gradients and kinetic energy densities of a structured
  density from its spin blocks in the atomic basis, batch by batch,
  assembles the potential of each spin and places it into the stored
  blocks of the structure, projected onto irrep sectors where there are
  any. The restricted and unrestricted classes, real and complex, are
  served by the one implementation (RKS and UKS); the generalized
  classes need the noncollinear evaluation of PLAN.md §5.4 and are
  rejected, as is a k axis until the periodic part of M7 (both since
  served: ADR-0027 and ADR-0029). `kohn_sham`
  mirrors `hartree_fock` with a functional and `KohnShamOptions`, the SCF
  options plus the grid.

## Consequences

- The build stays Rust only, and every functional comes with analytic
  first derivatives and, through `xcx`, second derivatives for the
  stability analysis and response properties of later milestones. PLAN.md
  §6.3 spoke of libxc's derivatives up to fourth order; `xcx` provides
  two, which is what gradients and Hessians of the energy need.
- The functional list is `xcx`'s: Slater, VWN, PW92, PBE and its
  variants, B88, LYP, P86, TPSS, r²SCAN, M06-L, M06-2X, B3LYP, PBE0,
  PW6B95, the ωB97 family and the B97 series. SCAN, HSE, CAM-B3LYP,
  functionals of the Laplacian and the VV10 nonlocal correlation of the
  ωB97 functionals are not available; a functional defined with VV10 is
  refused unless the user drops the term explicitly
  (`Functional.without_vv10`).
- Kohn–Sham energies agree with PySCF to 1e-10 Hartree for RKS and UKS
  across the rungs, with exact and fitted integrals, at every grid level
  tried, so the comparison tests run at the energy tolerance of the
  Hartree–Fock ones rather than at a grid-limited one.
- The entry point has seven parameters, so the limit on function
  arguments in CONVENTIONS.md §1 rises from six to seven.
- Periodic Kohn–Sham (grids with lattice images and Bloch-phased basis
  values), noncollinear functionals for the generalized classes, VV10,
  dispersion corrections and the derivatives of the grid weights for the
  nuclear gradients of M11 remain.
