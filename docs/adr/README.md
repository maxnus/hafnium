# Architecture decision records

Each non-obvious decision is recorded in a short file with the sections
Context, Decision and Consequences, using [the template](0000-template.md).
An accepted ADR is never edited afterwards; a change is recorded by a new
ADR that supersedes it.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-license-apache-2.0.md) | Apache-2.0 license | accepted 2026-10-21 |
| [0002](0002-rust-for-compiled-layer.md) | Rust for the compiled layer | accepted 2026-10-21 |
| [0003](0003-package-name-hafnium.md) | Package name `hafnium` | accepted 2026-10-21 |
| [0004](0004-python-toolchain-baseline.md) | Python toolchain baseline | accepted 2026-10-21 |
| [0005](0005-immutable-data-with-dataclasses.md) | Immutable data with standard-library dataclasses | accepted 2026-10-21 |
| [0006](0006-basis-function-conventions.md) | Basis function and physical conventions | accepted 2026-10-21 |
| [0007](0007-periodic-systems-as-k-point-sectors.md) | Periodic systems as k-point sectors | accepted 2026-10-21 |
| [0008](0008-mcmurchie-davidson-integrals.md) | McMurchie–Davidson for the native integral engine | accepted 2026-10-21 |
| [0009](0009-p-shell-order-verified.md) | Spherical function conventions verified against libcint | accepted 2026-10-21 |
| [0010](0010-scf-driver-over-structured-matrices.md) | One SCF driver over structured matrices | accepted 2026-10-21 |
| [0011](0011-general-contractions-in-the-kernels.md) | General contractions are evaluated once per shell | accepted 2026-10-21 |
| [0012](0012-ewald-sums-in-reduced-dimensions.md) | One Ewald regularization in every dimension | accepted 2026-10-21 |
| [0013](0013-molecule-and-cell-as-siblings.md) | `Molecule` and `Cell` are siblings under `AtomicSystem` | accepted 2026-10-21 |
| [0014](0014-no-libcint-adapter.md) | No libcint adapter | accepted 2026-10-21 |
| [0015](0015-k-axis-of-structured-matrices.md) | The k axis of structured matrices | accepted 2026-10-22 |
| [0016](0016-lattice-sums-of-one-electron-integrals.md) | Lattice sums of one-electron integrals | accepted 2026-10-22 |
| [0017](0017-probe-charge-exchange-correction.md) | Probe-charge Ewald correction of the exchange divergence | accepted 2026-10-22 |
| [0018](0018-screened-lattice-sums-in-the-kernels.md) | Screened lattice sums in the kernels and the regularized nuclear attraction | accepted 2026-10-22 |
| [0019](0019-density-fitting-with-momentum-transfer.md) | Density fitting with momentum transfer | accepted 2026-10-22 |
| [0020](0020-k-point-hartree-fock-assembly.md) | k-point Hartree–Fock through the structured matrices | accepted 2026-10-22 |
| [0021](0021-reduced-dimension-kernels-by-quadrature.md) | Reduced-dimension reciprocal-space kernels by quadrature | accepted 2026-10-22 |
| [0022](0022-molecular-point-groups-in-the-standard-frame.md) | Molecular point groups in the standard frame | accepted 2026-10-22 |
| [0023](0023-irrep-sectors-of-the-structured-matrices.md) | Irrep sectors of the structured matrices | accepted 2026-10-22 |
| [0024](0024-space-groups-reduce-the-k-point-set.md) | Space groups reduce the k-point set | accepted 2026-10-23 |
| [0025](0025-lattice-sums-reach-by-exponent.md) | Lattice sums reach by exponent, and the Bloch three-centre kernel accumulates by class | accepted 2026-10-24 |
| [0026](0026-dft-grids-and-functionals-from-xcx.md) | DFT grids follow PySCF's tables, and the functionals come from `xcx` | accepted 2026-10-24 |
| [0027](0027-periodic-kohn-sham-on-local-partition-grids.md) | Periodic Kohn–Sham integrates on atom-centred grids with a local partition | accepted 2026-10-24 |
| [0028](0028-folded-phases-and-coulomb-only-fitting.md) | The Bloch three-centre kernel folds both phases, and pure functionals fit the Coulomb term alone | accepted 2026-10-24 |
| [0029](0029-noncollinear-functionals-by-the-multi-collinear-approach.md) | The generalized classes take the multi-collinear form of a collinear functional | accepted 2026-10-25 |
| [0030](0030-block-tensors-with-a-conservation-rule.md) | Block tensors conserve spin and irrep charges, and DF-MP2 is written once over them | accepted 2026-10-28 |
| [0031](0031-correlated-tensors-over-k-points.md) | Correlated tensors over k-points conserve crystal momentum and store one block per orbit of time reversal | accepted 2026-10-28 |
| [0032](0032-coupled-cluster-equations-from-an-in-house-generator.md) | Coupled-cluster equations come from an in-house Wick generator and run as one program over block tensors | accepted 2026-10-30 |
| [0033](0033-the-triples-correction-over-windows-of-the-occupied-axis.md) | The (T) correction comes from the same generator and runs over windows of the occupied axis | accepted 2026-10-31 |
| [0034](0034-the-lambda-equations-and-the-density-from-the-same-generator.md) | The Λ equations and the one-particle density come from the same generator as the linked terms of the energy functional | accepted 2026-10-31 |
| [0035](0035-derivative-integrals-from-the-undifferentiated-kernels.md) | Derivative integrals come from the undifferentiated kernels, and the gradient contracts them at fixed density | accepted 2026-11-01 |
