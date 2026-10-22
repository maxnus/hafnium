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
