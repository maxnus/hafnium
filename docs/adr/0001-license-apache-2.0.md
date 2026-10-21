# ADR-0001: Apache-2.0 license

Status: accepted, 2026-10-21.

## Context

The library is meant to be used by academic groups and industry, and to be
callable from other codes. Its dependencies are permissive or weak
copyleft: libcint (BSD-2-Clause), libxc (MPL-2.0 since 4.2.1), the Basis
Set Exchange code (BSD-3-Clause), NumPy and SciPy (BSD-3-Clause), JAX
(Apache-2.0), CuPy (MIT), and the Rust ecosystem (MIT or Apache-2.0).
PySCF (Apache-2.0) is used only as a test oracle and is never distributed
with the library. Options considered: Apache-2.0, MIT, MPL-2.0, LGPL-3.

## Decision

The whole repository, Python and Rust, is licensed under Apache-2.0.
Copyright is held as "The Hafnium Authors" with an AUTHORS file.
Contributions are accepted under the Developer Certificate of Origin
(sign-off on commits); there is no contributor license agreement.
A NOTICE file lists bundled third-party components and their licenses.

## Reasoning

- Permissive terms match the adoption goal; copyleft would exclude
  proprietary pipelines and complicate linking from other packages.
- Apache-2.0 includes an explicit patent grant and patent-retaliation
  clause, which MIT and BSD lack; this matters once companies contribute.
- Section 5 of the license covers inbound contributions, which makes a CLA
  unnecessary.
- A single license is simpler than the Rust convention of dual MIT or
  Apache-2.0; the crates are internal and compiled into the wheel.
- MPL-2.0 would keep modifications to the library's own files open but
  adds a compliance question for every downstream packager, with little
  benefit for a research library.

## Consequences

- Apache-2.0 is incompatible with GPL version 2 only. GPLv2-only programs
  cannot link the library. GPL-3, "GPL-2 or later" and AGPL-3 codes can.
- If libxc source files are ever modified and shipped, those files remain
  under MPL-2.0 and their source must be made available; pointing to the
  upstream repository and version satisfies this for unmodified binaries.
- Before bundling basis set data files, their terms must be checked and
  the Basis Set Exchange attribution and citation request preserved.
- SPDX identifier `Apache-2.0` goes into `pyproject.toml` and
  `Cargo.toml`; per-file headers are optional and not used.
