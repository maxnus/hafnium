# ADR-0002: Rust for the compiled layer

Status: accepted, 2026-10-21.

## Context

The library is a Python package with a compiled layer for the expensive,
index-heavy numerical work: Gaussian integrals (one-electron, four-centre,
two- and three-centre for density fitting, and their derivatives), DFT grid
generation, and basis-function evaluation on grids. Matrix contractions are
not part of this layer; they run in Python on a configurable array backend
(NumPy, CuPy, JAX). Candidates were Rust, C++20 and C. Maintainability and
code quality are explicit top priorities of the project.

## Decision

The compiled layer is written in Rust, exposed to Python through PyO3 and
built with maturin. The layer presents a plain C ABI over contiguous buffers
(buffer protocol on CPU, DLPack on GPU), and JAX reaches it through
`jax.ffi`. Existing C libraries (libcint for bootstrapping and testing,
libxc for functionals) are used through bindgen-generated FFI.

## Reasoning

- The defect classes typical of integral code (out-of-bounds writes into
  packed storage, aliasing between buffers, races when reducing into shared
  matrices) are rejected at compile time or turned into deterministic
  panics, instead of silent memory corruption.
- Shell-quartet parallelism with rayon is data-race free by construction.
- One toolchain (cargo) covers build, test, bench, docs, lint and format,
  which removes a set of decisions C++ would require and keeps all
  contributors on identical tooling.
- Codegen quality equals clang (shared LLVM backend); no implicit fast-math
  or FMA contraction, which helps reproducibility across backends.
- C interop with libcint and libxc is direct and already proven in quantum
  chemistry (REST's `rest_libcint`).
- Plain C was rejected because it lacks generics, ownership and structured
  error handling, which works against the code-quality goal.

## Consequences

- Native GPU kernels are not written in Rust for now. GPU work flows
  through the array backend first; custom kernels, when needed, live in a
  separate CUDA C++ crate behind the same C ABI. This is an accepted cost.
- libint2 is not used; the integral path is libcint behind an interface,
  then a native Rust engine.
- Per-angular-momentum specialization uses a build-time generator rather
  than C++-style templates.
- Contributors to the compiled layer need Rust. The layer is kept small
  and stable so that most contribution happens in Python.

## Revisit if

Custom GPU kernels are needed within the first year, or libint2 becomes a
hard requirement. Either change stays confined to the compiled crate
because method code only sees the Python integral interface and the C ABI.
