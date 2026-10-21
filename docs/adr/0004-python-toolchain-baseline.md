# ADR-0004: Python toolchain baseline

Status: accepted, 2026-10-21 (chosen as the bootstrap default; cheap to
revisit before the first release).

## Context

The bootstrap milestone needs a minimum Python version, an array library
baseline, and the tools that CI enforces. The plan left the type checker
and the minimum Python version open.

## Decision

- Python 3.11 or newer; NumPy 2.0 or newer only.
- `uv` for environments and lock file; `maturin` as the build backend.
- `ruff` for formatting and linting, with the rule set in `pyproject.toml`.
  The ban on importing NumPy outside `hafnium/backend/` is enforced by the
  tidy-imports rule, not by review.
- `pyright` in strict mode as the single type checker.
- `pytest` with `hypothesis`; markers `reference`, `gpu` and `slow`.
- `pre-commit` runs the formatters and linters locally.

## Reasoning

- Python 3.11 brings `typing.Self`, exception groups and the faster
  interpreter; older versions are out of security support by 2027.
- NumPy 2 is where the array API standard is implemented on the root
  namespace, which the backend design relies on.
- `pyright` over `mypy`: faster, stricter inference on protocols and
  generics, which matters for backend-agnostic array code, and no plugin
  system to maintain. Both are strict-mode capable; using one avoids
  satisfying two checkers with different opinions.
- `uv` over `pip` and virtualenv: one tool for environments, locking,
  Python installation and running tools, with a committed lock file for
  reproducible CI.

## Consequences

- Contributors need `uv` and a Rust toolchain; `CONTRIBUTING.md` says how.
- Switching to `mypy` later means replacing one configuration table and
  one CI step; code annotated for pyright strict will largely pass mypy.
- Dropping or adding Python versions is a change to `requires-python`, the
  CI matrix and this ADR's successor.
