# Contributing

## Setup

Requirements: Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and a
stable Rust toolchain with `rustfmt` and `clippy` (install via
[rustup](https://rustup.rs)).

```bash
uv sync --group dev --group docs   # creates .venv and builds the Rust extension
uv run pre-commit install          # formatters and linters on every commit
```

After changing Rust code, rebuild the extension with `uv sync` (or
`uv run maturin develop` for a faster incremental build).

## Checks

Every pull request must pass:

```bash
uv run ruff check . && uv run ruff format --check .
uv run pyright
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
uv run pytest
cargo test --workspace
uv run mkdocs build --strict
```

## Rules

- Code follows [CONVENTIONS.md](CONVENTIONS.md). Read it before your first
  change; it is short and the tools enforce most of it.
- Non-obvious decisions are recorded as ADRs under `docs/adr/`, using the
  template there. Open a pull request with the ADR before or together with
  the code.
- Commits use conventional-commit prefixes (`feat:`, `fix:`, `docs:`,
  `refactor:`, `test:`, `perf:`, `build:`), imperative mood, and a body
  that explains why.
- Every bug fix adds a test that fails before the fix.

## Developer Certificate of Origin

Contributions are accepted under the
[Developer Certificate of Origin](https://developercertificate.org/) rather
than a contributor license agreement. Sign off each commit with
`git commit -s`, which adds a `Signed-off-by:` line certifying that you have
the right to submit the work under the project's Apache-2.0 license.
