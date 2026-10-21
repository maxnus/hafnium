# Instructions for Claude Code in this repository

Read [PLAN.md](PLAN.md) for the design and roadmap, [CONVENTIONS.md](CONVENTIONS.md)
for naming, notation and style, and [docs/adr/](docs/adr/) for the decisions
already taken. Record new non-obvious decisions as ADRs.

## Pull requests

- Work on a branch and open a pull request; never commit to `main` directly.
- **Before opening a pull request, and again after pushing substantial
  changes to it, run the `code-review` skill on the branch** (for example
  `/code-review <PR number> high`). Fix every finding you agree with, and
  list the ones you disagree with in the pull request description so the
  reviewer can decide. A pull request is not ready for review until this
  has been done.
- Run the full check set locally before pushing: `ruff check` and
  `ruff format --check`, `pyright`, `cargo fmt --check` and `cargo clippy`,
  `pytest`, `cargo test`, and `mkdocs build --strict`.
- Put a summary, the verification performed, and review-focus notes in
  the pull request description.

## Tooling notes

- Rust is installed via rustup without modifying the shell profile; prefix
  cargo and maturin commands with `export PATH="$HOME/.cargo/bin:$PATH"`.
- `uv sync --group dev --group docs` builds the extension and installs the
  tools; `uv run <tool>` runs them.
