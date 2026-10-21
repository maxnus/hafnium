# ADR-0005: Immutable data with standard-library dataclasses

Status: accepted, 2026-10-21.

## Context

CONVENTIONS.md §4 requires specifications, structures and results to be
immutable, with validation on construction. Options were standard-library
`dataclasses` with `frozen=True`, the `attrs` package, and `pydantic`.

## Decision

Standard-library dataclasses with `frozen=True, slots=True`, validating in
`__post_init__` and raising subclasses of `HafniumError`. Where a derived
object is needed, `dataclasses.replace` produces it.

## Reasoning

- No dependency, no learning curve, and type checkers understand them
  natively, which matters for pyright strict.
- `slots=True` keeps instances small and catches attribute typos.
- `attrs` would add validators and converters as declarative features, but
  the validation here is cross-field and reads more clearly as code.
- `pydantic` is aimed at parsing untrusted input and brings runtime
  coercion that conflicts with the explicit-dtype rule.

## Consequences

- Classes that hold arrays (`BlockMatrix`, `BlockVector`) are frozen
  containers; the arrays inside are never mutated in place, which is also
  what JAX pytree registration will require.
- Equality of dataclasses holding arrays is not meaningful (arrays do not
  compare to a single boolean); such classes set `eq=False` and provide
  explicit comparison helpers in tests.
