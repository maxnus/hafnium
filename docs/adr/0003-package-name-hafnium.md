# ADR-0003: Package name `hafnium`

Status: accepted, 2026-10-21.

## Context

The library needed a name that is not a person's name, carries no `qc` or
`py` prefix (MolSSI and the PySCF family occupy those), and is not an
established package elsewhere. Element names were attractive because of
their symbols, and hafnium's symbol Hf matches Hartree–Fock. The element
is named after Hafnia, the Latin name of Copenhagen, where it was
discovered in 1923 at Bohr's institute.

Checks on 2026-10-21 found three conflicts:

- PyPI `hafnium` is a 0.0.0 placeholder uploaded 2024-02-04 with the
  stock tutorial description and the summary "A hobby project". The same
  owner holds identical placeholders for many other element names, which
  indicates bulk name squatting rather than a stalled project.
- crates.io `hafnium` is a small, real crate (a reverse proxy).
- Hafnium is also the name of Google's Arm hypervisor in the Trusted
  Firmware project, a different domain.

## Decision

The Python package, import name and repository are `hafnium`. Publication
is far off, so the PyPI name is pursued rather than treated as blocking:

1. Contact the placeholder's owner and ask for deletion or a transfer.
2. File a PEP 541 request on github.com/pypi/support as soon as the
   repository exists, citing the placeholder's contents, the lack of
   releases for over a year, the negligible download numbers, and the
   bulk registration of element names by the same owner.
3. If the name has not been obtained by the first public release, publish
   under the fallback `hafnio` (the element's name in Spanish, Italian and
   Portuguese; free on PyPI and crates.io at the time of writing) and keep
   `hafnium` as the import name only if the moderators grant it later.

Rust crates are workspace-internal (`hafnium-core`, `hafnium-py`) and are
compiled into the wheel by maturin; they are not published to crates.io,
so the crate-name conflict has no effect.

## Consequences

- Search results will be shared with the hypervisor; documentation and the
  README should say "Hafnium quantum chemistry" in titles.
- The name decision is final for code; only the PyPI distribution name has
  a fallback.
- `hafnian`, an unrelated quantum library, differs by one letter; the
  README should not use that word.
