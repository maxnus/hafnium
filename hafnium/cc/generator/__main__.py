"""Rewrite the generated equation modules of :mod:`hafnium.cc` from the generator."""

from __future__ import annotations

from pathlib import Path

from hafnium.cc.generator.emit import ccsd_module_source, ccsd_t_module_source

PACKAGE = Path(__file__).resolve().parent.parent
TARGETS = {
    PACKAGE / "_ccsd_equations.py": ccsd_module_source,
    PACKAGE / "_ccsd_t_equations.py": ccsd_t_module_source,
}
"""The generated modules and the functions that write their source."""


def main() -> None:
    """Write the generated modules next to the package."""
    for target, source in TARGETS.items():
        target.write_text(source(), encoding="utf-8")


if __name__ == "__main__":
    main()
