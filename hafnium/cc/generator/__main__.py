"""Rewrite :mod:`hafnium.cc._ccsd_equations` from the generator."""

from __future__ import annotations

from pathlib import Path

from hafnium.cc.generator.emit import module_source

TARGET = Path(__file__).resolve().parent.parent / "_ccsd_equations.py"


def main() -> None:
    """Write the generated module next to the package."""
    TARGET.write_text(module_source(), encoding="utf-8")


if __name__ == "__main__":
    main()
