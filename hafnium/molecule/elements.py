"""Chemical elements: symbols and atomic numbers."""

from hafnium.errors import InputError

SYMBOLS: tuple[str, ...] = (
    "X",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)  # fmt: skip
"""Element symbols indexed by atomic number; index 0 is the ghost atom ``X``."""

_NUMBERS: dict[str, int] = {
    symbol.lower(): number for number, symbol in enumerate(SYMBOLS)
}


def atomic_number(symbol: str) -> int:
    """Return the atomic number of an element symbol (case-insensitive).

    Raises
    ------
    InputError
        If the symbol is not a chemical element.
    """
    try:
        return _NUMBERS[symbol.strip().lower()]
    except KeyError:
        raise InputError(f"unknown element symbol {symbol!r}") from None


def element_symbol(number: int) -> str:
    """Return the symbol of the element with atomic number ``number``."""
    if not 0 <= number < len(SYMBOLS):
        raise InputError(f"no element with atomic number {number}")
    return SYMBOLS[number]
