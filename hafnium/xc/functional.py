"""Functionals as linear combinations of libxc's, with their exact exchange."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from hafnium import _kernels
from hafnium.errors import InputError


class Rung(Enum):
    """What a functional reads at every point."""

    LDA = "lda"
    """The spin densities."""
    GGA = "gga"
    """The densities and their gradients."""
    META_GGA = "meta-gga"
    """The densities, their gradients and the kinetic energy densities."""

    @property
    def rank(self) -> int:
        """Position of the rung on the ladder, for comparisons."""
        return ("lda", "gga", "meta-gga").index(self.value)


@dataclass(frozen=True, slots=True)
class ComponentInfo:
    """Metadata of one libxc functional.

    Parameters
    ----------
    id
        The libxc id.
    name
        The canonical libxc name.
    rung
        Its rung.
    needs_laplacian
        Whether it reads the Laplacian of the density, which is not
        supported.
    exact_exchange
        Fraction of exact exchange with the full Coulomb kernel.
    range_separation
        ``(omega, alpha, beta)`` of a range-separated hybrid, whose exact
        exchange is ``alpha + beta erf(ω r)`` of the Coulomb interaction;
        ``None`` otherwise.
    vv10
        Parameters ``(b, C)`` of the VV10 nonlocal correlation the
        functional is defined with; ``None`` without.
    """

    id: int
    name: str
    rung: Rung
    needs_laplacian: bool
    exact_exchange: float
    range_separation: tuple[float, float, float] | None
    vv10: tuple[float, float] | None


@functools.cache
def component_info(name: str) -> ComponentInfo:
    """Metadata of the libxc functional ``name`` (case-insensitive).

    Raises
    ------
    InputError
        If no functional of that name is available.
    """
    try:
        raw: dict[str, Any] = _kernels.functional_info(name)
    except ValueError as error:
        raise InputError(str(error)) from None
    separation = raw["range_separation"]
    vv10 = raw["vv10"]
    return ComponentInfo(
        int(raw["id"]),
        str(raw["name"]),
        Rung(raw["rung"]),
        bool(raw["needs_laplacian"]),
        float(raw["exact_exchange"]),
        None if separation is None else tuple(float(x) for x in separation),  # type: ignore[arg-type]
        None if vv10 is None else (float(vv10[0]), float(vv10[1])),
    )


@dataclass(frozen=True, slots=True)
class Functional:
    """A linear combination of libxc functionals and the exact exchange it needs.

    The exchange–correlation energy is
    ``Σ_i weight_i E_i[ρ] − (a/2) Σ_σ tr(D_σ K_σ) − (b/2) Σ_σ tr(D_σ K_σ^{lr})``
    with the exact exchange matrices ``K`` of the full Coulomb kernel and
    ``K^{lr}`` of its long-range part ``erf(ω r)/r``, so a global hybrid
    has ``a > 0`` and ``b = 0``, and a range-separated hybrid in libxc's
    convention ``a + b erf(ω r)`` has both.

    Parameters
    ----------
    name
        Label for messages and logs.
    components
        Pairs ``(libxc name, weight)`` of the semilocal parts; empty for a
        functional of exact exchange alone.
    exact_exchange
        ``a``, the fraction of exact exchange with the full kernel.
    long_range_exchange
        ``b``, the fraction of long-range exact exchange.
    omega
        Range-separation parameter of the long-range exchange in inverse
        Bohr; zero when ``long_range_exchange`` is zero.
    vv10
        Parameters ``(b, C)`` of a VV10 nonlocal correlation the functional
        is defined with and this version does not evaluate; the
        Kohn–Sham builder rejects such a functional, and
        :meth:`without_vv10` drops the term explicitly.
    """

    name: str
    components: tuple[tuple[str, float], ...] = ()
    exact_exchange: float = 0.0
    long_range_exchange: float = 0.0
    omega: float = 0.0
    vv10: tuple[float, float] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        components = tuple(
            (component_info(component).name, float(weight))
            for component, weight in self.components
        )
        object.__setattr__(self, "components", components)
        if any(not math.isfinite(weight) for _, weight in components):
            raise InputError("functional weights are finite")
        for value, what in (
            (self.exact_exchange, "exact exchange"),
            (self.long_range_exchange, "long-range exchange"),
        ):
            if not math.isfinite(value):
                raise InputError(f"the fraction of {what} is finite")
        if self.long_range_exchange != 0.0 and not (
            math.isfinite(self.omega) and self.omega > 0.0
        ):
            raise InputError(
                "long-range exact exchange needs a positive range-separation "
                "parameter omega"
            )
        if self.long_range_exchange == 0.0 and self.omega != 0.0:
            raise InputError("omega is zero without long-range exact exchange")
        if (
            not components
            and self.exact_exchange == 0.0
            and not self.long_range_exchange
        ):
            raise InputError("a functional needs components or exact exchange")

    @classmethod
    def from_libxc(
        cls,
        *names: str,
        weights: Sequence[float] | None = None,
        name: str | None = None,
    ) -> Functional:
        """Combine libxc functionals with their own exact exchange.

        ``weights`` default to one each. The exact exchange is the weighted
        sum of the components' fractions; a range-separated component sets
        ``omega`` and the long-range fraction, which requires it to be the
        only range-separated component and to carry weight one, so that the
        combination stays the functional libxc defines.
        """
        if not names:
            raise InputError("name at least one libxc functional")
        scales = (1.0,) * len(names) if weights is None else tuple(weights)
        if len(scales) != len(names):
            raise InputError("one weight per functional")
        infos = [component_info(component) for component in names]
        exact = sum(
            weight * info.exact_exchange
            for info, weight in zip(infos, scales, strict=True)
        )
        separated = [
            (info, weight)
            for info, weight in zip(infos, scales, strict=True)
            if info.range_separation is not None
        ]
        omega = long_range = 0.0
        if separated:
            if len(separated) > 1 or separated[0][1] != 1.0:
                raise InputError(
                    "a range-separated functional enters with weight one and alone; "
                    "give omega and the exchange fractions explicitly to combine "
                    "range-separated parts"
                )
            separation = separated[0][0].range_separation
            assert separation is not None
            omega, long_range = separation[0], separation[2]
        vv10 = {info.vv10 for info in infos if info.vv10 is not None}
        if len(vv10) > 1:
            raise InputError("the components ask for different VV10 parameters")
        return cls(
            name if name is not None else " + ".join(info.name for info in infos),
            tuple(zip((info.name for info in infos), scales, strict=True)),
            exact,
            long_range,
            omega,
            vv10=next(iter(vv10)) if vv10 else None,
        )

    @classmethod
    def named(cls, name: str) -> Functional:
        """Return a functional of :data:`NAMED_FUNCTIONALS` by its common name."""
        key = name.strip().lower()
        try:
            components = NAMED_FUNCTIONALS[key]
        except KeyError:
            raise InputError(
                f"unknown functional {name!r}; the named functionals are "
                f"{', '.join(NAMED_FUNCTIONALS)}, and Functional.from_libxc takes "
                "libxc names"
            ) from None
        if not components:
            return cls(key, (), 1.0)
        return cls.from_libxc(*components, name=key)

    @property
    def rung(self) -> Rung:
        """The highest rung among the components; LDA for exact exchange alone."""
        rungs = [component_info(component).rung for component, _ in self.components]
        return max(rungs, key=lambda rung: rung.rank, default=Rung.LDA)

    @property
    def needs_gradient(self) -> bool:
        """Whether the density gradient enters."""
        return self.rung.rank >= Rung.GGA.rank

    @property
    def needs_tau(self) -> bool:
        """Whether the kinetic energy density enters."""
        return self.rung is Rung.META_GGA

    @property
    def is_hybrid(self) -> bool:
        """Whether any exact exchange enters."""
        return self.exact_exchange != 0.0 or self.long_range_exchange != 0.0

    @property
    def is_range_separated(self) -> bool:
        """Whether long-range exact exchange enters."""
        return self.long_range_exchange != 0.0

    def without_vv10(self) -> Functional:
        """Return the same functional with its VV10 term dropped."""
        return Functional(
            self.name,
            self.components,
            self.exact_exchange,
            self.long_range_exchange,
            self.omega,
            vv10=None,
        )

    def describe(self) -> str:
        """One line naming the components and the exact exchange."""
        parts = [f"{weight:g} {component}" for component, weight in self.components]
        if self.exact_exchange:
            parts.append(f"{self.exact_exchange:g} exact exchange")
        if self.long_range_exchange:
            parts.append(
                f"{self.long_range_exchange:g} long-range exact exchange "
                f"(omega {self.omega:g})"
            )
        return f"{self.name}: {' + '.join(parts)}"


NAMED_FUNCTIONALS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "hf": (),
        "svwn": ("lda_x", "lda_c_vwn"),
        "pbe": ("gga_x_pbe", "gga_c_pbe"),
        "blyp": ("gga_x_b88", "gga_c_lyp"),
        "b3lyp": ("hyb_gga_xc_b3lyp",),
        "b3lyp5": ("hyb_gga_xc_b3lyp5",),
        "pbe0": ("hyb_gga_xc_pbeh",),
        "tpss": ("mgga_x_tpss", "mgga_c_tpss"),
        "r2scan": ("mgga_x_r2scan", "mgga_c_r2scan"),
        "m06-l": ("mgga_x_m06_l", "mgga_c_m06_l"),
        "m06-2x": ("hyb_mgga_x_m06_2x", "mgga_c_m06_2x"),
        "wb97x-v": ("hyb_gga_xc_wb97x_v",),
        "wb97m-v": ("hyb_mgga_xc_wb97m_v",),
    }
)
"""Common names and the libxc functionals behind them.

``hf`` is exact exchange alone, which turns the Kohn–Sham solver into
Hartree–Fock; ``svwn`` uses VWN5, ``b3lyp`` libxc's definition with the
RPA parametrization of VWN as in Gaussian, ``b3lyp5`` the VWN5 variant.
The ωB97 functionals carry VV10 nonlocal correlation, see
:attr:`Functional.vv10`.
"""
