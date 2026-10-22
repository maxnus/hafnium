"""The probe-charge Madelung constant of a k-point mesh (ADR-0017)."""

import numpy as np
import pytest

from hafnium.errors import InputError
from hafnium.integrals import madelung_constant
from hafnium.molecule import Lattice
from hafnium.symmetry import KPoints
from tests.cells import GENERAL_VECTORS
from tests.tolerances import ENERGY_TOL
from tests.unit.test_ewald import POINT_CHARGE_CUBIC


def test_cubic_box_gives_the_point_charge_constant() -> None:
    length = 3.0
    cubic = Lattice(((length, 0.0, 0.0), (0.0, length, 0.0), (0.0, 0.0, length)))
    assert madelung_constant(cubic, KPoints.gamma()) == pytest.approx(
        POINT_CHARGE_CUBIC / length, abs=ENERGY_TOL
    )
    # A mesh of n³ points samples the supercell of side nL.
    assert madelung_constant(cubic, KPoints.mesh((2, 2, 2))) == pytest.approx(
        POINT_CHARGE_CUBIC / (2.0 * length), abs=ENERGY_TOL
    )
    shifted = KPoints.mesh((2, 2, 2), shift=(0.5, 0.5, 0.5))
    assert madelung_constant(cubic, shifted) == pytest.approx(
        POINT_CHARGE_CUBIC / (2.0 * length), abs=ENERGY_TOL
    )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_mesh_on_the_cell_equals_gamma_on_the_supercell(dimension: int) -> None:
    lattice = Lattice(GENERAL_VECTORS[:dimension])
    multiples = (3, 1, 2)[:dimension]
    mesh = madelung_constant(lattice, KPoints.mesh(multiples))
    assert mesh > 0.0
    gamma = madelung_constant(lattice.supercell(multiples), KPoints.gamma(dimension))
    assert mesh == pytest.approx(gamma, abs=ENERGY_TOL)
    assert mesh < madelung_constant(lattice, KPoints.gamma(dimension))


def test_only_uniform_meshes_are_accepted() -> None:
    lattice = Lattice(GENERAL_VECTORS)
    with pytest.raises(InputError, match="uniform mesh"):
        madelung_constant(
            lattice,
            KPoints.from_fractional(
                [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.7, 0.0, 0.0)]
            ),
        )
    with pytest.raises(InputError, match="supercell multiples"):
        madelung_constant(lattice, KPoints.mesh((2, 2)))
    assert madelung_constant(lattice, KPoints.mesh((2, 1, 1))) == pytest.approx(
        madelung_constant(
            lattice, KPoints.from_fractional([(0.5, 0.0, 0.0), (0.0, 0.0, 0.0)])
        ),
        abs=ENERGY_TOL,
    )
    assert np.isfinite(madelung_constant(lattice, KPoints.gamma()))
