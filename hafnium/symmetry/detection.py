"""Point-group detection of molecules and their standard orientation (ADR-0022).

The full point group of a molecule follows from its symmetry elements:
rotation axes, mirror planes, an inversion centre and improper axes, found
by testing candidate directions built from the positions of equivalent
atoms. The Abelian subgroup a calculation imposes, D2h or one of its
subgroups, is then placed in its standard frame, in which its operations
are the sign operations of :class:`~hafnium.symmetry.Operation`.

The frame rules follow PySCF so that irrep labels agree between the two
programs: the principal axis is ``z``; a planar C2v molecule lies in the
``yz`` plane; a planar D2h molecule lies in the ``yz`` plane with ``z``
along the in-plane axis more atoms are displaced along; the axes of a D2
or a cubic frame are the symmetry axes closest to the input axes; a
twofold axis or a mirror normal of an asymmetric top becomes ``z`` by a
cyclic permutation of the axes of the charge tensor; and when a group of
odd order ``n`` descends to its subgroup a perpendicular twofold axis, or
the normal of a vertical mirror, becomes ``z``. Which axes are ``z`` and
``x`` fixes the labels; the signs of the axes do not, and they are chosen
to keep the rotation into the frame closest to the identity, as a proper
rotation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from hafnium.backend import Array, array_namespace, eigh, host_namespace
from hafnium.errors import InputError, SymmetryError
from hafnium.molecule import Atom, Molecule
from hafnium.symmetry.point_group import PointGroup

DEFAULT_SYMMETRY_TOLERANCE = 1e-5
"""Distance in Bohr within which an operation must map every atom onto an atom."""

Vector = tuple[float, float, float]
Frame = tuple[Vector, Vector, Vector]
"""Three orthonormal rows: the standard ``x``, ``y``, ``z`` in the input frame."""

IDENTITY_FRAME: Frame = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_ORIGIN: Vector = (0.0, 0.0, 0.0)
_PAIR_ATOMS = 24
"""Atoms per equivalence class combined into pair and triple candidates."""
_CUBIC = {"T", "Td", "Th", "O", "Oh", "I", "Ih"}


# ------------------------------------------------------------------ small vectors


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vector) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vector) -> Vector:
    length = _norm(a)
    return (a[0] / length, a[1] / length, a[2] / length)


def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(s: float, a: Vector) -> Vector:
    return (s * a[0], s * a[1], s * a[2])


def _is_parallel(a: Vector, b: Vector, tolerance: float) -> bool:
    """Whether the unit vectors ``a`` and ``b`` are parallel or antiparallel."""
    return abs(abs(_dot(a, b)) - 1.0) < tolerance


def _is_perpendicular(a: Vector, b: Vector, tolerance: float) -> bool:
    return abs(_dot(a, b)) < tolerance


def _rotation(axis: Vector, angle: float) -> Frame:
    """Rotation by ``angle`` about the unit vector ``axis`` (Rodrigues)."""
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = axis
    return (
        (c + (1 - c) * x * x, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y),
        ((1 - c) * y * x + s * z, c + (1 - c) * y * y, (1 - c) * y * z - s * x),
        ((1 - c) * z * x - s * y, (1 - c) * z * y + s * x, c + (1 - c) * z * z),
    )


def _reflection(normal: Vector) -> Frame:
    """Reflection in the plane perpendicular to the unit vector ``normal``."""
    x, y, z = normal
    return (
        (1 - 2 * x * x, -2 * x * y, -2 * x * z),
        (-2 * y * x, 1 - 2 * y * y, -2 * y * z),
        (-2 * z * x, -2 * z * y, 1 - 2 * z * z),
    )


def _matmul(a: Frame, b: Frame) -> Frame:
    columns = tuple(zip(*b, strict=True))
    return tuple(  # type: ignore[return-value]
        tuple(_dot(row, column) for column in columns) for row in a
    )


def _improper(axis: Vector, order: int) -> Frame:
    """Improper rotation ``S_order``: rotation by ``2π/order`` then reflection."""
    return _matmul(_reflection(axis), _rotation(axis, 2.0 * math.pi / order))


_INVERSION: Frame = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))


def determinant(frame: Frame) -> float:
    """Return the determinant of a ``[3, 3]`` matrix given as three rows."""
    return _dot(frame[0], _cross(frame[1], frame[2]))


def _make_frame(z: Vector, x: Vector) -> Frame:
    """Right-handed orthonormal frame with ``z`` and the part of ``x`` normal to it."""
    y = _unit(_cross(z, x))
    return (_unit(_cross(y, z)), y, _unit(z))


def _proper(frame: Frame) -> Frame:
    """Flip axes so the frame is right-handed and closest to the input axes."""
    flip = -1.0 if determinant(frame) < 0.0 else 1.0
    best: Frame | None = None
    best_trace = -math.inf
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            sz = flip * sx * sy
            signed = (_scale(sx, frame[0]), _scale(sy, frame[1]), _scale(sz, frame[2]))
            trace = signed[0][0] + signed[1][1] + signed[2][2]
            if trace > best_trace + 1e-12:
                best, best_trace = signed, trace
    assert best is not None
    return best


def _closest_frame(axes: Frame) -> Frame:
    """Assign three mutually perpendicular axes to ``x, y, z`` closest to the input."""
    best: Frame | None = None
    best_trace = -math.inf
    for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        candidate = _proper((axes[order[0]], axes[order[1]], axes[order[2]]))
        trace = candidate[0][0] + candidate[1][1] + candidate[2][2]
        if trace > best_trace + 1e-12:
            best, best_trace = candidate, trace
    assert best is not None
    return best


def _unique_directions(vectors: list[Vector], tolerance: float) -> list[Vector]:
    """Normalize, drop short and repeated vectors (up to sign), keep the order."""
    seen: dict[tuple[float, float, float], Vector] = {}
    for vector in vectors:
        if _norm(vector) < tolerance:
            continue
        unit = _unit(vector)
        for component in unit:
            if abs(component) > tolerance:
                if component < 0.0:
                    unit = _scale(-1.0, unit)
                break
        key = (round(unit[0], 6), round(unit[1], 6), round(unit[2], 6))
        seen.setdefault(key, unit)
    return list(seen.values())


# ------------------------------------------------------------------ the geometry


@dataclass(frozen=True, slots=True, eq=False)
class _Geometry:
    """A molecule relative to its centre of nuclear charge."""

    positions: Array
    """``[atom, 3]`` in Bohr, relative to the centre."""
    points: tuple[Vector, ...]
    classes: tuple[tuple[int, ...], ...]
    """Atoms of one element at one distance from the centre: equivalence candidates."""
    axes: Frame
    """Eigenvectors of the charge tensor ``Σ_A Z_A r_A r_Aᵀ``, ascending eigenvalue."""
    extents: Vector
    """Its eigenvalues."""
    tolerance: float

    @classmethod
    def build(cls, molecule: Molecule, tolerance: float) -> tuple[_Geometry, Vector]:
        xp = host_namespace()
        positions = molecule.positions_array(xp)
        charges = xp.asarray(molecule.nuclear_charges, dtype=xp.float64)
        centre = (charges @ positions) / xp.sum(charges)
        positions = positions - centre
        tensor = xp.linalg.matrix_transpose(positions * xp.expand_dims(charges, 1))
        values, vectors = eigh(tensor @ positions)
        points = tuple(
            (float(row[0]), float(row[1]), float(row[2])) for row in positions
        )
        axes = tuple(
            (float(vectors[0, k]), float(vectors[1, k]), float(vectors[2, k]))
            for k in range(3)
        )
        extents = (float(values[0]), float(values[1]), float(values[2]))
        geometry = cls(
            positions,
            points,
            _classes(molecule.symbols, points, tolerance),
            axes,  # type: ignore[arg-type]
            extents,
            tolerance,
        )
        return geometry, (float(centre[0]), float(centre[1]), float(centre[2]))

    def is_symmetric(self, matrix: Frame) -> bool:
        """Whether ``matrix`` maps every atom onto an atom of its class."""
        xp = array_namespace(self.positions)
        moved = self.positions @ xp.linalg.matrix_transpose(
            xp.asarray(matrix, dtype=xp.float64)
        )
        for members in self.classes:
            index = xp.asarray(members, dtype=xp.int64)
            original = xp.take(self.positions, index, axis=0)
            image = xp.take(moved, index, axis=0)
            distance = xp.linalg.vector_norm(
                xp.expand_dims(image, 1) - xp.expand_dims(original, 0), axis=-1
            )
            if not bool(xp.all(xp.min(distance, axis=1) < self.tolerance)):
                return False
        return True

    def has_rotation(self, axis: Vector, order: int) -> bool:
        return self.is_symmetric(_rotation(axis, 2.0 * math.pi / order))

    def has_mirror(self, normal: Vector) -> bool:
        return self.is_symmetric(_reflection(normal))

    def is_planar_along(self, normal: Vector) -> bool:
        """Whether every atom lies in the plane through the centre with ``normal``."""
        return all(abs(_dot(point, normal)) < self.tolerance for point in self.points)

    def count_displaced_along(self, axis: Vector) -> int:
        """Count the atoms with a component along ``axis``."""
        return sum(abs(_dot(point, axis)) > self.tolerance for point in self.points)

    def extent_along(self, axis: Vector) -> float:
        """Return ``Σ_A Z_A (r_A · axis)²``."""
        return sum(self.extents[k] * _dot(axis, self.axes[k]) ** 2 for k in range(3))


def _classes(
    symbols: tuple[str, ...], points: tuple[Vector, ...], tolerance: float
) -> tuple[tuple[int, ...], ...]:
    by_symbol: dict[str, list[int]] = {}
    for index, symbol in enumerate(symbols):
        by_symbol.setdefault(symbol, []).append(index)
    classes: list[tuple[int, ...]] = []
    for members in by_symbol.values():
        ordered = sorted(members, key=lambda index: _norm(points[index]))
        start = 0
        for position in range(1, len(ordered) + 1):
            if position == len(ordered) or (
                _norm(points[ordered[position]]) - _norm(points[ordered[position - 1]])
                > tolerance
            ):
                classes.append(tuple(sorted(ordered[start:position])))
                start = position
    return tuple(classes)


# ------------------------------------------------------------------ the elements


def _candidate_directions(geometry: _Geometry) -> list[Vector]:
    """Directions that can carry a rotation axis or a mirror normal.

    The axes of the charge tensor, the direction of every atom, the sums,
    differences and cross products of pairs of equivalent atoms (twofold
    axes and mirror normals that exchange them), and the normals of the
    planes through three equivalent atoms (higher axes, whose orbits are
    regular polygons).
    """
    candidates: list[Vector] = list(geometry.axes)
    for members in geometry.classes:
        points = [geometry.points[index] for index in members]
        candidates.extend(points)
        head = points[:_PAIR_ATOMS]
        for i, a in enumerate(head):
            for b in head[i + 1 :]:
                candidates.extend((_add(a, b), _sub(a, b), _cross(a, b)))
        differences = [_sub(point, head[0]) for point in head[1:]]
        for i, a in enumerate(differences):
            candidates.extend(_cross(a, b) for b in differences[i + 1 :])
    return _unique_directions(candidates, geometry.tolerance)


def _rotation_orders(geometry: _Geometry, axis: Vector) -> list[int]:
    """Orders a rotation about ``axis`` could have, highest first.

    The atoms of the first class with members off the axis that share the
    height and the radius of its outermost member sit at angles ``2π/n``
    from it for every order ``n`` the axis carries.
    """
    for members in geometry.classes:
        points = [geometry.points[index] for index in members]
        radial = [_sub(p, _scale(_dot(p, axis), axis)) for p in points]
        radii = [_norm(r) for r in radial]
        if max(radii) < geometry.tolerance:
            continue
        reference = max(range(len(points)), key=lambda i: radii[i])
        height, radius = _dot(points[reference], axis), radii[reference]
        first = _unit(radial[reference])
        second = _cross(axis, first)
        orders: set[int] = set()
        for point, direction, length in zip(points, radial, radii, strict=True):
            if abs(_dot(point, axis) - height) > geometry.tolerance:
                continue
            if abs(length - radius) > geometry.tolerance:
                continue
            angle = math.atan2(_dot(direction, second), _dot(direction, first))
            angle %= 2.0 * math.pi
            if angle < geometry.tolerance / radius:
                continue
            # An order n is possible when the atom sits within the
            # tolerance of the angle 2π/n, and no further than the class
            # size, the length of an orbit off the axis.
            order = round(2.0 * math.pi / angle)
            if 2 <= order <= len(points) and (
                abs(angle - 2.0 * math.pi / order) * radius <= 2.0 * geometry.tolerance
            ):
                orders.add(order)
        return sorted(orders, reverse=True)
    return []


@dataclass(frozen=True, slots=True)
class _Elements:
    """The symmetry elements of a molecule."""

    rotations: tuple[tuple[Vector, int], ...]
    """Axes with the highest order of rotation they carry, at least two."""
    mirrors: tuple[Vector, ...]
    """Normals of the mirror planes."""
    improper: tuple[tuple[Vector, int], ...]
    """Axes carrying an improper rotation ``S_2n`` with the order ``2n``."""
    inversion: bool
    tolerance: float

    def perpendicular_twofold(self, axis: Vector) -> tuple[Vector, ...]:
        return tuple(
            other
            for other, _ in self.rotations
            if _is_perpendicular(axis, other, self.tolerance)
        )

    def has_horizontal_mirror(self, axis: Vector) -> bool:
        return any(
            _is_parallel(axis, normal, self.tolerance) for normal in self.mirrors
        )

    def vertical_mirrors(self, axis: Vector) -> tuple[Vector, ...]:
        return tuple(
            normal
            for normal in self.mirrors
            if _is_perpendicular(axis, normal, self.tolerance)
        )

    def has_improper(self, axis: Vector, order: int) -> bool:
        return any(
            _is_parallel(axis, other, self.tolerance) and other_order == order
            for other, other_order in self.improper
        )


def _find_elements(geometry: _Geometry) -> _Elements:
    """Test every candidate direction; parallel elements are recorded once."""
    tolerance = geometry.tolerance
    rotations: list[tuple[Vector, int]] = []
    mirrors: list[Vector] = []
    for axis in _candidate_directions(geometry):
        if not any(_is_parallel(axis, other, tolerance) for other, _ in rotations):
            for order in _rotation_orders(geometry, axis):
                if geometry.has_rotation(axis, order):
                    rotations.append((axis, order))
                    break
        if not any(
            _is_parallel(axis, other, tolerance) for other in mirrors
        ) and geometry.has_mirror(axis):
            mirrors.append(axis)
    improper = tuple(
        (axis, 2 * order)
        for axis, order in rotations
        if geometry.is_symmetric(_improper(axis, 2 * order))
    )
    return _Elements(
        tuple(rotations),
        tuple(mirrors),
        improper,
        geometry.is_symmetric(_INVERSION),
        geometry.tolerance,
    )


# ------------------------------------------------------------------ classification


def _classify(elements: _Elements) -> tuple[str, int, Vector | None]:
    """Return the Schoenflies symbol, the principal order and the principal axis.

    Cubic and icosahedral groups have no principal axis; groups without a
    rotation axis have order one and, for Cs, the mirror normal as axis.
    """
    high = [(axis, order) for axis, order in elements.rotations if order >= 3]
    if len(high) >= 2:
        return _cubic_name(elements, high), 0, None
    if not elements.rotations:
        if elements.mirrors:
            return "Cs", 1, elements.mirrors[0]
        return ("Ci" if elements.inversion else "C1"), 1, None
    n = max(order for _, order in elements.rotations)
    z = _principal_axis(elements, n)
    return _axial_name(elements, z, n), n, z


def _cubic_name(elements: _Elements, high: list[tuple[Vector, int]]) -> str:
    orders = {order for _, order in high}
    if 5 in orders:
        return "Ih" if elements.inversion else "I"
    if 4 in orders:
        return "Oh" if elements.inversion else "O"
    if elements.inversion:
        return "Th"
    return "Td" if elements.mirrors else "T"


def _principal_axis(elements: _Elements, n: int) -> Vector:
    """Return the unique axis of order ``n``, of a twofold set the ``S4`` axis."""
    axes = [axis for axis, order in elements.rotations if order == n]
    if len(axes) == 1:
        return axes[0]
    for axis, order in elements.improper:
        if order == 4:
            return axis
    return axes[0]


def _axial_name(elements: _Elements, z: Vector, n: int) -> str:
    if elements.perpendicular_twofold(z):
        if elements.has_horizontal_mirror(z):
            return f"D{n}h"
        if elements.has_improper(z, 2 * n):
            return f"D{n}d"
        return f"D{n}"
    if elements.has_horizontal_mirror(z):
        return f"C{n}h"
    if elements.vertical_mirrors(z):
        return f"C{n}v"
    if elements.has_improper(z, 2 * n):
        return f"S{2 * n}"
    return f"C{n}"


# ------------------------------------------------------------------ the frame


def _frame(
    name: str, n: int, z: Vector | None, elements: _Elements, geometry: _Geometry
) -> tuple[Frame, str]:
    """Return the standard frame and the Abelian subgroup of the full group."""
    if name in _CUBIC:
        return _cubic_frame(elements), ("D2h" if elements.inversion else "D2")
    if name == "C1":
        return IDENTITY_FRAME, "C1"
    if name == "Ci":
        return IDENTITY_FRAME, "Ci"
    assert z is not None
    if name == "Cs":
        return _asymmetric_frame(z, name, elements, geometry), "Cs"
    if n == 2 and name in ("D2", "D2h"):
        return _twofold_frame(name, elements, geometry), name
    if n == 2 and name != "D2d":
        subgroup = "C2" if name == "S4" else name
        return _asymmetric_frame(z, name, elements, geometry), subgroup
    return _axial_frame(name, n, z, elements, geometry)


def _cubic_frame(elements: _Elements) -> Frame:
    """Three mutually perpendicular fourfold or twofold axes, closest to the input."""
    orders = {order for _, order in elements.rotations}
    best = 4 if 4 in orders else 2
    axes = [axis for axis, order in elements.rotations if order == best]
    tolerance = elements.tolerance
    triples = [
        (a, b, c)
        for i, a in enumerate(axes)
        for j, b in enumerate(axes[i + 1 :], start=i + 1)
        if _is_perpendicular(a, b, tolerance)
        for c in axes[j + 1 :]
        if _is_perpendicular(a, c, tolerance) and _is_perpendicular(b, c, tolerance)
    ]
    if not triples:
        raise SymmetryError(
            "no three mutually perpendicular symmetry axes were found for a cubic "
            "group; the tolerance admits spurious elements"
        )
    frames = [_closest_frame(triple) for triple in triples]
    return max(frames, key=lambda frame: frame[0][0] + frame[1][1] + frame[2][2])


def _twofold_frame(name: str, elements: _Elements, geometry: _Geometry) -> Frame:
    """D2 axes closest to the input; D2h axes by extent, planar molecules in ``yz``."""
    axes = tuple(axis for axis, _ in elements.rotations)[:3]
    if name == "D2":
        return _closest_frame(axes)  # type: ignore[arg-type]
    ordered = sorted(axes, key=geometry.extent_along)
    planar = [axis for axis in ordered if geometry.is_planar_along(axis)]
    if not planar:
        return _proper((ordered[0], ordered[1], ordered[2]))
    x = planar[0]
    rest = [axis for axis in ordered if axis is not x]
    rest.sort(
        key=lambda axis: (
            geometry.count_displaced_along(axis),
            geometry.extent_along(axis),
        )
    )
    return _proper(_make_frame(rest[1], x))


def _asymmetric_frame(
    z: Vector, name: str, elements: _Elements, geometry: _Geometry
) -> Frame:
    """Put ``z`` along the twofold axis or mirror normal, the rest cyclically."""
    tolerance = geometry.tolerance
    parallel = [k for k in range(3) if _is_parallel(z, geometry.axes[k], tolerance)]
    distinct = all(
        abs(geometry.extents[(k + 1) % 3] - geometry.extents[(k + 2) % 3])
        > tolerance * max(1.0, sum(geometry.extents))
        for k in parallel
    )
    if parallel and distinct:
        k = parallel[0]
        frame = _make_frame(geometry.axes[k], geometry.axes[(k + 1) % 3])
    else:
        mirrors = elements.vertical_mirrors(z) if name == "C2v" else ()
        x = (
            _aligned(mirrors, IDENTITY_FRAME[0])
            if mirrors
            else _input_axis_off(z, tolerance)
        )
        frame = _make_frame(z, x)
    if name == "C2v" and geometry.is_planar_along(frame[1]):
        frame = (_scale(-1.0, frame[1]), frame[0], frame[2])
    return _proper(frame)


def _input_axis_off(z: Vector, tolerance: float) -> Vector:
    """Return the first input axis not parallel to ``z``."""
    for axis in IDENTITY_FRAME:
        if not _is_parallel(z, axis, tolerance):
            return axis
    raise AssertionError("no input axis is off a unit vector")


def _axial_frame(
    name: str, n: int, z: Vector, elements: _Elements, geometry: _Geometry
) -> tuple[Frame, str]:
    """Frame and subgroup of a group with a principal axis of order ``n``."""
    tolerance = geometry.tolerance
    family = name[0] + name[len(str(n)) + 1 :]
    if family in ("Dh", "Dd", "D"):
        x = _perpendicular_axis(elements.perpendicular_twofold(z), geometry)
    elif family == "Cv":
        # The normal becomes z for odd n and x for even n (below); the one
        # closest to that input axis keeps a standard frame standard.
        target = IDENTITY_FRAME[2] if n % 2 else IDENTITY_FRAME[0]
        x = _aligned(elements.vertical_mirrors(z), target)
    else:
        x = _input_axis_off(z, tolerance)
    frame = _make_frame(z, x)
    if n % 2 == 0:
        subgroup = {"Dh": "D2h", "Dd": "D2", "D": "D2", "Ch": "C2h", "Cv": "C2v"}
        return _proper(frame), subgroup.get(family, "C2")
    if family in ("Dh", "Dd", "D", "Cv"):
        frame = (frame[1], frame[2], frame[0])
        subgroup = {"Dh": "C2v", "Dd": "C2h", "D": "C2", "Cv": "Cs"}[family]
        return _proper(frame), subgroup
    return _proper(frame), {"Ch": "Cs", "S": "Ci"}.get(family, "C1")


def _perpendicular_axis(axes: tuple[Vector, ...], geometry: _Geometry) -> Vector:
    """Return the perpendicular twofold axis through the first atom or pair."""
    tolerance = geometry.tolerance
    for point in geometry.points:
        for axis in axes:
            if _norm(point) > tolerance and _is_parallel(_unit(point), axis, tolerance):
                return axis
    for i, first in enumerate(geometry.points):
        for second in geometry.points[i + 1 :]:
            middle = _add(first, second)
            for axis in axes:
                if _norm(middle) > tolerance and _is_parallel(
                    _unit(middle), axis, tolerance
                ):
                    return axis
    return axes[0]


def _aligned(directions: tuple[Vector, ...], target: Vector) -> Vector:
    """Return the direction most parallel to ``target``, the first among ties."""
    return max(
        directions,
        key=lambda direction: (
            abs(_dot(direction, target)) - 1e-12 * directions.index(direction)
        ),
    )


# ------------------------------------------------------------------ the result


@dataclass(frozen=True, slots=True)
class PointGroupDetection:
    """The point group of a molecule and the frame of its Abelian subgroup.

    Parameters
    ----------
    full_group
        Schoenflies symbol of the full point group, ``Dooh``, ``Coov`` and
        ``SO3`` for linear molecules and an atom.
    group
        The Abelian subgroup imposed on calculations, in its standard frame.
    origin
        Centre of nuclear charge in the input frame, Bohr; the origin of
        the standard frame.
    axes
        Rows: the standard ``x``, ``y`` and ``z`` axes in the input frame.
    """

    full_group: str
    group: PointGroup
    origin: Vector
    axes: Frame
    tolerance: float

    @property
    def is_standard(self) -> bool:
        """Whether the input frame already is the standard frame."""
        return _norm(self.origin) < self.tolerance and all(
            abs(self.axes[i][j] - IDENTITY_FRAME[i][j]) < self.tolerance
            for i in range(3)
            for j in range(3)
        )

    def orient(self, molecule: Molecule) -> Molecule:
        """Return ``molecule`` moved into the standard frame.

        The positions become ``axes @ (r − origin)``: a proper rotation after
        a translation, so bond lengths and chirality are preserved.
        """
        atoms = tuple(
            Atom(
                atom.symbol,
                tuple(
                    _dot(axis, _sub(atom.position, self.origin)) for axis in self.axes
                ),  # type: ignore[arg-type]
            )
            for atom in molecule.atoms
        )
        return replace(molecule, atoms=atoms)


def detect_point_group(
    molecule: Molecule, *, tolerance: float = DEFAULT_SYMMETRY_TOLERANCE
) -> PointGroupDetection:
    """Detect the point group of ``molecule`` and the frame of its Abelian subgroup.

    Atoms are equivalent when they are the same element; the basis set is
    checked for the same symmetry when the symmetry adaptation is built.
    Every operation must map each atom onto an atom within ``tolerance``
    Bohr. A C1 molecule keeps its frame; every other molecule is centred
    at its centre of nuclear charge.
    """
    if not tolerance > 0.0:
        raise InputError("the symmetry tolerance is positive")
    geometry, origin = _Geometry.build(molecule, tolerance)
    if molecule.n_atoms == 1:
        return PointGroupDetection(
            "SO3", PointGroup.named("D2h"), origin, IDENTITY_FRAME, tolerance
        )
    axis = _linear_axis(geometry)
    if axis is not None:
        name = "Dooh" if geometry.is_symmetric(_INVERSION) else "Coov"
        frame = _proper(_make_frame(axis, _input_axis_off(axis, tolerance)))
        group = PointGroup.named("D2h" if name == "Dooh" else "C2v")
        return PointGroupDetection(name, group, origin, frame, tolerance)
    elements = _find_elements(geometry)
    name, n, z = _classify(elements)
    frame, subgroup = _frame(name, n, z, elements, geometry)
    if name == "C1":
        origin = _ORIGIN
    return PointGroupDetection(
        name, PointGroup.named(subgroup), origin, frame, tolerance
    )


def _linear_axis(geometry: _Geometry) -> Vector | None:
    """Return the unit direction of a linear molecule, ``None`` for any other."""
    farthest = max(geometry.points, key=_norm)
    axis = _unit(farthest)
    if all(
        _norm(_cross(point, axis)) < geometry.tolerance for point in geometry.points
    ):
        return axis
    return None
