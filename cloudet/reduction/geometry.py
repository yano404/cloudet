"""Constructive geometry on planes, lines, and points (mm).

Operations for survey reduction: offset planes from drawing dimensions,
intersect planes into virtual axes, and intersect axes with faces to
obtain analysis parameters (e.g. beam-on-target).

Planes use the same Hesse form as ``cloudet.plane.Plane``:
``n · x + d = 0`` with ``|n| = 1``. Offset distance is positive along
the plane normal (after the Plane sign convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cloudet.core.plane import Plane

__all__ = [
    "Line",
    "offset_plane",
    "intersect_planes",
    "intersect_three_planes",
    "intersect_line_plane",
    "intersect_normal_plane",
    "line_from_point_normal",
    "line_from_two_points",
    "midpoint_line_planes",
    "plane_from_line_point",
    "plane_from_plane_point",
    "plane_from_two_lines",
    "rotate_line_about_line",
    "rotate_plane_about_line",
    "rotate_point_about_line",
    "plane_patch_corners",
    "line_segment_points",
    "axis_arrow_points",
    "distance_points",
    "distance_point_plane",
    "distance_point_line",
    "angle_planes_deg",
    "angle_lines_deg",
    "angle_line_plane_deg",
    "project_point_to_plane",
    "project_point_to_line",
]

_PARALLEL_EPS = 1e-12


def _unit_direction(direction, *, fix_sign: bool) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("line direction must be finite and non-zero")
    d = d / n
    if fix_sign:
        k = int(np.argmax(np.abs(d)))
        if d[k] < 0:
            d = -d
    return d


@dataclass(frozen=True)
class Line:
    """Parametric line: ``x(t) = point + t * direction``, ``|direction| = 1``."""

    point: np.ndarray  # (3,)
    direction: np.ndarray  # (3,), unit

    def __post_init__(self):
        p = np.asarray(self.point, dtype=np.float64).reshape(3)
        d = _unit_direction(self.direction, fix_sign=True)
        object.__setattr__(self, "point", p)
        object.__setattr__(self, "direction", d)

    @classmethod
    def from_point_direction(
        cls, point, direction, *, fix_sign: bool = True
    ) -> "Line":
        """Build a line; ``fix_sign=False`` keeps the given direction hemisphere."""
        obj = object.__new__(cls)
        object.__setattr__(
            obj, "point", np.asarray(point, dtype=np.float64).reshape(3).copy()
        )
        object.__setattr__(
            obj, "direction", _unit_direction(direction, fix_sign=fix_sign)
        )
        return obj

    def point_at(self, t: float) -> np.ndarray:
        return self.point + float(t) * self.direction


def offset_plane(plane: Plane, distance_mm: float) -> Plane:
    """Return a plane parallel to ``plane``, shifted by ``distance_mm``.

    Positive ``distance_mm`` moves the plane in the direction of its
    unit normal (Hesse sign convention). A point that lay on the original
    plane then has signed distance ``-distance_mm`` to the new plane.
    """
    # n·x + d = 0; after shift x' = x + distance * n on the new plane:
    # n·x' + (d - distance) = 0.
    return Plane(plane.normal, plane.d - float(distance_mm))


def intersect_planes(p1: Plane, p2: Plane) -> Line:
    """Intersection line of two non-parallel planes."""
    n1 = p1.normal
    n2 = p2.normal
    direction = np.cross(n1, n2)
    denom = float(direction @ direction)
    if denom < _PARALLEL_EPS:
        raise ValueError("planes are parallel; no unique intersection line")
    # Point on both planes, closest to origin among those with dir·x = 0.
    # p = ((n2 × dir)*(-d1) + (dir × n1)*(-d2)) / |dir|^2
    point = (np.cross(n2, direction) * (-p1.d) + np.cross(direction, n1) * (-p2.d)) / denom
    return Line(point, direction)


def intersect_three_planes(p1: Plane, p2: Plane, p3: Plane) -> np.ndarray:
    """Intersection point of three planes (unique if normals are independent)."""
    A = np.stack([p1.normal, p2.normal, p3.normal], axis=0)
    b = np.array([-p1.d, -p2.d, -p3.d], dtype=np.float64)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        raise ValueError("planes do not intersect in a unique point") from e


def intersect_line_plane(line: Line, plane: Plane) -> np.ndarray:
    """Intersection point of a line and a non-parallel plane."""
    nd = float(plane.normal @ line.direction)
    if abs(nd) < _PARALLEL_EPS:
        raise ValueError("line is parallel to plane; no unique intersection")
    t = -(float(plane.normal @ line.point) + plane.d) / nd
    return line.point_at(t)


def intersect_normal_plane(
    plane_src: Plane,
    plane_dst: Plane,
    through: np.ndarray | None = None,
) -> np.ndarray:
    """Intersect the ray along ``plane_src``'s normal with ``plane_dst``.

    The ray is ``through + t * plane_src.normal``. If ``through`` is
    omitted, the projection of the origin onto ``plane_src`` is used
    (``through = -d * n``).
    """
    if through is None:
        through = -plane_src.d * plane_src.normal
    else:
        through = np.asarray(through, dtype=np.float64).reshape(3)
    line = Line(through, plane_src.normal)
    return intersect_line_plane(line, plane_dst)


def line_from_point_normal(point: np.ndarray, normal) -> Line:
    """Line through ``point`` in the direction of ``normal``.

    ``normal`` may be a length-3 vector or a ``Plane`` (its Hesse unit
    normal). The point need not lie on that plane.
    """
    if isinstance(normal, Plane):
        n = np.asarray(normal.normal, dtype=np.float64)
    else:
        n = np.asarray(normal, dtype=np.float64).reshape(3)
    return Line(np.asarray(point, dtype=np.float64).reshape(3), n)


def line_from_two_points(p1: np.ndarray, p2: np.ndarray) -> Line:
    """Line through two distinct points. Origin is ``p1``."""
    a = np.asarray(p1, dtype=np.float64).reshape(3)
    b = np.asarray(p2, dtype=np.float64).reshape(3)
    d = b - a
    if float(d @ d) < _PARALLEL_EPS:
        raise ValueError("points coincide; no unique line")
    return Line(a, d)


def midpoint_line_planes(line: Line, plane_a: Plane, plane_b: Plane) -> np.ndarray:
    """Midpoint of the segment where ``line`` meets two planes."""
    a = intersect_line_plane(line, plane_a)
    b = intersect_line_plane(line, plane_b)
    return 0.5 * (a + b)


def plane_from_plane_point(plane: Plane, point: np.ndarray) -> Plane:
    """Plane parallel to ``plane`` passing through ``point``."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    n = plane.normal
    return Plane(n, -float(n @ p))


def plane_from_line_point(line: Line, point: np.ndarray) -> Plane:
    """Plane through ``point`` with normal = the line direction."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    n = line.direction
    return Plane(n, -float(n @ p))


def plane_from_two_lines(line_a: Line, line_b: Line) -> Plane:
    """Plane containing two non-skew lines.

    Parallel lines must be coplanar; intersecting lines must meet. Skew or
    collinear lines are an error.
    """
    d1 = line_a.direction
    d2 = line_b.direction
    w = line_b.point - line_a.point
    cross = np.cross(d1, d2)
    cross_norm = float(cross @ cross)

    if cross_norm < _PARALLEL_EPS:
        sep = np.cross(d1, w)
        sep_norm = float(sep @ sep)
        if sep_norm < _PARALLEL_EPS:
            raise ValueError("lines are collinear; no unique plane")
        n = sep / np.sqrt(sep_norm)
        return Plane(n, -float(n @ line_a.point))

    triple = float(w @ cross)
    if abs(triple) > _PARALLEL_EPS * max(1.0, float(np.linalg.norm(w))):
        raise ValueError("lines are skew; no common plane")

    n = cross / np.sqrt(cross_norm)
    a11 = float(d1 @ d1)
    a12 = float(d1 @ d2)
    a22 = float(d2 @ d2)
    b1 = float(d1 @ w)
    b2 = float(d2 @ w)
    denom = a11 * a22 - a12 * a12
    if abs(denom) < _PARALLEL_EPS:
        raise ValueError("lines do not define a unique plane")
    t = (a12 * b2 - a22 * b1) / denom
    pt = line_a.point + t * d1
    return Plane(n, -float(n @ pt))


def _rotate_vector_about_axis(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    k = np.asarray(axis, dtype=np.float64).reshape(3)
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return v * c + np.cross(k, v) * s + k * float(k @ v) * (1.0 - c)


def _rotate_point_about_axis(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    rel = np.asarray(point, dtype=np.float64).reshape(3) - np.asarray(
        origin, dtype=np.float64
    ).reshape(3)
    return np.asarray(origin, dtype=np.float64).reshape(3) + _rotate_vector_about_axis(
        rel, direction, angle_rad
    )


def rotate_plane_about_line(plane: Plane, axis: Line, angle_deg: float) -> Plane:
    """Rotate ``plane`` as a rigid object about ``axis`` by ``angle_deg``.

    The axis need not lie in the plane. Positive angle follows the
    right-hand rule around ``axis.direction``. Rotation about an axis
    parallel to the plane normal leaves the plane equation unchanged.
    """
    k = axis.direction
    a = np.asarray(axis.point, dtype=np.float64).reshape(3)
    theta = float(np.deg2rad(angle_deg))
    n_rot = _rotate_vector_about_axis(plane.normal, k, theta)
    p0 = a - plane.signed_distances(a.reshape(1, 3))[0] * plane.normal
    p_rot = _rotate_point_about_axis(p0, a, k, theta)
    return Plane(n_rot, -float(n_rot @ p_rot))


def rotate_point_about_line(
    point: np.ndarray, axis: Line, angle_deg: float
) -> np.ndarray:
    """Rotate *point* about *axis* by *angle_deg* (right-hand rule)."""
    theta = float(np.deg2rad(angle_deg))
    return _rotate_point_about_axis(
        np.asarray(point, dtype=np.float64).reshape(3),
        axis.point,
        axis.direction,
        theta,
    )


def rotate_line_about_line(target: Line, axis: Line, angle_deg: float) -> Line:
    """Rotate *target* line rigidly about *axis* by *angle_deg*."""
    theta = float(np.deg2rad(angle_deg))
    p_rot = _rotate_point_about_axis(target.point, axis.point, axis.direction, theta)
    d_rot = _rotate_vector_about_axis(target.direction, axis.direction, theta)
    return Line(p_rot, d_rot)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)
    return u, v


def plane_patch_corners(
    plane: Plane,
    *,
    center: np.ndarray | None = None,
    size_mm: float = 200.0,
) -> np.ndarray:
    """Return (4, 3) corners of a square patch lying on ``plane``."""
    if center is None:
        center = -plane.d * plane.normal
    else:
        # Project onto plane so the patch is coplanar.
        center = np.asarray(center, dtype=np.float64).reshape(3)
        center = center - plane.signed_distances(center.reshape(1, 3))[0] * plane.normal
    u, v = _plane_basis(plane.normal)
    h = 0.5 * float(size_mm)
    return np.stack([
        center + h * (-u - v),
        center + h * (u - v),
        center + h * (u + v),
        center + h * (-u + v),
    ])


def line_segment_points(
    line: Line,
    *,
    half_length_mm: float = 300.0,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Return (2, 3) endpoints of a finite segment along ``line``."""
    if center is None:
        c = line.point
    else:
        # Project center onto the line.
        c0 = np.asarray(center, dtype=np.float64).reshape(3)
        t = float((c0 - line.point) @ line.direction)
        c = line.point + t * line.direction
    h = float(half_length_mm)
    return np.stack([c - h * line.direction, c + h * line.direction])


def axis_arrow_points(line: Line, length_mm: float) -> np.ndarray:
    """Return (2, 3) origin → ``+direction * length_mm`` (FRAME triad style)."""
    origin = np.asarray(line.point, dtype=np.float64).reshape(3)
    d = np.asarray(line.direction, dtype=np.float64).reshape(3)
    return np.stack([origin, origin + float(length_mm) * d])


def distance_points(point_a, point_b) -> float:
    """Euclidean distance between two points (mm)."""
    pa = np.asarray(point_a, dtype=np.float64).reshape(3)
    pb = np.asarray(point_b, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(pb - pa))


def distance_point_plane(point, plane: Plane) -> float:
    """Perpendicular distance from a point to a plane (mm)."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    return float(abs(plane.signed_distances(p.reshape(1, 3))[0]))


def distance_point_line(point, line: Line) -> float:
    """Perpendicular distance from a point to a line (mm)."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(np.cross(p - line.point, line.direction)))


def angle_planes_deg(plane_a: Plane, plane_b: Plane) -> float:
    """Smallest angle between two planes, in degrees in ``[0, 90]``."""
    return float(np.degrees(plane_a.angle_to(plane_b)))


def angle_lines_deg(line_a: Line, line_b: Line) -> float:
    """Smallest angle between two line directions, in degrees in ``[0, 90]``."""
    c = float(np.clip(abs(float(line_a.direction @ line_b.direction)), 0.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def angle_line_plane_deg(line: Line, plane: Plane) -> float:
    """Angle between a line and a plane, in degrees in ``[0, 90]``.

    Parallel to the plane is 0°; perpendicular to the plane is 90°.
    """
    s = float(np.clip(abs(float(line.direction @ plane.normal)), 0.0, 1.0))
    return float(np.degrees(np.arcsin(s)))


def project_point_to_plane(point, plane: Plane) -> np.ndarray:
    """Closest point on ``plane`` to ``point``."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    r = float(plane.signed_distances(p.reshape(1, 3))[0])
    return p - r * plane.normal


def project_point_to_line(point, line: Line) -> np.ndarray:
    """Closest point on ``line`` to ``point``."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    t = float((p - line.point) @ line.direction)
    return line.point_at(t)
