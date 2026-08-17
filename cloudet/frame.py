"""Rigid view frame: align a chosen axis to global +Z without mutating survey data.

The transform is ``x' = R (x − origin)``, where ``R`` is the smallest rotation
that maps the axis direction onto ``(0, 0, 1)``. Rotation about Z is left
unspecified (no extra yaw). Survey coordinates, Groups, and recipes stay in
the original frame; only display and optional geometry export apply this pose.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from cloudet.geometry import Line
from cloudet.plane import Plane

__all__ = [
    "RigidFrame",
    "rotation_mapping_to_z",
    "result_in_frame",
    "transform_record",
    "with_aligned_copy",
]

_ALIGN_EPS = 1e-12


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = (float(v[0]), float(v[1]), float(v[2]))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_mapping_to_z(direction: np.ndarray) -> np.ndarray:
    """Return the 3×3 rotation that maps ``direction`` onto ``(0, 0, 1)``.

    Uses the smallest rotation (Rodrigues). If the direction is already +Z,
    this is identity. If it is antiparallel, a 180° rotation about a stable
    perpendicular is used.
    """
    a = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("axis direction must be finite and non-zero")
    a = a / n
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = float(a @ z)
    if c > 1.0 - _ALIGN_EPS:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + _ALIGN_EPS:
        perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(a @ perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        perp = perp - (perp @ a) * a
        perp = perp / float(np.linalg.norm(perp))
        return 2.0 * np.outer(perp, perp) - np.eye(3, dtype=np.float64)
    v = np.cross(a, z)
    vx = _skew(v)
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / float(v @ v))


def _as_nx3(xyz) -> tuple[np.ndarray, bool]:
    p = np.asarray(xyz, dtype=np.float64)
    if p.ndim == 1:
        if p.shape != (3,):
            raise ValueError(f"expected (3,) or (N, 3), got {p.shape}")
        return p.reshape(1, 3), True
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (3,) or (N, 3), got {p.shape}")
    return p, False


@dataclass(frozen=True)
class RigidFrame:
    """Survey → view: ``x' = R (x − origin)``. ``R`` maps the axis to +Z."""

    origin: np.ndarray
    rotation: np.ndarray
    axis_id: str = ""
    origin_id: str = ""
    flip_z: bool = False

    def __post_init__(self):
        o = np.asarray(self.origin, dtype=np.float64).reshape(3).copy()
        r = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3).copy()
        object.__setattr__(self, "origin", o)
        object.__setattr__(self, "rotation", r)

    @classmethod
    def align_z(
        cls,
        direction,
        origin,
        *,
        flip_z: bool = False,
        axis_id: str = "",
        origin_id: str = "",
    ) -> "RigidFrame":
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        if flip_z:
            d = -d
        return cls(
            origin=np.asarray(origin, dtype=np.float64).reshape(3),
            rotation=rotation_mapping_to_z(d),
            axis_id=str(axis_id),
            origin_id=str(origin_id),
            flip_z=bool(flip_z),
        )

    def apply_points(self, xyz) -> np.ndarray:
        p, single = _as_nx3(xyz)
        out = (p - self.origin) @ self.rotation.T
        return out.reshape(3) if single else out

    def inverse_points(self, xyz) -> np.ndarray:
        p, single = _as_nx3(xyz)
        out = p @ self.rotation + self.origin
        return out.reshape(3) if single else out

    def apply_direction(self, direction) -> np.ndarray:
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        return self.rotation @ d

    def apply_plane(self, plane: Plane) -> Plane:
        n = np.asarray(plane.normal, dtype=np.float64).reshape(3)
        n2 = self.rotation @ n
        d2 = float(plane.d) + float(n @ self.origin)
        return Plane(n2, d2)

    def apply_line(self, line: Line) -> Line:
        return Line(self.apply_points(line.point), self.apply_direction(line.direction))

    def relabel(self, old_id: str, new_id: str) -> "RigidFrame":
        return RigidFrame(
            origin=self.origin,
            rotation=self.rotation,
            axis_id=new_id if self.axis_id == old_id else self.axis_id,
            origin_id=new_id if self.origin_id == old_id else self.origin_id,
            flip_z=self.flip_z,
        )

    def to_dict(self) -> dict:
        return {
            "kind": "aligned",
            "axis": self.axis_id,
            "origin": self.origin_id,
            "flip_z": self.flip_z,
            "origin_xyz": self.origin.tolist(),
            "rotation": self.rotation.tolist(),
        }


def transform_record(kind: str, record: dict, frame: RigidFrame) -> dict:
    """Copy an entity record and rewrite geometric fields into ``frame``."""
    rec = copy.deepcopy(record)
    if kind == "plane" and "abcd" in rec:
        rec["abcd"] = frame.apply_plane(Plane.from_array(rec["abcd"])).as_array().tolist()
        return rec
    if kind == "line":
        line = Line(rec["point"], rec["direction"])
        line2 = frame.apply_line(line)
        rec["point"] = line2.point.tolist()
        rec["direction"] = line2.direction.tolist()
        return rec
    if kind == "point":
        if "xyz" in rec:
            rec["xyz"] = frame.apply_points(rec["xyz"]).tolist()
        if rec.get("ends"):
            rec["ends"] = [frame.apply_points(e).tolist() for e in rec["ends"]]
        if rec.get("through") is not None:
            rec["through"] = frame.apply_points(rec["through"]).tolist()
        return rec
    return rec


def result_in_frame(result, frame: RigidFrame):
    """Return a copy of ``ReductionResult`` with geometry in ``frame``.

    Recipe echo stays in survey coordinates. ``result.frame`` describes the pose.
    """
    from cloudet.reduce import ReductionResult

    out = ReductionResult(
        recipe=copy.deepcopy(result.recipe),
        source_project=result.source_project,
        exported=list(result.exported),
        frame=frame.to_dict(),
    )
    for eid, rec in result.planes.items():
        out.planes[eid] = transform_record("plane", rec, frame)
    for eid, rec in result.lines.items():
        out.lines[eid] = transform_record("line", rec, frame)
    for eid, rec in result.points.items():
        out.points[eid] = transform_record("point", rec, frame)
    return out


def with_aligned_copy(result, frame: RigidFrame):
    """Keep survey geometry and attach an ``aligned`` copy plus ``frame`` pose."""
    from cloudet.reduce import ReductionResult

    aligned = result_in_frame(result, frame)
    return ReductionResult(
        planes=result.planes,
        lines=result.lines,
        points=result.points,
        recipe=result.recipe,
        source_project=result.source_project,
        exported=list(result.exported),
        frame=aligned.frame,
        aligned={
            "planes": aligned.planes,
            "lines": aligned.lines,
            "points": aligned.points,
        },
    )
