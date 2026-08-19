"""Rigid view frame: align a chosen axis to global +Z without mutating survey data.

The transform is ``x' = R (x − origin)``. ``R`` is the smallest rotation that
maps the axis onto ``(0, 0, 1)``, optionally followed by a rotation about Z
so another line's direction maps onto ±X or ±Y in the view. Survey
coordinates, Groups, and recipes stay in the original frame; only display
and optional geometry export apply this pose.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from cloudet.reduction.geometry import Line
from cloudet.core.plane import Plane

__all__ = [
    "ALIGNED_AXIS_IDS",
    "ALIGNED_AXIS_LABELS",
    "ALIGNED_ENTITY_IDS",
    "ALIGNED_KIND",
    "ALIGNED_LABELS",
    "ALIGNED_ORIGIN_ID",
    "ALIGNED_PLANE_IDS",
    "RigidFrame",
    "YAW_TO",
    "aligned_axis_line",
    "aligned_origin_point",
    "aligned_plane",
    "is_aligned_axis_id",
    "is_aligned_id",
    "is_aligned_origin_id",
    "is_aligned_plane_id",
    "result_in_frame",
    "rotation_mapping_to_z",
    "rotation_yaw_about_z",
    "transform_record",
    "with_aligned_copy",
]

_ALIGN_EPS = 1e-12
YAW_TO = ("x", "-x", "y", "-y")
ALIGNED_ORIGIN_ID = "aligned.origin"
ALIGNED_AXIS_IDS = ("aligned.x", "aligned.y", "aligned.z")
ALIGNED_PLANE_IDS = ("aligned.yz", "aligned.zx", "aligned.xy")
ALIGNED_ENTITY_IDS = (ALIGNED_ORIGIN_ID,) + ALIGNED_AXIS_IDS + ALIGNED_PLANE_IDS
ALIGNED_AXIS_LABELS = {
    "aligned.x": "aligned X axis",
    "aligned.y": "aligned Y axis",
    "aligned.z": "aligned Z axis",
}
ALIGNED_LABELS = {
    ALIGNED_ORIGIN_ID: "aligned origin",
    **ALIGNED_AXIS_LABELS,
    "aligned.yz": "aligned YZ plane",
    "aligned.zx": "aligned ZX plane",
    "aligned.xy": "aligned XY plane",
}
ALIGNED_KIND = {
    ALIGNED_ORIGIN_ID: "point",
    "aligned.x": "line",
    "aligned.y": "line",
    "aligned.z": "line",
    "aligned.yz": "plane",
    "aligned.zx": "plane",
    "aligned.xy": "plane",
}
_ALIGNED_AXIS_INDEX = {"aligned.x": 0, "aligned.y": 1, "aligned.z": 2}
_ALIGNED_PLANE_NORMAL_INDEX = {"aligned.yz": 0, "aligned.zx": 1, "aligned.xy": 2}


def is_aligned_id(entity_id: str) -> bool:
    return str(entity_id) in ALIGNED_KIND


def is_aligned_axis_id(entity_id: str) -> bool:
    return str(entity_id) in _ALIGNED_AXIS_INDEX


def is_aligned_plane_id(entity_id: str) -> bool:
    return str(entity_id) in _ALIGNED_PLANE_NORMAL_INDEX


def is_aligned_origin_id(entity_id: str) -> bool:
    return str(entity_id) == ALIGNED_ORIGIN_ID


def aligned_origin_point(frame: "RigidFrame") -> np.ndarray:
    """FRAME origin in survey coordinates."""
    return np.asarray(frame.origin, dtype=np.float64).reshape(3).copy()


def aligned_axis_line(frame: "RigidFrame", axis_id: str) -> Line:
    """Survey-frame line along an aligned axis, through the frame origin.

    Direction matches the view triad (+X/+Y/+Z), not ``Line``'s usual
    largest-component sign convention.
    """
    key = str(axis_id)
    if key not in _ALIGNED_AXIS_INDEX:
        raise KeyError(f"unknown aligned axis {axis_id!r}")
    direction = np.asarray(frame.rotation[_ALIGNED_AXIS_INDEX[key]], dtype=np.float64)
    return Line.from_point_direction(frame.origin, direction, fix_sign=False)


def aligned_plane(frame: "RigidFrame", plane_id: str) -> Plane:
    """Survey-frame coordinate plane through the FRAME origin.

    ``aligned.xy`` has normal +Z, ``aligned.yz`` normal +X, ``aligned.zx``
    normal +Y (view-triad signs).
    """
    key = str(plane_id)
    if key not in _ALIGNED_PLANE_NORMAL_INDEX:
        raise KeyError(f"unknown aligned plane {plane_id!r}")
    normal = np.asarray(
        frame.rotation[_ALIGNED_PLANE_NORMAL_INDEX[key]], dtype=np.float64
    )
    origin = np.asarray(frame.origin, dtype=np.float64).reshape(3)
    return Plane(normal, -float(normal @ origin))


_YAW_XY = {
    "x": np.array([1.0, 0.0], dtype=np.float64),
    "-x": np.array([-1.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, 1.0], dtype=np.float64),
    "-y": np.array([0.0, -1.0], dtype=np.float64),
}


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


def rotation_yaw_about_z(direction_view: np.ndarray, yaw_to: str) -> np.ndarray:
    """Rotate about +Z so the XY part of ``direction_view`` maps onto ``yaw_to``.

    ``yaw_to`` is one of ``x``, ``-x``, ``y``, ``-y``. The Z component is
    ignored (the line need not be horizontal). Parallel to Z has no XY
    projection and is an error.
    """
    key = str(yaw_to)
    if key not in _YAW_XY:
        raise ValueError(f"yaw_to must be one of {list(YAW_TO)}, got {yaw_to!r}")
    d = np.asarray(direction_view, dtype=np.float64).reshape(3)
    xy = d[:2]
    n = float(np.linalg.norm(xy))
    if not np.isfinite(n) or n < _ALIGN_EPS:
        raise ValueError("yaw reference is parallel to Z; cannot set XY")
    xy = xy / n
    target = _YAW_XY[key]
    delta = float(np.arctan2(target[1], target[0]) - np.arctan2(xy[1], xy[0]))
    c = float(np.cos(delta))
    s = float(np.sin(delta))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


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
    yaw_id: str = ""
    yaw_kind: str = ""  # "line" | "plane"
    yaw_to: str = ""

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
        yaw_direction=None,
        yaw_to: str | None = None,
        yaw_id: str = "",
        yaw_kind: str = "line",
    ) -> "RigidFrame":
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        if flip_z:
            d = -d
        rotation = rotation_mapping_to_z(d)
        yaw_key = str(yaw_to or "")
        kind = str(yaw_kind or "line")
        if yaw_direction is not None:
            if not yaw_key:
                yaw_key = "x"
            if kind not in ("line", "plane"):
                raise ValueError(f"yaw_kind must be 'line' or 'plane', got {kind!r}")
            yaw_view = rotation @ np.asarray(yaw_direction, dtype=np.float64).reshape(3)
            rotation = rotation_yaw_about_z(yaw_view, yaw_key) @ rotation
        elif yaw_key:
            raise ValueError("yaw_to needs a yaw reference direction")
        return cls(
            origin=np.asarray(origin, dtype=np.float64).reshape(3),
            rotation=rotation,
            axis_id=str(axis_id),
            origin_id=str(origin_id),
            flip_z=bool(flip_z),
            yaw_id=str(yaw_id) if yaw_direction is not None else "",
            yaw_kind=kind if yaw_direction is not None else "",
            yaw_to=yaw_key if yaw_direction is not None else "",
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
            yaw_id=new_id if self.yaw_id == old_id else self.yaw_id,
            yaw_kind=self.yaw_kind,
            yaw_to=self.yaw_to,
        )

    def to_dict(self) -> dict:
        out = {
            "kind": "aligned",
            "axis": self.axis_id,
            "origin": self.origin_id,
            "flip_z": self.flip_z,
            "origin_xyz": self.origin.tolist(),
            "rotation": self.rotation.tolist(),
        }
        if self.yaw_id and self.yaw_to:
            yaw_key = "yaw_plane" if self.yaw_kind == "plane" else "yaw_line"
            out[yaw_key] = self.yaw_id
            out["yaw_to"] = self.yaw_to
        return out


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
    from cloudet.reduction.session import ReductionResult

    out = ReductionResult(
        recipe=copy.deepcopy(result.recipe),
        source_project=result.source_project,
        exported=list(result.exported),
        frame=frame.to_dict(),
        measures=copy.deepcopy(result.measures) if result.measures else None,
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
    from cloudet.reduction.session import ReductionResult

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
        measures=list(result.measures) if result.measures else None,
    )
