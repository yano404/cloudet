"""Declarative geometry reduction from a saved project + recipe.

Scanned faces (fit abcd) are bound by group name, then construct steps
build offset planes, intersection lines, and points for analysis export.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cloudet.reduction.geometry import (
    Line,
    angle_lines_deg,
    angle_line_plane_deg,
    angle_planes_deg,
    distance_point_line,
    distance_point_plane,
    distance_points,
    intersect_line_plane,
    intersect_normal_plane,
    intersect_planes,
    intersect_three_planes,
    line_from_point_normal,
    line_from_two_points,
    midpoint_line_planes,
    offset_plane,
    plane_from_line_point,
    plane_from_plane_point,
    plane_from_two_lines,
    project_point_to_plane,
    rotate_line_about_line,
    rotate_plane_about_line,
    rotate_point_about_line,
)
from cloudet.reduction.frame import (
    ALIGNED_AXIS_IDS,
    ALIGNED_ENTITY_IDS,
    ALIGNED_KIND,
    ALIGNED_ORIGIN_ID,
    ALIGNED_PLANE_IDS,
    aligned_axis_line,
    aligned_origin_point,
    aligned_plane,
    is_aligned_axis_id,
    is_aligned_id,
    is_aligned_origin_id,
    is_aligned_plane_id,
)
from cloudet.core.plane import Plane
from cloudet.project.schema import (
    RECIPE_VERSION,
    migrate_construct_step,
    migrate_measure_spec,
    migrate_recipe,
    plane_from_json,
    plane_to_json,
)
from cloudet.project import (
    FittedPlane,
    load_fitted_circle,
    load_fitted_cylinder,
    load_fitted_plane,
    load_group_doc,
)
from cloudet.reduction.ops import MEASURE_OPERAND_FIELDS, REDUCTION_OP_BY_RECIPE

__all__ = [
    "ConstructPreview",
    "ReductionResult",
    "ReductionSession",
    "export_reduction_result",
    "geometry_summary_dict",
    "geometry_summary_path",
    "load_recipe",
    "preview_construct_step",
    "run_reduction",
    "scanned_plane_record",
    "scanned_cylinder_line_record",
    "scanned_circle_point_record",
    "write_geometry_json",
    "write_geometry_summary_json",
    "write_recipe_json",
]

_SCANNED_QUALITY_KEYS = (
    "status",
    "mad_sigma_mm",
    "threshold_mm",
    "n_points",
    "bimodal",
    "reasons",
)


def scanned_plane_record(
    plane: Plane,
    *,
    group_id: int,
    group_name: str,
    plane_index: int = 0,
    quality: dict | None = None,
) -> dict:
    """Record dict for a scanned (Groups) plane binding."""
    quality = dict(quality or {})
    return {
        **plane_to_json(plane),
        "provenance": "scanned",
        "group_id": int(group_id),
        "group_name": str(group_name),
        "plane_index": int(plane_index),
        "quality": {
            k: quality[k]
            for k in _SCANNED_QUALITY_KEYS
            if quality.get(k) is not None
        },
    }


def scanned_cylinder_line_record(
    cyl,
    *,
    group_id: int,
    group_name: str,
    cylinder_index: int = 0,
    quality: dict | None = None,
) -> dict:
    """Record for a scanned cylinder bound as an axis line."""
    quality = dict(quality or {})
    return {
        "point": np.asarray(cyl.point, dtype=np.float64).tolist(),
        "direction": np.asarray(cyl.direction, dtype=np.float64).tolist(),
        "provenance": "scanned",
        "source_kind": "cylinder",
        "group_id": int(group_id),
        "group_name": str(group_name),
        "cylinder_index": int(cylinder_index),
        "diameter_mm": float(cyl.diameter_mm),
        "diameter_fixed": bool(cyl.diameter_fixed),
        "quality": {
            k: quality[k]
            for k in _SCANNED_QUALITY_KEYS
            if quality.get(k) is not None
        },
    }


def scanned_circle_point_record(
    cir,
    *,
    group_id: int,
    group_name: str,
    circle_index: int = 0,
    quality: dict | None = None,
) -> dict:
    """Record for a scanned circle bound as its center point."""
    quality = dict(quality or {})
    return {
        "xyz": np.asarray(cir.center, dtype=np.float64).tolist(),
        "provenance": "scanned",
        "source_kind": "circle",
        "group_id": int(group_id),
        "group_name": str(group_name),
        "circle_index": int(circle_index),
        "diameter_mm": float(cir.diameter_mm),
        "diameter_fixed": bool(cir.diameter_fixed),
        "normal": np.asarray(cir.normal, dtype=np.float64).tolist(),
        "quality": {
            k: quality[k]
            for k in _SCANNED_QUALITY_KEYS
            if quality.get(k) is not None
        },
    }



@dataclass
class _Entity:
    kind: str  # plane | line | point
    value: Any
    record: dict


@dataclass
class ReductionResult:
    planes: dict[str, dict] = field(default_factory=dict)
    lines: dict[str, dict] = field(default_factory=dict)
    points: dict[str, dict] = field(default_factory=dict)
    recipe: dict = field(default_factory=dict)
    source_project: str = ""
    exported: list[str] = field(default_factory=list)
    # Optional view pose used when writing aligned coordinates. Survey
    # recipe echo is never transformed.
    frame: dict | None = None
    # Optional copy of planes/lines/points in the aligned frame.
    aligned: dict | None = None
    # Evaluated measurements (distances / angles). Invariant under Align Z.
    measures: list[dict] | None = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "version": RECIPE_VERSION,
            "units": "mm",
            "source_project": self.source_project,
            "recipe": self.recipe,
            "planes": self.planes,
            "lines": self.lines,
            "points": self.points,
        }
        if self.exported:
            out["export"] = list(self.exported)
        if self.frame:
            out["frame"] = self.frame
        if self.aligned:
            out["aligned"] = self.aligned
        if self.measures:
            out["measures"] = list(self.measures)
        return out


def load_recipe(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if int(doc.get("version", 1)) > RECIPE_VERSION:
        raise ValueError(f"unsupported recipe version {doc.get('version')}")
    if doc.get("units", "mm") != "mm":
        raise ValueError(f"recipe units must be mm, got {doc.get('units')!r}")
    return migrate_recipe(doc)


def _recipe_fingerprint(recipe: dict) -> dict:
    raw = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "echo": recipe}


def _bind_face(project_dir: Path, alias: str, spec: dict) -> tuple[str, Any, dict]:
    """Resolve a recipe face to ``(kind, value, record)``.

    ``kind`` is ``plane``, ``line`` (cylinder axis), or ``point`` (circle center).
    """
    if not isinstance(spec, dict):
        raise ValueError(f"faces.{alias}: expected object, got {type(spec).__name__}")
    src = spec.get("from", "group")
    if src != "group":
        raise ValueError(f"faces.{alias}: unsupported from={src!r}")

    name = spec.get("name")
    group_id = spec.get("group_id")
    if name is not None and group_id is not None:
        raise ValueError(f"faces.{alias}: provide name or group_id, not both")
    if name is None and group_id is None:
        raise ValueError(f"faces.{alias}: need name or group_id")

    kind = str(spec.get("kind", "plane")).lower()
    if kind in ("", "plane", "face"):
        plane_index = int(spec.get("plane_index", 0))
        fitted: FittedPlane = load_fitted_plane(
            project_dir,
            name=None if name is None else str(name),
            group_id=None if group_id is None else int(group_id),
            plane_index=plane_index,
        )
        record = scanned_plane_record(
            fitted.plane,
            group_id=fitted.group_id,
            group_name=fitted.group_name,
            plane_index=fitted.plane_index,
            quality=fitted.quality,
        )
        return "plane", fitted.plane, record

    if kind == "cylinder":
        cylinder_index = int(spec.get("cylinder_index", 0))
        fitted_c = load_fitted_cylinder(
            project_dir,
            name=None if name is None else str(name),
            group_id=None if group_id is None else int(group_id),
            cylinder_index=cylinder_index,
        )
        cyl = fitted_c.cylinder
        # Recipe may override / document fixed diameter.
        if "diameter_mm" in spec:
            from cloudet.core.cylinder import Cylinder

            diam = float(spec["diameter_mm"])
            fixed = bool(spec.get("diameter_fixed", True))
            cyl = Cylinder(
                point=cyl.point,
                direction=cyl.direction,
                diameter_mm=diam,
                diameter_fixed=fixed,
            )
        line = Line(point=cyl.point, direction=cyl.direction)
        record = scanned_cylinder_line_record(
            cyl,
            group_id=fitted_c.group_id,
            group_name=fitted_c.group_name,
            cylinder_index=fitted_c.cylinder_index,
            quality=fitted_c.quality,
        )
        return "line", line, record

    if kind == "circle":
        circle_index = int(spec.get("circle_index", 0))
        fitted_r = load_fitted_circle(
            project_dir,
            name=None if name is None else str(name),
            group_id=None if group_id is None else int(group_id),
            circle_index=circle_index,
        )
        cir = fitted_r.circle
        if "diameter_mm" in spec:
            from cloudet.core.circle import Circle

            diam = float(spec["diameter_mm"])
            fixed = bool(spec.get("diameter_fixed", True))
            cir = Circle(
                center=cir.center,
                normal=cir.normal,
                diameter_mm=diam,
                diameter_fixed=fixed,
            )
        record = scanned_circle_point_record(
            cir,
            group_id=fitted_r.group_id,
            group_name=fitted_r.group_name,
            circle_index=fitted_r.circle_index,
            quality=fitted_r.quality,
        )
        return "point", np.asarray(cir.center, dtype=np.float64), record

    raise ValueError(
        f"faces.{alias}: unsupported kind={kind!r} (use plane, cylinder, or circle)"
    )


def _require_plane(
    store: dict[str, _Entity],
    key: str,
    *,
    where: str,
    extra_planes: dict[str, Plane] | None = None,
) -> Plane:
    extra_planes = extra_planes or {}
    if is_aligned_plane_id(key):
        plane = extra_planes.get(key)
        if plane is None:
            raise ValueError(f"{where}: {key} needs FRAME axis and origin")
        return plane
    ent = store.get(key)
    if ent is None:
        raise KeyError(f"{where}: unknown id {key!r}")
    if ent.kind != "plane":
        raise TypeError(f"{where}: {key!r} is a {ent.kind}, expected plane")
    return ent.value


def _require_line(
    store: dict[str, _Entity],
    key: str,
    *,
    where: str,
    extra_lines: dict[str, Line] | None = None,
) -> Line:
    extra_lines = extra_lines or {}
    if is_aligned_axis_id(key):
        line = extra_lines.get(key)
        if line is None:
            raise ValueError(f"{where}: {key} needs FRAME axis and origin")
        return line
    ent = store.get(key)
    if ent is None:
        raise KeyError(f"{where}: unknown id {key!r}")
    if ent.kind != "line":
        raise TypeError(f"{where}: {key!r} is a {ent.kind}, expected line")
    return ent.value


def _require_point(
    store: dict[str, _Entity],
    key: str,
    *,
    where: str,
    extra_points: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    extra_points = extra_points or {}
    if is_aligned_origin_id(key):
        pt = extra_points.get(key)
        if pt is None:
            raise ValueError(f"{where}: {key} needs FRAME axis and origin")
        return np.asarray(pt, dtype=np.float64).reshape(3)
    ent = store.get(key)
    if ent is None:
        raise KeyError(f"{where}: unknown id {key!r}")
    if ent.kind != "point":
        raise TypeError(f"{where}: {key!r} is a {ent.kind}, expected point")
    return np.asarray(ent.value, dtype=np.float64).reshape(3)


def _put(store: dict[str, _Entity], entity_id: str, kind: str, value: Any, record: dict) -> None:
    if entity_id in store:
        raise ValueError(f"duplicate id {entity_id!r}")
    store[entity_id] = _Entity(kind=kind, value=value, record=record)


def _run_construct_step(
    store: dict[str, _Entity],
    step: dict,
    extra_lines: dict[str, Line] | None = None,
    extra_planes: dict[str, Plane] | None = None,
    extra_points: dict[str, np.ndarray] | None = None,
    anchors: dict[str, np.ndarray] | None = None,
) -> None:
    if not isinstance(step, dict):
        raise ValueError("construct step must be an object")
    entity_id = step.get("id")
    op = step.get("op")
    if not entity_id or not op:
        raise ValueError("construct step needs id and op")
    where = f"construct[{entity_id}]"
    extra_lines = extra_lines or {}
    extra_planes = extra_planes or {}
    extra_points = extra_points or {}

    def require_line(key: str) -> Line:
        return _require_line(store, key, where=where, extra_lines=extra_lines)

    def require_plane(key: str) -> Plane:
        return _require_plane(store, key, where=where, extra_planes=extra_planes)

    def require_point(key: str) -> np.ndarray:
        return _require_point(store, key, where=where, extra_points=extra_points)

    if op == "offset":
        plane_id = step["plane"]
        distance = float(step["distance_mm"])
        src = require_plane(plane_id)
        plane = offset_plane(src, distance)
        _put(
            store,
            entity_id,
            "plane",
            plane,
            {
                **plane_to_json(plane),
                "provenance": "offset",
                "parents": plane_id,
                "distance_mm": distance,
            },
        )
        return

    if op == "intersect_planes":
        plane_a = step["plane_a"]
        plane_b = step["plane_b"]
        line = intersect_planes(
            require_plane(plane_a),
            require_plane(plane_b),
        )
        _put(
            store,
            entity_id,
            "line",
            line,
            {
                "point": line.point.tolist(),
                "direction": line.direction.tolist(),
                "provenance": "intersection",
                "parents": [plane_a, plane_b],
            },
        )
        return

    if op == "intersect_three_planes":
        keys = [step["plane_a"], step["plane_b"], step["plane_c"]]
        pt = intersect_three_planes(
            *(require_plane(k) for k in keys)
        )
        _put(
            store,
            entity_id,
            "point",
            pt,
            {
                "xyz": np.asarray(pt, dtype=np.float64).tolist(),
                "provenance": "intersection",
                "parents": keys,
            },
        )
        return

    if op == "intersect_line_plane":
        line_id = step["line"]
        plane_id = step["plane"]
        pt = intersect_line_plane(
            require_line(line_id),
            require_plane(plane_id),
        )
        _put(
            store,
            entity_id,
            "point",
            pt,
            {
                "xyz": np.asarray(pt, dtype=np.float64).tolist(),
                "provenance": "intersection",
                "parents": [line_id, plane_id],
            },
        )
        return

    if op == "intersect_normal_plane":
        src_id = step["source_plane"]
        dst_id = step["destination_plane"]
        src_plane = require_plane(src_id)
        through = step.get("through")
        if through is not None:
            through_arr = np.asarray(through, dtype=np.float64)
        elif anchors is not None and src_id in anchors:
            through_arr = project_point_to_plane(anchors[src_id], src_plane)
        else:
            through_arr = None
        pt = intersect_normal_plane(
            src_plane,
            require_plane(dst_id),
            through=through_arr,
        )
        record = {
            "xyz": np.asarray(pt, dtype=np.float64).tolist(),
            "provenance": "intersection",
            "parents": [src_id, dst_id],
            "op": "intersect_normal_plane",
        }
        if through is not None:
            record["through"] = np.asarray(through_arr, dtype=np.float64).tolist()
        _put(store, entity_id, "point", pt, record)
        return

    if op == "line_from_point_normal":
        point_id = step["point"]
        plane_id = step["plane"]
        line = line_from_point_normal(
            require_point(point_id),
            require_plane(plane_id),
        )
        _put(
            store,
            entity_id,
            "line",
            line,
            {
                "point": line.point.tolist(),
                "direction": line.direction.tolist(),
                "provenance": "constructed",
                "parents": [point_id, plane_id],
                "op": "line_from_point_normal",
            },
        )
        return

    if op == "line_from_two_points":
        a = step["point_a"]
        b = step["point_b"]
        if a == b:
            raise ValueError(f"{where}: points a and b must differ")
        line = line_from_two_points(
            require_point(a),
            require_point(b),
        )
        _put(
            store,
            entity_id,
            "line",
            line,
            {
                "point": line.point.tolist(),
                "direction": line.direction.tolist(),
                "provenance": "constructed",
                "parents": [a, b],
                "op": "line_from_two_points",
            },
        )
        return

    if op == "midpoint_line_planes":
        line_id = step["line"]
        a = step["plane_a"]
        b = step["plane_b"]
        if a == b:
            raise ValueError(f"{where}: planes a and b must differ")
        line = require_line(line_id)
        pa = require_plane(a)
        pb = require_plane(b)
        end_a = intersect_line_plane(line, pa)
        end_b = intersect_line_plane(line, pb)
        pt = 0.5 * (end_a + end_b)
        _put(
            store,
            entity_id,
            "point",
            pt,
            {
                "xyz": np.asarray(pt, dtype=np.float64).tolist(),
                "provenance": "constructed",
                "parents": [line_id, a, b],
                "op": "midpoint_line_planes",
                "ends": [
                    np.asarray(end_a, dtype=np.float64).tolist(),
                    np.asarray(end_b, dtype=np.float64).tolist(),
                ],
            },
        )
        return

    if op == "plane_from_plane_point":
        plane_id = step["plane"]
        point_id = step["point"]
        plane = plane_from_plane_point(
            require_plane(plane_id),
            require_point(point_id),
        )
        _put(
            store,
            entity_id,
            "plane",
            plane,
            {
                **plane_to_json(plane),
                "provenance": "constructed",
                "parents": [plane_id, point_id],
                "op": "plane_from_plane_point",
            },
        )
        return

    if op == "plane_from_line_point":
        line_id = step["line"]
        point_id = step["point"]
        plane = plane_from_line_point(
            require_line(line_id),
            require_point(point_id),
        )
        _put(
            store,
            entity_id,
            "plane",
            plane,
            {
                **plane_to_json(plane),
                "provenance": "constructed",
                "parents": [line_id, point_id],
                "op": "plane_from_line_point",
            },
        )
        return

    if op == "plane_from_two_lines":
        a = step["line_a"]
        b = step["line_b"]
        if a == b:
            raise ValueError(f"{where}: lines a and b must differ")
        plane = plane_from_two_lines(
            require_line(a),
            require_line(b),
        )
        _put(
            store,
            entity_id,
            "plane",
            plane,
            {
                **plane_to_json(plane),
                "provenance": "constructed",
                "parents": [a, b],
                "op": "plane_from_two_lines",
            },
        )
        return

    if op == "rotate_plane_about_line":
        plane_id = step["plane"]
        line_id = step["line"]
        angle_deg = float(step["angle_deg"])
        plane = rotate_plane_about_line(
            require_plane(plane_id),
            require_line(line_id),
            angle_deg,
        )
        _put(
            store,
            entity_id,
            "plane",
            plane,
            {
                **plane_to_json(plane),
                "provenance": "constructed",
                "parents": [plane_id, line_id],
                "op": "rotate_plane_about_line",
                "angle_deg": angle_deg,
            },
        )
        return

    if op == "rotate_point_about_line":
        point_id = step["point"]
        line_id = step["line"]
        angle_deg = float(step["angle_deg"])
        pt = rotate_point_about_line(
            require_point(point_id),
            require_line(line_id),
            angle_deg,
        )
        _put(
            store,
            entity_id,
            "point",
            pt,
            {
                "xyz": pt.tolist(),
                "provenance": "constructed",
                "parents": [point_id, line_id],
                "op": "rotate_point_about_line",
                "angle_deg": angle_deg,
            },
        )
        return

    if op == "rotate_line_about_line":
        line_id = step["line"]
        axis_id = step["axis"]
        angle_deg = float(step["angle_deg"])
        ln = rotate_line_about_line(
            require_line(line_id),
            require_line(axis_id),
            angle_deg,
        )
        _put(
            store,
            entity_id,
            "line",
            ln,
            {
                "point": ln.point.tolist(),
                "direction": ln.direction.tolist(),
                "provenance": "constructed",
                "parents": [line_id, axis_id],
                "op": "rotate_line_about_line",
                "angle_deg": angle_deg,
            },
        )
        return

    raise ValueError(f"{where}: unknown op {op!r}")


def _anchor_from_project(project_dir: Path, group_id: int | None) -> np.ndarray | None:
    if group_id is None:
        return None
    doc = load_group_doc(project_dir, int(group_id))
    if not doc:
        return None
    clicked = doc.get("clicked")
    if clicked is None:
        return None
    arr = np.asarray(clicked, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        return None
    return arr


def _check_recipe(recipe: dict) -> None:
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be an object")
    if "faces" not in recipe and "planes" in recipe:
        raise ValueError("this looks like geometry.json; load a recipe.json instead")
    if int(recipe.get("version", 1)) > RECIPE_VERSION:
        raise ValueError(f"unsupported recipe version {recipe.get('version')}")
    migrated = migrate_recipe(recipe)
    recipe.clear()
    recipe.update(migrated)
    if recipe.get("units", "mm") != "mm":
        raise ValueError(f"recipe units must be mm, got {recipe.get('units')!r}")
    faces = recipe.get("faces") or {}
    if not isinstance(faces, dict) or not faces:
        raise ValueError("recipe.faces must be a non-empty object")
    construct = recipe.get("construct") or []
    if not isinstance(construct, list):
        raise ValueError("recipe.construct must be a list")
    seen_ids: set[str] = set()
    for i, step in enumerate(construct):
        parsed = _parse_construct_step(step, where=f"recipe.construct[{i}]")
        entity_id = str(parsed["id"])
        if entity_id in seen_ids:
            raise ValueError(f"recipe.construct duplicate id {entity_id!r}")
        seen_ids.add(entity_id)
    export_ids = recipe.get("export")
    if export_ids is not None and not isinstance(export_ids, list):
        raise ValueError("recipe.export must be a list of ids")
    if "frame" in recipe and recipe["frame"] is not None:
        _parse_recipe_frame(recipe["frame"])
    if "measures" in recipe and recipe["measures"] is not None:
        if not isinstance(recipe["measures"], list):
            raise ValueError("recipe.measures must be a list")
        seen: set[str] = set()
        for spec in recipe["measures"]:
            parsed = _parse_measure_spec(spec)
            mid = parsed["id"]
            if mid in seen:
                raise ValueError(f"recipe.measures duplicate id {mid!r}")
            seen.add(mid)


def build_frame_spec(
    *,
    axis: str,
    origin: str,
    flip_z: bool = False,
    yaw_to: str | None = None,
    yaw_kind: str | None = None,
    yaw_ref: str | None = None,
) -> dict:
    """Build a recipe-safe frame object with exclusive ``yaw_line`` / ``yaw_plane``."""
    out = {
        "axis": str(axis),
        "origin": str(origin),
        "flip_z": bool(flip_z),
    }
    if yaw_to and yaw_ref:
        key = str(yaw_to)
        if key not in ("x", "-x", "y", "-y"):
            raise ValueError(
                f"frame.yaw_to must be one of ['x', '-x', 'y', '-y'], got {key!r}"
            )
        kind = str(yaw_kind or "line")
        if kind == "plane":
            out["yaw_plane"] = str(yaw_ref)
        elif kind == "line":
            out["yaw_line"] = str(yaw_ref)
        else:
            raise ValueError(f"frame yaw_kind must be 'line' or 'plane', got {kind!r}")
        out["yaw_to"] = key
    elif yaw_to:
        raise ValueError("frame.yaw_to needs yaw_line or yaw_plane")
    elif yaw_ref:
        raise ValueError("frame yaw_line/yaw_plane needs yaw_to")
    return out


def normalize_frame_spec(spec: dict) -> dict:
    """Return a cleaned frame object suitable for recipe export."""
    yaw_plane = spec.get("yaw_plane")
    yaw_line = spec.get("yaw_line")
    if yaw_plane and yaw_line:
        raise ValueError("frame: use yaw_line or yaw_plane, not both")
    yaw_kind = "plane" if yaw_plane else "line" if yaw_line else None
    yaw_ref = yaw_plane or yaw_line
    return build_frame_spec(
        axis=str(spec["axis"]),
        origin=str(spec["origin"]),
        flip_z=bool(spec.get("flip_z", False)),
        yaw_to=spec.get("yaw_to"),
        yaw_kind=yaw_kind,
        yaw_ref=str(yaw_ref) if yaw_ref else None,
    )


def _parse_recipe_frame(spec) -> dict | None:
    """Optional Align Z metadata: ``{axis, origin, flip_z, yaw_*, yaw_to}``.

    Not a construct step. Set ``yaw_line`` or ``yaw_plane`` (not both) with
    ``yaw_to`` to fix rotation about Z after the axis is mapped to +Z.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("recipe.frame must be an object")
    axis = spec.get("axis")
    origin = spec.get("origin")
    if not axis or not origin:
        raise ValueError("recipe.frame needs axis and origin ids")
    yaw_line = spec.get("yaw_line")
    yaw_plane = spec.get("yaw_plane")
    if yaw_line and yaw_plane:
        raise ValueError("recipe.frame: use yaw_line or yaw_plane, not both")
    yaw_kind = "plane" if yaw_plane else "line" if yaw_line else None
    yaw_ref = yaw_plane or yaw_line
    return build_frame_spec(
        axis=str(axis),
        origin=str(origin),
        flip_z=bool(spec.get("flip_z", False)),
        yaw_to=spec.get("yaw_to"),
        yaw_kind=yaw_kind,
        yaw_ref=str(yaw_ref) if yaw_ref else None,
    )


def _construct_step_schema(op: str) -> tuple[str, tuple[tuple[str, str], ...], tuple[str, ...], bool]:
    if op in REDUCTION_OP_BY_RECIPE:
        defn = REDUCTION_OP_BY_RECIPE[op]
        operands = tuple((field.step_key, field.kind) for field in defn.operands)
        scalars = tuple(field.step_key for field in defn.scalars)
        return defn.result_kind, operands, scalars, defn.operands_must_differ
    raise ValueError(f"unknown op {op!r}")


def _parse_construct_step(step, *, where: str) -> dict:
    """Validate construct-step shape without checking entity availability.

    Mutates ``step`` in place to v2 operand keys when legacy keys are present.
    """
    if not isinstance(step, dict):
        raise ValueError(f"{where}: expected object")
    migrated = migrate_construct_step(step)
    if migrated is not step:
        step.clear()
        step.update(migrated)
    entity_id = step.get("id")
    op = step.get("op")
    if not entity_id:
        raise ValueError(f"{where}: needs id")
    entity_id = str(entity_id)
    if is_aligned_id(entity_id):
        raise ValueError(f"{where}: id {entity_id!r} is reserved")
    if not op:
        raise ValueError(f"{where}: needs op")
    op = str(op)
    try:
        _result_kind, operands, scalars, must_differ = _construct_step_schema(op)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    resolved: list[str] = []
    for key, kind in operands:
        val = step.get(key)
        if not val:
            raise ValueError(f"{where}: needs {key}")
        if not isinstance(val, str):
            raise ValueError(f"{where}.{key}: expected string id")
        resolved.append(str(val))
    if must_differ and len(set(resolved)) < len(resolved):
        raise ValueError(f"{where}: operands must differ")
    for key in scalars:
        if key not in step:
            raise ValueError(f"{where}: needs {key}")
        try:
            float(step[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}.{key}: expected number") from exc
    if op == "intersect_normal_plane" and "through" in step:
        through = step["through"]
        if not isinstance(through, (list, tuple)) or len(through) != 3:
            raise ValueError(f"{where}.through: expected [x, y, z]")
        try:
            [float(x) for x in through]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}.through: expected numeric [x, y, z]") from exc
    return dict(step)


def _validate_construct_step_refs(
    session: "ReductionSession",
    step: dict,
    allowed: set[str],
    *,
    where: str,
) -> None:
    """Ensure construct operands exist with the expected kinds."""
    parsed = _parse_construct_step(step, where=where)
    op = str(parsed["op"])
    _result_kind, operands, _scalars, _must_differ = _construct_step_schema(op)
    for key, kind in operands:
        eid = str(parsed[key])
        if is_aligned_id(eid):
            if ALIGNED_KIND[eid] != kind:
                raise ValueError(f"{where}.{key} {eid!r} must be a {kind}")
            if eid not in session.available_aligned_ids(kind=kind, before=allowed):
                raise KeyError(
                    f"{where}.{key}: aligned {eid!r} not available yet"
                )
            continue
        if eid not in allowed:
            raise KeyError(f"{where}.{key}: id {eid!r} not available yet")
        try:
            got = session.kind_of(eid)
        except KeyError as exc:
            raise KeyError(f"{where}.{key}: unknown id {eid!r}") from exc
        if got != kind:
            raise ValueError(f"{where}.{key} {eid!r} must be a {kind}")


def _frame_yaw_direction(session: "ReductionSession", spec: dict):
    """Line direction or plane normal used to fix yaw, or ``None``."""
    yaw_plane = spec.get("yaw_plane")
    if yaw_plane:
        return session.plane(yaw_plane).normal, "plane", str(yaw_plane)
    yaw_line = spec.get("yaw_line")
    if yaw_line:
        return session.line(yaw_line).direction, "line", str(yaw_line)
    return None, "", ""


def _frame_ref_ids(spec: dict | None) -> set[str]:
    if not spec:
        return set()
    return {
        str(x)
        for x in (
            spec.get("axis"),
            spec.get("origin"),
            spec.get("yaw_line"),
            spec.get("yaw_plane"),
        )
        if x
    }


def _validate_frame_spec(session: "ReductionSession", spec: dict) -> None:
    axis = spec["axis"]
    origin = spec["origin"]
    if is_aligned_id(axis):
        raise ValueError("recipe.frame.axis cannot be an aligned entity")
    if is_aligned_id(origin):
        raise ValueError("recipe.frame.origin cannot be an aligned entity")
    if axis not in session._store:
        raise KeyError(f"recipe.frame.axis: unknown id {axis!r}")
    if origin not in session._store:
        raise KeyError(f"recipe.frame.origin: unknown id {origin!r}")
    if session.kind_of(axis) != "line":
        raise ValueError(f"recipe.frame.axis {axis!r} must be a line")
    if session.kind_of(origin) != "point":
        raise ValueError(f"recipe.frame.origin {origin!r} must be a point")
    yaw_plane = spec.get("yaw_plane")
    if yaw_plane:
        if is_aligned_id(yaw_plane):
            raise ValueError("recipe.frame.yaw_plane cannot be an aligned entity")
        if yaw_plane not in session._store:
            raise KeyError(f"recipe.frame.yaw_plane: unknown id {yaw_plane!r}")
        if session.kind_of(yaw_plane) != "plane":
            raise ValueError(f"recipe.frame.yaw_plane {yaw_plane!r} must be a plane")
        return
    yaw_line = spec.get("yaw_line")
    if yaw_line:
        if is_aligned_id(yaw_line):
            raise ValueError("recipe.frame.yaw_line cannot be an aligned entity")
        if yaw_line not in session._store:
            raise KeyError(f"recipe.frame.yaw_line: unknown id {yaw_line!r}")
        if session.kind_of(yaw_line) != "line":
            raise ValueError(f"recipe.frame.yaw_line {yaw_line!r} must be a line")


_MEASURE_OPS = MEASURE_OPERAND_FIELDS


def _parse_measure_spec(spec) -> dict:
    """Optional measurement: distances / angles. Not a construct step."""
    if not isinstance(spec, dict):
        raise ValueError("recipe.measures[] must be an object")
    migrated = migrate_measure_spec(spec)
    if migrated is not spec:
        spec.clear()
        spec.update(migrated)
    op = spec.get("op")
    if op not in _MEASURE_OPS:
        raise ValueError(f"recipe.measures unknown op {op!r}")
    mid = str(spec.get("id") or "").strip()
    if not mid:
        raise ValueError("recipe.measures[] needs id")
    if any(ch.isspace() for ch in mid):
        raise ValueError("measure id must not contain whitespace")
    out = {"id": mid, "op": str(op)}
    fields = []
    for key, _kind in _MEASURE_OPS[op]:
        val = spec.get(key)
        if not val:
            raise ValueError(f"recipe.measures.{mid} needs {key}")
        out[key] = str(val)
        fields.append(str(val))
    if len(set(fields)) < len(fields):
        raise ValueError(f"recipe.measures.{mid}: operands must differ")
    return out


def _validate_measure_spec(session: "ReductionSession", spec: dict) -> None:
    mid = spec["id"]
    op = spec["op"]
    for key, kind in _MEASURE_OPS[op]:
        eid = spec[key]
        try:
            got = session.kind_of(eid)
        except KeyError as exc:
            raise KeyError(f"recipe.measures.{mid}.{key}: unknown id {eid!r}") from exc
        if got != kind:
            raise ValueError(
                f"recipe.measures.{mid}.{key} {eid!r} must be a {kind}"
            )


def _measure_operand_ids(spec: dict) -> set[str]:
    op = spec.get("op")
    if op not in _MEASURE_OPS:
        return set()
    return {spec[key] for key, _kind in _MEASURE_OPS[op] if spec.get(key)}


def run_reduction(project_dir: str | Path, recipe: dict) -> ReductionResult:
    """Execute a reduction recipe against a saved project.

    Survey coordinates stay at the top level. If the recipe has ``frame``,
    an ``aligned`` copy is attached (same rule as GUI export with the
    aligned-frame checkbox on).
    """
    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {project_dir}")
    sess = ReductionSession()
    sess.apply_recipe(recipe, project_dir=project_dir)
    export_ids = recipe.get("export")
    return export_reduction_result(
        sess,
        source_project=str(project_dir.resolve()),
        export=None if export_ids is None else [str(x) for x in export_ids],
        aligned_frame=sess.rigid_frame(),
    )


def export_reduction_result(
    session: ReductionSession,
    *,
    source_project: str = "",
    export: list[str] | None = None,
    aligned_frame=None,
) -> ReductionResult:
    """Build ``ReductionResult`` for CLI/GUI export with optional Align Z copy."""
    result = session.to_result(source_project=source_project, export=export)
    if aligned_frame is not None:
        from cloudet.reduction.frame import with_aligned_copy

        result = with_aligned_copy(result, aligned_frame)
    return result


@dataclass(frozen=True)
class ConstructPreview:
    """Dry-run result of one construct step (for GUI preview / tests)."""

    entity_id: str
    kind: str
    plane: Plane | None = None
    line: Line | None = None
    point: np.ndarray | None = None
    anchor: np.ndarray | None = None
    overlay_mm: float = 200.0
    overlay_width_mm: float = 1.0
    segment_ends: tuple[np.ndarray, np.ndarray] | None = None


def preview_construct_step(session: ReductionSession, step: dict) -> ConstructPreview:
    """Execute ``step`` on a session copy; return geometry without mutating ``session``."""
    step = dict(step)
    entity_id = str(step.get("id") or "")
    if not entity_id:
        raise ValueError("construct step needs id")
    trial = copy.deepcopy(session)
    if entity_id in trial._store:
        raise ValueError(f"preview id {entity_id!r} already exists")
    trial.apply_step(step)
    kind = trial.kind_of(entity_id)
    anchor = trial.anchors.get(entity_id)
    anchor_arr = None if anchor is None else np.asarray(anchor, dtype=np.float64)
    if kind == "plane":
        return ConstructPreview(
            entity_id=entity_id,
            kind=kind,
            plane=trial.plane(entity_id),
            anchor=anchor_arr,
            overlay_mm=trial.overlay_mm(entity_id),
        )
    if kind == "line":
        return ConstructPreview(
            entity_id=entity_id,
            kind=kind,
            line=trial.line(entity_id),
            anchor=anchor_arr,
            overlay_mm=trial.overlay_mm(entity_id),
            overlay_width_mm=trial.overlay_width_mm(entity_id),
        )
    if kind == "point":
        rec = trial.record_of(entity_id)
        ends = rec.get("ends")
        segment_ends = None
        if isinstance(ends, list) and len(ends) == 2:
            segment_ends = (
                np.asarray(ends[0], dtype=np.float64).reshape(3),
                np.asarray(ends[1], dtype=np.float64).reshape(3),
            )
        return ConstructPreview(
            entity_id=entity_id,
            kind=kind,
            point=trial.point(entity_id),
            anchor=anchor_arr if anchor_arr is not None else trial.point(entity_id),
            overlay_mm=trial.overlay_mm(entity_id),
            overlay_width_mm=trial.overlay_width_mm(entity_id),
            segment_ends=segment_ends,
        )
    raise ValueError(f"unknown preview kind {kind!r}")


def write_geometry_json(path: str | Path, result: ReductionResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
        f.write("\n")
    return path


def geometry_summary_path(geometry_path: str | Path) -> Path:
    """Sibling path for the slim summary next to a full geometry.json."""
    path = Path(geometry_path)
    if path.name == "geometry.json":
        return path.with_name("geometry_summary.json")
    return path.with_name(f"{path.stem}_summary.json")


def geometry_summary_dict(result: ReductionResult) -> dict:
    """Slim view: entity names + coordinates (aligned frame when available).

    Includes only ``result.exported`` ids when that list is non-empty; otherwise
    all entities. Drops recipe echo, provenance, parents, and quality metadata.
    """
    use_aligned = isinstance(result.aligned, dict)
    if use_aligned:
        planes_src = result.aligned.get("planes") or {}
        lines_src = result.aligned.get("lines") or {}
        points_src = result.aligned.get("points") or {}
        frame_label = "aligned"
    else:
        planes_src = result.planes
        lines_src = result.lines
        points_src = result.points
        frame_label = "survey"

    wanted = [str(x) for x in (result.exported or [])]
    if wanted:
        wanted_set = set(wanted)

        def _select(mapping: dict) -> dict:
            return {k: mapping[k] for k in wanted if k in mapping}
    else:

        def _select(mapping: dict) -> dict:
            return dict(mapping)

    planes_out: dict[str, dict] = {}
    for eid, rec in _select(planes_src).items():
        if not isinstance(rec, dict):
            continue
        if "normal" in rec and "d" in rec:
            planes_out[eid] = {
                "normal": list(rec["normal"]),
                "d": float(rec["d"]),
            }
        elif "abcd" in rec:
            plane = Plane.from_array(rec["abcd"])
            planes_out[eid] = {
                "normal": plane.normal.tolist(),
                "d": float(plane.d),
            }

    lines_out: dict[str, dict] = {}
    for eid, rec in _select(lines_src).items():
        if not isinstance(rec, dict):
            continue
        if "point" in rec and "direction" in rec:
            lines_out[eid] = {
                "point": list(rec["point"]),
                "direction": list(rec["direction"]),
            }

    points_out: dict[str, dict] = {}
    for eid, rec in _select(points_src).items():
        if not isinstance(rec, dict):
            continue
        if "xyz" in rec:
            points_out[eid] = {"xyz": list(rec["xyz"])}

    out: dict[str, Any] = {
        "units": "mm",
        "frame": frame_label,
        "planes": planes_out,
        "lines": lines_out,
        "points": points_out,
    }
    return out


def write_geometry_summary_json(
    path: str | Path, result: ReductionResult
) -> Path:
    """Write ``geometry_summary.json`` (or a custom path) for human-facing use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geometry_summary_dict(result), f, indent=2)
        f.write("\n")
    return path


def write_recipe_json(path: str | Path, recipe: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)
        f.write("\n")
    return path


_REF_KEYS = ( "id", "plane", "plane_a", "plane_b", "plane_c", "line", "line_a", "line_b", "point", "point_a", "point_b", "axis", "source_plane", "destination_plane", "parents",)


def _rewrite_id_refs(obj: dict, old_id: str, new_id: str) -> None:
    """Replace ``old_id`` with ``new_id`` in construct/record id fields."""
    for key in _REF_KEYS:
        val = obj.get(key)
        if val == old_id:
            obj[key] = new_id
        elif isinstance(val, list):
            obj[key] = [new_id if x == old_id else x for x in val]


def _step_operand_ids(step: dict) -> set[str]:
    """Ids this construct step reads (not its own result id)."""
    out: set[str] = set()
    for key in _REF_KEYS:
        if key == "id":
            continue
        val = step.get(key)
        if isinstance(val, str) and val:
            out.add(val)
        elif isinstance(val, list):
            out.update(str(x) for x in val if x)
    return out


@dataclass
class ReductionSession:
    """Interactive reduction state: scanned faces + construct history.

    GUI and tests drive this without touching the filesystem. Export via
    ``to_recipe()`` / ``to_result()`` matches the CLI ``cloudet reduce``
    contract.
    """

    _store: dict[str, _Entity] = field(default_factory=dict)
    _face_specs: dict[str, dict] = field(default_factory=dict)
    _construct: list[dict] = field(default_factory=list)
    visible: dict[str, bool] = field(default_factory=dict)
    # Optional draw hints: entity id → world point used to centre overlays.
    anchors: dict[str, np.ndarray] = field(default_factory=dict)
    # Overlay size (mm): plane patch side, line half-length, point radius.
    display_mm: dict[str, float] = field(default_factory=dict)
    # Line tube diameter overrides (mm). Length stays in display_mm.
    display_width_mm: dict[str, float] = field(default_factory=dict)
    display_default_mm: dict[str, float] = field(
        default_factory=lambda: {
            "plane": 200.0,
            "line": 300.0,
            "point": 4.0,
            "line_diameter": 1.0,
        }
    )
    # Optional Align Z pick: {axis, origin, flip_z, yaw_line|yaw_plane, yaw_to?}.
    frame_spec: dict | None = None
    # Pinned measurements: [{id, op, ...operands}]. Not construct entities.
    measures: list[dict] = field(default_factory=list)
    # Warnings from the latest construct replay (invalid frame/measures dropped).
    replay_warnings: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self._store.clear()
        self._face_specs.clear()
        self._construct.clear()
        self.visible.clear()
        self.anchors.clear()
        self.display_mm.clear()
        self.display_width_mm.clear()
        self.display_default_mm = {
            "plane": 200.0,
            "line": 300.0,
            "point": 4.0,
            "line_diameter": 1.0,
        }
        self.frame_spec = None
        self.measures = []
        self.replay_warnings = []

    def ids(self, *, kind: str | None = None) -> list[str]:
        if kind is None:
            return list(self._store.keys())
        return [k for k, e in self._store.items() if e.kind == kind]

    def kind_of(self, entity_id: str) -> str:
        if is_aligned_id(entity_id):
            if entity_id not in self.available_aligned_ids():
                raise KeyError(f"unknown id {entity_id!r}")
            return ALIGNED_KIND[str(entity_id)]
        return self._store[entity_id].kind

    def record_of(self, entity_id: str) -> dict:
        return self._store[entity_id].record

    def plane(self, entity_id: str) -> Plane:
        if is_aligned_plane_id(entity_id):
            extra = self.aligned_planes()
            if entity_id not in extra:
                raise KeyError(f"{entity_id!r} needs FRAME axis and origin")
            return extra[entity_id]
        ent = self._store[entity_id]
        if ent.kind != "plane":
            raise TypeError(f"{entity_id!r} is a {ent.kind}, expected plane")
        return ent.value

    def line(self, entity_id: str) -> Line:
        if is_aligned_axis_id(entity_id):
            extra = self.aligned_axis_lines()
            if entity_id not in extra:
                raise KeyError(f"{entity_id!r} needs FRAME axis and origin")
            return extra[entity_id]
        ent = self._store[entity_id]
        if ent.kind != "line":
            raise TypeError(f"{entity_id!r} is a {ent.kind}, expected line")
        return ent.value

    def point(self, entity_id: str) -> np.ndarray:
        if is_aligned_origin_id(entity_id):
            extra = self.aligned_origin_points()
            if entity_id not in extra:
                raise KeyError(f"{entity_id!r} needs FRAME axis and origin")
            return extra[entity_id]
        ent = self._store[entity_id]
        if ent.kind != "point":
            raise TypeError(f"{entity_id!r} is a {ent.kind}, expected point")
        return np.asarray(ent.value, dtype=np.float64)

    def bind_scanned(
        self,
        alias: str,
        plane: Plane,
        *,
        group_name: str,
        group_id: int,
        plane_index: int = 0,
        quality: dict | None = None,
        anchor: np.ndarray | None = None,
    ) -> None:
        """Register a fitted face under ``alias`` (overwrites if same alias)."""
        alias = str(alias)
        if alias in self._store and self._store[alias].kind != "plane":
            raise ValueError(f"{alias!r} exists as non-plane")
        # Drop previous face with this alias from face specs / construct refs
        # is caller's responsibility; simple overwrite of scanned binding.
        if alias in self._store:
            del self._store[alias]
        quality = dict(quality or {})
        record = scanned_plane_record(
            plane,
            group_name=str(group_name),
            group_id=int(group_id),
            plane_index=int(plane_index),
            quality=quality,
        )
        self._store[alias] = _Entity(kind="plane", value=plane, record=record)
        self._face_specs[alias] = {
            "from": "group",
            "name": str(group_name),
            "kind": "plane",
            "plane_index": int(plane_index),
        }
        self.visible[alias] = True
        if anchor is not None:
            self.anchors[alias] = np.asarray(anchor, dtype=np.float64).reshape(3)
        if self._construct:
            self._replay_construct()

    def bind_scanned_line(
        self,
        alias: str,
        line: Line,
        *,
        group_name: str,
        group_id: int,
        cylinder_index: int = 0,
        diameter_mm: float,
        diameter_fixed: bool = False,
        quality: dict | None = None,
        anchor: np.ndarray | None = None,
        face_spec: dict | None = None,
    ) -> None:
        """Register a scanned cylinder axis as a line entity."""
        from cloudet.core.cylinder import Cylinder

        alias = str(alias)
        if alias in self._store and self._store[alias].kind != "line":
            raise ValueError(f"{alias!r} exists as non-line")
        if alias in self._store:
            del self._store[alias]
        cyl = Cylinder(
            point=line.point,
            direction=line.direction,
            diameter_mm=float(diameter_mm),
            diameter_fixed=bool(diameter_fixed),
        )
        record = scanned_cylinder_line_record(
            cyl,
            group_name=str(group_name),
            group_id=int(group_id),
            cylinder_index=int(cylinder_index),
            quality=dict(quality or {}),
        )
        self._store[alias] = _Entity(kind="line", value=line, record=record)
        spec = {
            "from": "group",
            "name": str(group_name),
            "kind": "cylinder",
            "cylinder_index": int(cylinder_index),
            "diameter_mm": float(diameter_mm),
            "diameter_fixed": bool(diameter_fixed),
        }
        if face_spec:
            spec.update(face_spec)
        self._face_specs[alias] = spec
        self.visible[alias] = True
        if anchor is not None:
            self.anchors[alias] = np.asarray(anchor, dtype=np.float64).reshape(3)
        else:
            self.anchors[alias] = np.asarray(line.point, dtype=np.float64).reshape(3)
        if self._construct:
            self._replay_construct()

    def bind_scanned_point(
        self,
        alias: str,
        point: np.ndarray,
        *,
        group_name: str,
        group_id: int,
        circle_index: int = 0,
        diameter_mm: float,
        diameter_fixed: bool = False,
        normal: np.ndarray | None = None,
        quality: dict | None = None,
        face_spec: dict | None = None,
    ) -> None:
        """Register a scanned circle center as a point entity."""
        from cloudet.core.circle import Circle

        alias = str(alias)
        if alias in self._store and self._store[alias].kind != "point":
            raise ValueError(f"{alias!r} exists as non-point")
        if alias in self._store:
            del self._store[alias]
        xyz = np.asarray(point, dtype=np.float64).reshape(3)
        nrm = (
            np.array([0.0, 0.0, 1.0])
            if normal is None
            else np.asarray(normal, dtype=np.float64).reshape(3)
        )
        cir = Circle(
            center=xyz,
            normal=nrm,
            diameter_mm=float(diameter_mm),
            diameter_fixed=bool(diameter_fixed),
        )
        record = scanned_circle_point_record(
            cir,
            group_name=str(group_name),
            group_id=int(group_id),
            circle_index=int(circle_index),
            quality=dict(quality or {}),
        )
        self._store[alias] = _Entity(kind="point", value=xyz, record=record)
        spec = {
            "from": "group",
            "name": str(group_name),
            "kind": "circle",
            "circle_index": int(circle_index),
            "diameter_mm": float(diameter_mm),
            "diameter_fixed": bool(diameter_fixed),
        }
        if face_spec:
            spec.update(face_spec)
        self._face_specs[alias] = spec
        self.visible[alias] = True
        self.anchors[alias] = xyz.copy()
        if self._construct:
            self._replay_construct()

    def apply_step(self, step: dict) -> str:
        """Append and execute one construct step. Returns the new entity id."""
        step = dict(step)
        entity_id = step.get("id")
        if not entity_id:
            raise ValueError("construct step needs id")
        entity_id = str(entity_id)
        if is_aligned_id(entity_id):
            raise ValueError(f"id {entity_id!r} is reserved")
        if entity_id in self._store:
            raise ValueError(f"duplicate id {entity_id!r}")
        _parse_construct_step(step, where=f"construct.{entity_id}")
        _run_construct_step(
            self._store,
            step,
            extra_lines=self.aligned_axis_lines(),
            extra_planes=self.aligned_planes(),
            extra_points=self.aligned_origin_points(),
            anchors=self.anchors,
        )
        self._construct.append(step)
        self.visible[entity_id] = True
        if step.get("op") == "line_from_two_points":
            pa = self.point(step["point_a"])
            pb = self.point(step["point_b"])
            self.anchors[entity_id] = 0.5 * (pa + pb)
            half = 0.5 * float(np.linalg.norm(pb - pa))
            default = float(self.display_default_mm.get("line", 300.0))
            if half > default:
                self.set_overlay_mm(entity_id, half)
            return entity_id
        kind = self.kind_of(entity_id)
        if kind == "point":
            self.anchors[entity_id] = self.point(entity_id)
            return entity_id
        # Inherit overlay centre from an operand when possible.
        pt_id = step.get("point")
        if pt_id and (pt_id in self._store or is_aligned_origin_id(pt_id)):
            try:
                if self.kind_of(pt_id) == "point":
                    self.anchors[entity_id] = self.point(pt_id)
                    return entity_id
            except (KeyError, TypeError):
                pass
        of = (
            step.get("plane")
            or step.get("plane_a")
            or step.get("source_plane")
            or step.get("line")
            or step.get("point")
            or step.get("point_a")
            or step.get("line_a")
        )
        if of in self.anchors:
            self.anchors[entity_id] = self.anchors[of].copy()
        elif step.get("plane") in self.anchors:
            self.anchors[entity_id] = self.anchors[step["plane"]].copy()
        elif is_aligned_id(str(of or "")) or is_aligned_id(str(step.get("plane") or "")):
            extra = self.aligned_origin_points()
            if ALIGNED_ORIGIN_ID in extra:
                self.anchors[entity_id] = extra[ALIGNED_ORIGIN_ID].copy()
        elif kind == "line":
            self.anchors[entity_id] = self.line(entity_id).point.copy()
        return entity_id

    def construct_step(self, entity_id: str) -> dict | None:
        """Copy of the construct step that produced ``entity_id``, if any."""
        entity_id = str(entity_id)
        for step in self._construct:
            if str(step.get("id")) == entity_id:
                return dict(step)
        return None

    def operand_ids_before(self, entity_id: str) -> set[str]:
        """Ids a construct step may read: scanned faces and earlier results."""
        entity_id = str(entity_id)
        idx = next(
            (i for i, s in enumerate(self._construct) if str(s.get("id")) == entity_id),
            None,
        )
        if idx is None:
            raise KeyError(f"{entity_id!r} is not a construct step")
        allowed = set(self._face_specs)
        for step in self._construct[:idx]:
            sid = str(step.get("id") or "")
            if sid:
                allowed.add(sid)
        return allowed

    def replace_construct_step(self, entity_id: str, step: dict) -> None:
        """Replace one construct step (same id) and replay from scanned faces.

        Operands must be scanned faces or earlier construct results — not this
        step and not anything after it. Overlay sizes, visibility, frame, and
        measures for surviving ids are kept. On failure the session is left
        unchanged.
        """
        entity_id = str(entity_id)
        if entity_id in self._face_specs:
            raise ValueError(f"{entity_id!r} is a scanned face; cannot edit here")
        idx = next(
            (i for i, s in enumerate(self._construct) if str(s.get("id")) == entity_id),
            None,
        )
        if idx is None:
            raise KeyError(f"{entity_id!r} is not a construct step")
        step = migrate_construct_step(dict(step))
        step["id"] = entity_id
        if not step.get("op"):
            raise ValueError("construct step needs op")
        old_op = str(self._construct[idx].get("op") or "")
        new_op = str(step.get("op") or "")
        if old_op != new_op:
            raise ValueError(
                f"cannot change op of {entity_id!r} from {old_op!r} to {new_op!r}"
            )
        allowed = self.operand_ids_before(entity_id)
        allowed.update(self.available_aligned_ids(before=allowed))
        ops = _step_operand_ids(step)
        if entity_id in ops:
            raise ValueError(f"{entity_id!r} cannot reference itself")
        bad = ops - allowed
        if bad:
            raise ValueError(
                "operand not available yet (must be a face or earlier "
                f"construct): {sorted(bad)}"
            )
        trial = copy.deepcopy(self)
        trial._construct[idx] = step
        trial._replay_construct()
        self._adopt(trial)

    def _replay_construct(self) -> list[str]:
        """Drop construct results and re-run ``_construct`` from scanned faces."""
        warnings: list[str] = []
        vis = dict(self.visible)
        display_mm = dict(self.display_mm)
        display_width = dict(self.display_width_mm)
        steps = [dict(s) for s in self._construct]
        face_ids = set(self._face_specs)
        for eid in list(self._store):
            if eid not in face_ids:
                self._store.pop(eid, None)
                self.visible.pop(eid, None)
                self.anchors.pop(eid, None)
                self.display_mm.pop(eid, None)
                self.display_width_mm.pop(eid, None)
        self._construct.clear()
        for step in steps:
            self.apply_step(step)
        for eid, shown in vis.items():
            if eid in self._store or is_aligned_id(eid):
                self.visible[eid] = shown
        self.display_mm = {k: v for k, v in display_mm.items() if k in self._store}
        self.display_width_mm = {
            k: v for k, v in display_width.items() if k in self._store
        }
        if self.frame_spec:
            try:
                _validate_frame_spec(self, self.frame_spec)
            except Exception as exc:
                warnings.append(f"dropped frame: {exc}")
                self.frame_spec = None
        kept: list[dict] = []
        for spec in self.measures:
            mid = str(spec.get("id") or "?")
            try:
                _validate_measure_spec(self, spec)
                kept.append(spec)
            except Exception as exc:
                warnings.append(f"dropped measure {mid!r}: {exc}")
        self.measures = kept
        self.replay_warnings = warnings
        return warnings

    def _adopt(self, other: "ReductionSession") -> None:
        self._store = other._store
        self._face_specs = other._face_specs
        self._construct = other._construct
        self.visible = other.visible
        self.anchors = other.anchors
        self.display_mm = other.display_mm
        self.display_width_mm = other.display_width_mm
        self.display_default_mm = other.display_default_mm
        self.frame_spec = other.frame_spec
        self.measures = other.measures
        self.replay_warnings = list(other.replay_warnings)

    def apply_recipe(
        self,
        recipe: dict,
        *,
        project_dir: str | Path | None = None,
        bind_face=None,
    ) -> None:
        """Replace this session by executing ``recipe``.

        ``bind_face(alias, spec)`` may return ``(plane, record, anchor)`` or
        ``None`` to fall back to ``project_dir`` (saved ``groups/``).
        """
        _check_recipe(recipe)
        if bind_face is None and project_dir is None:
            raise ValueError("apply_recipe needs project_dir or bind_face")
        project_path = None if project_dir is None else Path(project_dir)
        if project_path is not None and not project_path.is_dir():
            raise NotADirectoryError(f"not a directory: {project_path}")

        target = ReductionSession()
        target.frame_spec = _parse_recipe_frame(recipe.get("frame"))
        for alias, spec in (recipe.get("faces") or {}).items():
            alias = str(alias)
            resolved = None if bind_face is None else bind_face(alias, spec)
            if resolved is None:
                if project_path is None:
                    raise ValueError(f"faces.{alias}: cannot resolve (no project_dir)")
                kind, value, record = _bind_face(project_path, alias, spec)
                anchor = _anchor_from_project(project_path, record.get("group_id"))
            else:
                # Custom binders may return plane-style (plane, record, anchor)
                # or tagged (kind, value, record, anchor).
                if len(resolved) == 3:
                    kind, value, record = "plane", resolved[0], resolved[1]
                    anchor = resolved[2]
                else:
                    kind, value, record, anchor = resolved
            if kind == "plane":
                target.bind_scanned(
                    alias,
                    value,
                    group_name=str(record.get("group_name", alias)),
                    group_id=int(record.get("group_id", 0)),
                    plane_index=int(record.get("plane_index", 0)),
                    quality=record.get("quality") or {},
                    anchor=anchor,
                )
            elif kind == "line":
                target.bind_scanned_line(
                    alias,
                    value,
                    group_name=str(record.get("group_name", alias)),
                    group_id=int(record.get("group_id", 0)),
                    cylinder_index=int(record.get("cylinder_index", 0)),
                    diameter_mm=float(record["diameter_mm"]),
                    diameter_fixed=bool(record.get("diameter_fixed", False)),
                    quality=record.get("quality") or {},
                    anchor=anchor,
                    face_spec=dict(spec),
                )
            elif kind == "point":
                target.bind_scanned_point(
                    alias,
                    value,
                    group_name=str(record.get("group_name", alias)),
                    group_id=int(record.get("group_id", 0)),
                    circle_index=int(record.get("circle_index", 0)),
                    diameter_mm=float(record["diameter_mm"]),
                    diameter_fixed=bool(record.get("diameter_fixed", False)),
                    normal=record.get("normal"),
                    quality=record.get("quality") or {},
                    face_spec=dict(spec),
                )
            else:
                raise ValueError(f"faces.{alias}: unexpected bind kind {kind!r}")
            target._face_specs[alias] = dict(spec)
        allowed = set(target._face_specs)
        for i, step in enumerate(recipe.get("construct") or []):
            step = dict(step)
            where = f"recipe.construct[{i}]"
            _validate_construct_step_refs(target, step, allowed, where=where)
            target.apply_step(step)
            allowed.add(str(step["id"]))
        if target.frame_spec is not None:
            _validate_frame_spec(target, target.frame_spec)
        measures = []
        for spec in recipe.get("measures") or []:
            parsed = _parse_measure_spec(spec)
            _validate_measure_spec(target, parsed)
            measures.append(parsed)
        self.clear()
        self._store.update(target._store)
        self._face_specs.update(target._face_specs)
        self._construct.extend(target._construct)
        self.visible.update(target.visible)
        self.anchors.update({k: np.asarray(v, dtype=np.float64).copy() for k, v in target.anchors.items()})
        self.frame_spec = target.frame_spec
        self.measures = measures

    @classmethod
    def from_recipe(
        cls,
        recipe: dict,
        *,
        project_dir: str | Path | None = None,
        bind_face=None,
    ) -> "ReductionSession":
        sess = cls()
        sess.apply_recipe(recipe, project_dir=project_dir, bind_face=bind_face)
        return sess

    def rename(self, old_id: str, new_id: str) -> str:
        """Rename an entity and rewrite every construct/record reference."""
        old_id = str(old_id)
        new_id = str(new_id).strip()
        if old_id not in self._store:
            raise KeyError(f"unknown id {old_id!r}")
        if not new_id:
            raise ValueError("id must not be empty")
        if any(ch.isspace() for ch in new_id):
            raise ValueError("id must not contain whitespace")
        if new_id == old_id:
            return old_id
        if is_aligned_id(new_id):
            raise ValueError(f"id {new_id!r} is reserved")
        if new_id in self._store:
            raise ValueError(f"id {new_id!r} already exists")
        self._store[new_id] = self._store.pop(old_id)
        if old_id in self._face_specs:
            self._face_specs[new_id] = self._face_specs.pop(old_id)
        if old_id in self.visible:
            self.visible[new_id] = self.visible.pop(old_id)
        if old_id in self.anchors:
            self.anchors[new_id] = self.anchors.pop(old_id)
        if old_id in self.display_mm:
            self.display_mm[new_id] = self.display_mm.pop(old_id)
        if old_id in self.display_width_mm:
            self.display_width_mm[new_id] = self.display_width_mm.pop(old_id)
        for step in self._construct:
            _rewrite_id_refs(step, old_id, new_id)
        for ent in self._store.values():
            _rewrite_id_refs(ent.record, old_id, new_id)
        if self.frame_spec:
            if self.frame_spec.get("axis") == old_id:
                self.frame_spec["axis"] = new_id
            if self.frame_spec.get("origin") == old_id:
                self.frame_spec["origin"] = new_id
            if self.frame_spec.get("yaw_line") == old_id:
                self.frame_spec["yaw_line"] = new_id
            if self.frame_spec.get("yaw_plane") == old_id:
                self.frame_spec["yaw_plane"] = new_id
        for spec in self.measures:
            _rewrite_id_refs(spec, old_id, new_id)
        return new_id

    def dependents(self, entity_id: str) -> list[str]:
        """Construct result ids that transitively use ``entity_id``."""
        entity_id = str(entity_id)
        doomed: set[str] = {entity_id}
        if entity_id in _frame_ref_ids(self.frame_spec):
            for step in self._construct:
                sid = str(step.get("id", ""))
                if sid and (_step_operand_ids(step) & set(ALIGNED_ENTITY_IDS)):
                    doomed.add(sid)
        changed = True
        while changed:
            changed = False
            for step in self._construct:
                sid = str(step.get("id", ""))
                if not sid or sid in doomed:
                    continue
                if _step_operand_ids(step) & doomed:
                    doomed.add(sid)
                    changed = True
        return [sid for sid in self.ids() if sid in doomed and sid != entity_id]

    def remove(self, entity_id: str) -> list[str]:
        """Delete ``entity_id`` and anything built from it. Returns removed ids."""
        entity_id = str(entity_id)
        if entity_id not in self._store:
            raise KeyError(f"unknown id {entity_id!r}")
        doomed = {entity_id, *self.dependents(entity_id)}
        removed = [eid for eid in self.ids() if eid in doomed]
        self._construct = [
            s for s in self._construct if str(s.get("id", "")) not in doomed
        ]
        for eid in removed:
            self._store.pop(eid, None)
            self._face_specs.pop(eid, None)
            self.visible.pop(eid, None)
            self.anchors.pop(eid, None)
            self.display_mm.pop(eid, None)
            self.display_width_mm.pop(eid, None)
        if self.frame_spec:
            refs = {self.frame_spec.get("axis"), self.frame_spec.get("origin")}
            aligned_doomed: set[str] = set()
            if doomed & _frame_ref_ids(self.frame_spec):
                aligned_doomed.update(ALIGNED_ENTITY_IDS)
            if doomed & refs:
                self.frame_spec = None
            elif self.frame_spec.get("yaw_line") in doomed:
                self.frame_spec.pop("yaw_line", None)
                self.frame_spec.pop("yaw_to", None)
            elif self.frame_spec.get("yaw_plane") in doomed:
                self.frame_spec.pop("yaw_plane", None)
                self.frame_spec.pop("yaw_to", None)
        else:
            aligned_doomed = set()
        self.measures = [
            m
            for m in self.measures
            if not (_measure_operand_ids(m) & (doomed | aligned_doomed))
        ]
        return removed

    def overlay_mm(self, entity_id: str) -> float:
        """Display size for one entity (override or kind default)."""
        eid = str(entity_id)
        if eid in self.display_mm:
            return float(self.display_mm[eid])
        kind = self.kind_of(eid)
        return float(self.display_default_mm.get(kind, 200.0))

    def set_overlay_mm(self, entity_id: str, size_mm: float) -> None:
        self.display_mm[str(entity_id)] = float(size_mm)

    def clear_overlay_mm(self, entity_id: str) -> None:
        self.display_mm.pop(str(entity_id), None)

    def overlay_width_mm(self, entity_id: str) -> float:
        """Line tube diameter (mm): per-entity override or default."""
        eid = str(entity_id)
        if eid in self.display_width_mm:
            return float(self.display_width_mm[eid])
        return float(self.display_default_mm.get("line_diameter", 1.0))

    def set_overlay_width_mm(self, entity_id: str, diameter_mm: float) -> None:
        self.display_width_mm[str(entity_id)] = float(diameter_mm)

    def clear_overlay_width_mm(self, entity_id: str) -> None:
        self.display_width_mm.pop(str(entity_id), None)

    def offset(self, entity_id: str, plane: str, distance_mm: float) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "offset",
            "plane": plane,
            "distance_mm": float(distance_mm),
        })

    def intersect_planes(self, entity_id: str, plane_a: str, plane_b: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "intersect_planes",
            "plane_a": plane_a,
            "plane_b": plane_b,
        })

    def intersect_line_plane(self, entity_id: str, line: str, plane: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "intersect_line_plane",
            "line": line,
            "plane": plane,
        })

    def intersect_three_planes(
        self, entity_id: str, plane_a: str, plane_b: str, plane_c: str
    ) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "intersect_three_planes",
            "plane_a": plane_a,
            "plane_b": plane_b,
            "plane_c": plane_c,
        })

    def intersect_normal_plane(
        self,
        entity_id: str,
        source_plane: str,
        destination_plane: str,
        through: np.ndarray | None = None,
    ) -> str:
        step: dict[str, Any] = {
            "id": str(entity_id),
            "op": "intersect_normal_plane",
            "source_plane": source_plane,
            "destination_plane": destination_plane,
        }
        if through is not None:
            step["through"] = np.asarray(through, dtype=np.float64).reshape(3).tolist()
        return self.apply_step(step)

    def line_from_point_normal(self, entity_id: str, point: str, plane: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "line_from_point_normal",
            "point": point,
            "plane": plane,
        })

    def line_from_two_points(self, entity_id: str, point_a: str, point_b: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "line_from_two_points",
            "point_a": point_a,
            "point_b": point_b,
        })

    def midpoint_line_planes(
        self, entity_id: str, line: str, plane_a: str, plane_b: str
    ) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "midpoint_line_planes",
            "line": line,
            "plane_a": plane_a,
            "plane_b": plane_b,
        })

    def plane_from_plane_point(self, entity_id: str, plane: str, point: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "plane_from_plane_point",
            "plane": plane,
            "point": point,
        })

    def plane_from_line_point(self, entity_id: str, line: str, point: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "plane_from_line_point",
            "line": line,
            "point": point,
        })

    def plane_from_two_lines(self, entity_id: str, line_a: str, line_b: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "plane_from_two_lines",
            "line_a": line_a,
            "line_b": line_b,
        })

    def rotate_plane_about_line(
        self, entity_id: str, plane: str, line: str, angle_deg: float
    ) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "rotate_plane_about_line",
            "plane": plane,
            "line": line,
            "angle_deg": float(angle_deg),
        })

    def rotate_point_about_line(
        self, entity_id: str, point: str, line: str, angle_deg: float
    ) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "rotate_point_about_line",
            "point": point,
            "line": line,
            "angle_deg": float(angle_deg),
        })

    def rotate_line_about_line(
        self, entity_id: str, line: str, axis: str, angle_deg: float
    ) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "rotate_line_about_line",
            "line": line,
            "axis": axis,
            "angle_deg": float(angle_deg),
        })

    def unique_id(self, prefix: str) -> str:
        n = 1
        while True:
            candidate = f"{prefix}_{n}"
            if candidate not in self._store and not is_aligned_id(candidate):
                return candidate
            n += 1

    def to_recipe(self, export: list[str] | None = None) -> dict:
        if not self._face_specs:
            raise ValueError("no scanned faces bound")
        ids = list(self._store.keys()) if export is None else list(export)
        out = {
            "version": RECIPE_VERSION,
            "units": "mm",
            "faces": dict(self._face_specs),
            "construct": [dict(s) for s in self._construct],
            "export": ids,
        }
        if self.frame_spec:
            out["frame"] = normalize_frame_spec(self.frame_spec)
        if self.measures:
            out["measures"] = [dict(m) for m in self.measures]
        return out

    def unique_measure_id(self, prefix: str) -> str:
        taken = {m["id"] for m in self.measures}
        n = 1
        while f"{prefix}_{n}" in taken:
            n += 1
        return f"{prefix}_{n}"

    def add_measure(self, spec: dict) -> str:
        spec = dict(spec)
        if not str(spec.get("id") or "").strip():
            prefix = {
                "distance_points": "dist",
                "distance_point_plane": "dplane",
                "distance_point_line": "dline",
                "angle_planes": "angp",
                "angle_lines": "angl",
                "angle_line_plane": "anglp",
            }.get(spec.get("op"), "meas")
            spec["id"] = self.unique_measure_id(prefix)
        parsed = _parse_measure_spec(spec)
        if any(m["id"] == parsed["id"] for m in self.measures):
            raise ValueError(f"measure id {parsed['id']!r} already exists")
        _validate_measure_spec(self, parsed)
        self.evaluate_measure(parsed)
        self.measures.append(parsed)
        return parsed["id"]

    def remove_measure(self, measure_id: str) -> None:
        mid = str(measure_id)
        before = len(self.measures)
        self.measures = [m for m in self.measures if m["id"] != mid]
        if len(self.measures) == before:
            raise KeyError(f"unknown measure {mid!r}")

    def evaluate_measure(self, spec: dict) -> dict:
        """Return ``spec`` plus ``value`` and ``unit`` from current geometry."""
        parsed = _parse_measure_spec(spec)
        _validate_measure_spec(self, parsed)
        op = parsed["op"]
        if op == "distance_points":
            value = distance_points(
                self.point(parsed["point_a"]), self.point(parsed["point_b"])
            )
            unit = "mm"
        elif op == "distance_point_plane":
            value = distance_point_plane(
                self.point(parsed["point"]), self.plane(parsed["plane"])
            )
            unit = "mm"
        elif op == "distance_point_line":
            value = distance_point_line(
                self.point(parsed["point"]), self.line(parsed["line"])
            )
            unit = "mm"
        elif op == "angle_planes":
            value = angle_planes_deg(
                self.plane(parsed["plane_a"]), self.plane(parsed["plane_b"])
            )
            unit = "deg"
        elif op == "angle_lines":
            value = angle_lines_deg(
                self.line(parsed["line_a"]), self.line(parsed["line_b"])
            )
            unit = "deg"
        elif op == "angle_line_plane":
            value = angle_line_plane_deg(
                self.line(parsed["line"]), self.plane(parsed["plane"])
            )
            unit = "deg"
        else:
            raise ValueError(f"unknown measure op {op!r}")
        out = dict(parsed)
        out["value"] = float(value)
        out["unit"] = unit
        return out

    def _resolved_aligned_frame(self):
        """RigidFrame for virtual aligned entities, or ``None`` if cyclic/unset."""
        spec = self.frame_spec
        if not spec:
            return None
        if any(
            is_aligned_id(str(spec.get(key) or ""))
            for key in ("axis", "origin", "yaw_line", "yaw_plane")
        ):
            return None
        try:
            frame = self.rigid_frame()
        except (KeyError, TypeError, ValueError):
            return None
        return frame

    def aligned_axis_lines(self) -> dict[str, Line]:
        """Survey-frame lines for ``aligned.x/y/z`` when FRAME can be resolved."""
        frame = self._resolved_aligned_frame()
        if frame is None:
            return {}
        return {eid: aligned_axis_line(frame, eid) for eid in ALIGNED_AXIS_IDS}

    def aligned_origin_points(self) -> dict[str, np.ndarray]:
        """Survey-frame origin for ``aligned.origin`` when FRAME can be resolved."""
        frame = self._resolved_aligned_frame()
        if frame is None:
            return {}
        return {ALIGNED_ORIGIN_ID: aligned_origin_point(frame)}

    def aligned_planes(self) -> dict[str, Plane]:
        """Survey-frame coordinate planes when FRAME can be resolved."""
        frame = self._resolved_aligned_frame()
        if frame is None:
            return {}
        return {eid: aligned_plane(frame, eid) for eid in ALIGNED_PLANE_IDS}

    def available_aligned_ids(
        self, *, kind: str | None = None, before: set[str] | None = None
    ) -> set[str]:
        """Virtual aligned entity ids that may be used as operands."""
        spec = self.frame_spec
        if not spec or self._resolved_aligned_frame() is None:
            return set()
        needed = _frame_ref_ids(spec)
        pool = set(self._store) if before is None else before
        if not needed <= pool:
            return set()
        ids = set(ALIGNED_ENTITY_IDS)
        if kind is not None:
            ids = {eid for eid in ids if ALIGNED_KIND[eid] == kind}
        return ids

    def available_aligned_axis_ids(self, *, before: set[str] | None = None) -> set[str]:
        """Aligned axis ids that may be used as line operands."""
        return self.available_aligned_ids(kind="line", before=before)

    def rigid_frame(self):
        """Align Z pose from ``frame_spec``, or ``None`` if unset."""
        spec = self.frame_spec
        if not spec:
            return None
        from cloudet.reduction.frame import RigidFrame

        line = self.line(spec["axis"])
        origin = self.point(spec["origin"])
        yaw_dir, yaw_kind, yaw_id = _frame_yaw_direction(self, spec)
        return RigidFrame.align_z(
            line.direction,
            origin,
            flip_z=bool(spec.get("flip_z", False)),
            axis_id=spec["axis"],
            origin_id=spec["origin"],
            yaw_direction=yaw_dir,
            yaw_to=spec.get("yaw_to"),
            yaw_id=yaw_id,
            yaw_kind=yaw_kind or "line",
        )

    def to_result(
        self,
        *,
        source_project: str = "",
        export: list[str] | None = None,
    ) -> ReductionResult:
        recipe = self.to_recipe(export=export)
        result = ReductionResult(
            recipe=_recipe_fingerprint(recipe),
            source_project=source_project,
            exported=[str(x) for x in recipe["export"]],
        )
        unknown = set(result.exported) - set(self._store)
        if unknown:
            raise KeyError(f"recipe.export references unknown ids: {sorted(unknown)}")
        for eid, ent in self._store.items():
            if ent.kind == "plane":
                result.planes[eid] = ent.record
            elif ent.kind == "line":
                result.lines[eid] = ent.record
            elif ent.kind == "point":
                result.points[eid] = ent.record
        if self.measures:
            result.measures = [self.evaluate_measure(m) for m in self.measures]
        return result
