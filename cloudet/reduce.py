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

from cloudet.geometry import (
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
    offset_plane,
)
from cloudet.plane import Plane
from cloudet.project import FittedPlane, load_fitted_plane, load_group_doc

__all__ = [
    "ReductionResult",
    "ReductionSession",
    "load_recipe",
    "run_reduction",
    "write_geometry_json",
    "write_recipe_json",
]



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
            "version": 1,
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
    if int(doc.get("version", 1)) != 1:
        raise ValueError(f"unsupported recipe version {doc.get('version')}")
    if doc.get("units", "mm") != "mm":
        raise ValueError(f"recipe units must be mm, got {doc.get('units')!r}")
    return doc


def _recipe_fingerprint(recipe: dict) -> dict:
    raw = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "echo": recipe}


def _bind_face(project_dir: Path, alias: str, spec: dict) -> tuple[Plane, dict]:
    if not isinstance(spec, dict):
        raise ValueError(f"faces.{alias}: expected object, got {type(spec).__name__}")
    src = spec.get("from", "group")
    if src != "group":
        raise ValueError(f"faces.{alias}: unsupported from={src!r}")

    name = spec.get("name")
    group_id = spec.get("group_id")
    plane_index = int(spec.get("plane_index", 0))
    if name is not None and group_id is not None:
        raise ValueError(f"faces.{alias}: provide name or group_id, not both")
    if name is None and group_id is None:
        raise ValueError(f"faces.{alias}: need name or group_id")

    fitted: FittedPlane = load_fitted_plane(
        project_dir,
        name=None if name is None else str(name),
        group_id=None if group_id is None else int(group_id),
        plane_index=plane_index,
    )
    record = {
        "abcd": fitted.plane.as_array().tolist(),
        "provenance": "scanned",
        "group_id": fitted.group_id,
        "group_name": fitted.group_name,
        "plane_index": fitted.plane_index,
        "quality": {
            k: fitted.quality[k]
            for k in ("status", "mad_sigma_mm", "threshold_mm", "n_points", "bimodal", "reasons")
            if fitted.quality.get(k) is not None
        },
    }
    return fitted.plane, record


def _require_plane(store: dict[str, _Entity], key: str, *, where: str) -> Plane:
    ent = store.get(key)
    if ent is None:
        raise KeyError(f"{where}: unknown id {key!r}")
    if ent.kind != "plane":
        raise TypeError(f"{where}: {key!r} is a {ent.kind}, expected plane")
    return ent.value


def _require_line(store: dict[str, _Entity], key: str, *, where: str) -> Line:
    ent = store.get(key)
    if ent is None:
        raise KeyError(f"{where}: unknown id {key!r}")
    if ent.kind != "line":
        raise TypeError(f"{where}: {key!r} is a {ent.kind}, expected line")
    return ent.value


def _require_point(store: dict[str, _Entity], key: str, *, where: str) -> np.ndarray:
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


def _run_construct_step(store: dict[str, _Entity], step: dict) -> None:
    if not isinstance(step, dict):
        raise ValueError("construct step must be an object")
    entity_id = step.get("id")
    op = step.get("op")
    if not entity_id or not op:
        raise ValueError("construct step needs id and op")
    where = f"construct[{entity_id}]"

    if op == "offset":
        of = step["of"]
        distance = float(step["distance_mm"])
        src = _require_plane(store, of, where=where)
        plane = offset_plane(src, distance)
        _put(
            store,
            entity_id,
            "plane",
            plane,
            {
                "abcd": plane.as_array().tolist(),
                "provenance": "offset",
                "of": of,
                "distance_mm": distance,
            },
        )
        return

    if op == "intersect_planes":
        a = step["a"]
        b = step["b"]
        line = intersect_planes(
            _require_plane(store, a, where=where),
            _require_plane(store, b, where=where),
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
                "of": [a, b],
            },
        )
        return

    if op == "intersect_three_planes":
        keys = [step["a"], step["b"], step["c"]]
        pt = intersect_three_planes(
            *(_require_plane(store, k, where=where) for k in keys)
        )
        _put(
            store,
            entity_id,
            "point",
            pt,
            {
                "xyz": np.asarray(pt, dtype=np.float64).tolist(),
                "provenance": "intersection",
                "of": keys,
            },
        )
        return

    if op == "intersect_line_plane":
        line_id = step["line"]
        plane_id = step["plane"]
        pt = intersect_line_plane(
            _require_line(store, line_id, where=where),
            _require_plane(store, plane_id, where=where),
        )
        _put(
            store,
            entity_id,
            "point",
            pt,
            {
                "xyz": np.asarray(pt, dtype=np.float64).tolist(),
                "provenance": "intersection",
                "of": [line_id, plane_id],
            },
        )
        return

    if op == "intersect_normal_plane":
        src_id = step["src"]
        dst_id = step["dst"]
        through = step.get("through")
        through_arr = None if through is None else np.asarray(through, dtype=np.float64)
        pt = intersect_normal_plane(
            _require_plane(store, src_id, where=where),
            _require_plane(store, dst_id, where=where),
            through=through_arr,
        )
        record = {
            "xyz": np.asarray(pt, dtype=np.float64).tolist(),
            "provenance": "intersection",
            "of": [src_id, dst_id],
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
            _require_point(store, point_id, where=where),
            _require_plane(store, plane_id, where=where),
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
                "of": [point_id, plane_id],
                "op": "line_from_point_normal",
            },
        )
        return

    if op == "line_from_two_points":
        a = step["a"]
        b = step["b"]
        if a == b:
            raise ValueError(f"{where}: points a and b must differ")
        line = line_from_two_points(
            _require_point(store, a, where=where),
            _require_point(store, b, where=where),
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
                "of": [a, b],
                "op": "line_from_two_points",
            },
        )
        return

    if op == "midpoint_line_planes":
        line_id = step["line"]
        a = step["a"]
        b = step["b"]
        if a == b:
            raise ValueError(f"{where}: planes a and b must differ")
        line = _require_line(store, line_id, where=where)
        pa = _require_plane(store, a, where=where)
        pb = _require_plane(store, b, where=where)
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
                "of": [line_id, a, b],
                "op": "midpoint_line_planes",
                "ends": [
                    np.asarray(end_a, dtype=np.float64).tolist(),
                    np.asarray(end_b, dtype=np.float64).tolist(),
                ],
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
    if int(recipe.get("version", 1)) != 1:
        raise ValueError(f"unsupported recipe version {recipe.get('version')}")
    if recipe.get("units", "mm") != "mm":
        raise ValueError(f"recipe units must be mm, got {recipe.get('units')!r}")
    faces = recipe.get("faces") or {}
    if not isinstance(faces, dict) or not faces:
        raise ValueError("recipe.faces must be a non-empty object")
    construct = recipe.get("construct") or []
    if not isinstance(construct, list):
        raise ValueError("recipe.construct must be a list")
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


def _parse_recipe_frame(spec) -> dict | None:
    """Optional Align Z metadata: ``{axis, origin, flip_z}``. Not a construct step."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("recipe.frame must be an object")
    axis = spec.get("axis")
    origin = spec.get("origin")
    if not axis or not origin:
        raise ValueError("recipe.frame needs axis and origin ids")
    return {
        "axis": str(axis),
        "origin": str(origin),
        "flip_z": bool(spec.get("flip_z", False)),
    }


def _validate_frame_spec(session: "ReductionSession", spec: dict) -> None:
    axis = spec["axis"]
    origin = spec["origin"]
    if axis not in session._store:
        raise KeyError(f"recipe.frame.axis: unknown id {axis!r}")
    if origin not in session._store:
        raise KeyError(f"recipe.frame.origin: unknown id {origin!r}")
    if session.kind_of(axis) != "line":
        raise ValueError(f"recipe.frame.axis {axis!r} must be a line")
    if session.kind_of(origin) != "point":
        raise ValueError(f"recipe.frame.origin {origin!r} must be a point")


_MEASURE_OPS: dict[str, tuple[tuple[str, str], ...]] = {
    "distance_points": (("a", "point"), ("b", "point")),
    "distance_point_plane": (("point", "point"), ("plane", "plane")),
    "distance_point_line": (("point", "point"), ("line", "line")),
    "angle_planes": (("a", "plane"), ("b", "plane")),
    "angle_lines": (("a", "line"), ("b", "line")),
    "angle_line_plane": (("line", "line"), ("plane", "plane")),
}


def _parse_measure_spec(spec) -> dict:
    """Optional measurement: distances / angles. Not a construct step."""
    if not isinstance(spec, dict):
        raise ValueError("recipe.measures[] must be an object")
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
        if eid not in session._store:
            raise KeyError(f"recipe.measures.{mid}.{key}: unknown id {eid!r}")
        if session.kind_of(eid) != kind:
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
    an ``aligned`` copy is attached (same layout as GUI export with Align Z).
    """
    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {project_dir}")
    sess = ReductionSession()
    sess.apply_recipe(recipe, project_dir=project_dir)
    export_ids = recipe.get("export")
    result = sess.to_result(
        source_project=str(project_dir.resolve()),
        export=None if export_ids is None else [str(x) for x in export_ids],
    )
    frame = sess.rigid_frame()
    if frame is not None:
        from cloudet.frame import with_aligned_copy

        result = with_aligned_copy(result, frame)
    return result


def write_geometry_json(path: str | Path, result: ReductionResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path


def write_recipe_json(path: str | Path, recipe: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)
    return path


_REF_KEYS = ("id", "of", "a", "b", "c", "line", "plane", "point", "src", "dst")


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
            "point": 8.0,
            "line_diameter": 1.0,
        }
    )
    # Optional Align Z pick: {axis, origin, flip_z}. Display pose is not stored.
    frame_spec: dict | None = None
    # Pinned measurements: [{id, op, ...operands}]. Not construct entities.
    measures: list[dict] = field(default_factory=list)

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
            "point": 8.0,
            "line_diameter": 1.0,
        }
        self.frame_spec = None
        self.measures = []

    def ids(self, *, kind: str | None = None) -> list[str]:
        if kind is None:
            return list(self._store.keys())
        return [k for k, e in self._store.items() if e.kind == kind]

    def kind_of(self, entity_id: str) -> str:
        return self._store[entity_id].kind

    def record_of(self, entity_id: str) -> dict:
        return self._store[entity_id].record

    def plane(self, entity_id: str) -> Plane:
        ent = self._store[entity_id]
        if ent.kind != "plane":
            raise TypeError(f"{entity_id!r} is a {ent.kind}, expected plane")
        return ent.value

    def line(self, entity_id: str) -> Line:
        ent = self._store[entity_id]
        if ent.kind != "line":
            raise TypeError(f"{entity_id!r} is a {ent.kind}, expected line")
        return ent.value

    def point(self, entity_id: str) -> np.ndarray:
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
        record = {
            "abcd": plane.as_array().tolist(),
            "provenance": "scanned",
            "group_id": int(group_id),
            "group_name": str(group_name),
            "plane_index": int(plane_index),
            "quality": {
                k: quality[k]
                for k in (
                    "status",
                    "mad_sigma_mm",
                    "threshold_mm",
                    "n_points",
                    "bimodal",
                    "reasons",
                )
                if k in quality and quality[k] is not None
            },
        }
        self._store[alias] = _Entity(kind="plane", value=plane, record=record)
        self._face_specs[alias] = {
            "from": "group",
            "name": str(group_name),
            "plane_index": int(plane_index),
        }
        self.visible[alias] = True
        if anchor is not None:
            self.anchors[alias] = np.asarray(anchor, dtype=np.float64).reshape(3)

    def apply_step(self, step: dict) -> str:
        """Append and execute one construct step. Returns the new entity id."""
        step = dict(step)
        entity_id = step.get("id")
        if not entity_id:
            raise ValueError("construct step needs id")
        entity_id = str(entity_id)
        if entity_id in self._store:
            raise ValueError(f"duplicate id {entity_id!r}")
        _run_construct_step(self._store, step)
        self._construct.append(step)
        self.visible[entity_id] = True
        if step.get("op") == "line_from_two_points":
            pa = self.point(step["a"])
            pb = self.point(step["b"])
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
        of = step.get("of") or step.get("a") or step.get("src") or step.get("line")
        if of in self.anchors:
            self.anchors[entity_id] = self.anchors[of].copy()
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
        step = dict(step)
        step["id"] = entity_id
        if not step.get("op"):
            raise ValueError("construct step needs op")
        allowed = self.operand_ids_before(entity_id)
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

    def _replay_construct(self) -> None:
        """Drop construct results and re-run ``_construct`` from scanned faces."""
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
            if eid in self._store:
                self.visible[eid] = shown
        self.display_mm = {k: v for k, v in display_mm.items() if k in self._store}
        self.display_width_mm = {
            k: v for k, v in display_width.items() if k in self._store
        }
        if self.frame_spec:
            try:
                _validate_frame_spec(self, self.frame_spec)
            except Exception:
                self.frame_spec = None
        kept: list[dict] = []
        for spec in self.measures:
            try:
                _validate_measure_spec(self, spec)
                kept.append(spec)
            except Exception:
                continue
        self.measures = kept

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
        for alias, spec in (recipe.get("faces") or {}).items():
            alias = str(alias)
            resolved = None if bind_face is None else bind_face(alias, spec)
            if resolved is None:
                if project_path is None:
                    raise ValueError(f"faces.{alias}: cannot resolve (no project_dir)")
                plane, record = _bind_face(project_path, alias, spec)
                anchor = _anchor_from_project(project_path, record.get("group_id"))
            else:
                plane, record, anchor = resolved
            target.bind_scanned(
                alias,
                plane,
                group_name=str(record.get("group_name", alias)),
                group_id=int(record.get("group_id", 0)),
                plane_index=int(record.get("plane_index", 0)),
                quality=record.get("quality") or {},
                anchor=anchor,
            )
            target._face_specs[alias] = dict(spec)
        for step in recipe.get("construct") or []:
            target.apply_step(dict(step))
        frame_spec = _parse_recipe_frame(recipe.get("frame"))
        if frame_spec is not None:
            _validate_frame_spec(target, frame_spec)
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
        self.frame_spec = frame_spec
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
        for spec in self.measures:
            _rewrite_id_refs(spec, old_id, new_id)
        return new_id

    def dependents(self, entity_id: str) -> list[str]:
        """Construct result ids that transitively use ``entity_id``."""
        entity_id = str(entity_id)
        doomed: set[str] = {entity_id}
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
            if doomed & refs:
                self.frame_spec = None
        self.measures = [
            m for m in self.measures if not (_measure_operand_ids(m) & doomed)
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

    def offset(self, entity_id: str, of: str, distance_mm: float) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "offset",
            "of": of,
            "distance_mm": float(distance_mm),
        })

    def intersect_planes(self, entity_id: str, a: str, b: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "intersect_planes",
            "a": a,
            "b": b,
        })

    def intersect_line_plane(self, entity_id: str, line: str, plane: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "intersect_line_plane",
            "line": line,
            "plane": plane,
        })

    def intersect_three_planes(self, entity_id: str, a: str, b: str, c: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "intersect_three_planes",
            "a": a,
            "b": b,
            "c": c,
        })

    def intersect_normal_plane(
        self,
        entity_id: str,
        src: str,
        dst: str,
        through: np.ndarray | None = None,
    ) -> str:
        step: dict[str, Any] = {
            "id": str(entity_id),
            "op": "intersect_normal_plane",
            "src": src,
            "dst": dst,
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

    def line_from_two_points(self, entity_id: str, a: str, b: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "line_from_two_points",
            "a": a,
            "b": b,
        })

    def midpoint_line_planes(self, entity_id: str, line: str, a: str, b: str) -> str:
        return self.apply_step({
            "id": str(entity_id),
            "op": "midpoint_line_planes",
            "line": line,
            "a": a,
            "b": b,
        })

    def unique_id(self, prefix: str) -> str:
        n = 1
        while f"{prefix}_{n}" in self._store:
            n += 1
        return f"{prefix}_{n}"

    def to_recipe(self, export: list[str] | None = None) -> dict:
        if not self._face_specs:
            raise ValueError("no scanned faces bound")
        ids = list(self._store.keys()) if export is None else list(export)
        out = {
            "version": 1,
            "units": "mm",
            "faces": dict(self._face_specs),
            "construct": [dict(s) for s in self._construct],
            "export": ids,
        }
        if self.frame_spec:
            out["frame"] = dict(self.frame_spec)
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
            value = distance_points(self.point(parsed["a"]), self.point(parsed["b"]))
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
            value = angle_planes_deg(self.plane(parsed["a"]), self.plane(parsed["b"]))
            unit = "deg"
        elif op == "angle_lines":
            value = angle_lines_deg(self.line(parsed["a"]), self.line(parsed["b"]))
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

    def rigid_frame(self):
        """Align Z pose from ``frame_spec``, or ``None`` if unset."""
        spec = self.frame_spec
        if not spec:
            return None
        from cloudet.frame import RigidFrame

        line = self.line(spec["axis"])
        origin = self.point(spec["origin"])
        return RigidFrame.align_z(
            line.direction,
            origin,
            flip_z=bool(spec.get("flip_z", False)),
            axis_id=spec["axis"],
            origin_id=spec["origin"],
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
