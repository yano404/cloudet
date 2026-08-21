"""On-disk JSON schema helpers: plane records, recipe/geometry migration.

Canonical plane record (v2)::

    {"normal": [nx, ny, nz], "d": d, ...}

Legacy ``abcd`` and recipe v1 operand keys are accepted on read and rewritten
to the canonical form by the ``migrate_*`` functions.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from cloudet.core.circle import Circle
from cloudet.core.cylinder import Cylinder
from cloudet.core.plane import Plane

__all__ = [
    "RECIPE_VERSION",
    "migrate_recipe",
    "migrate_construct_step",
    "migrate_measure_spec",
    "migrate_plane_record",
    "migrate_geometry",
    "migrate_group_doc",
    "plane_to_json",
    "plane_from_json",
    "cylinder_to_json",
    "cylinder_from_json",
    "circle_to_json",
    "circle_from_json",
    "recipe_needs_migrate",
    "geometry_needs_migrate",
    "group_doc_needs_migrate",
    "migrate_project",
]

RECIPE_VERSION = 2

# recipe_op → {old_key: new_key}
_CONSTRUCT_V1_KEYS: dict[str, dict[str, str]] = {
    "offset": {"of": "plane"},
    "intersect_planes": {"a": "plane_a", "b": "plane_b"},
    "intersect_three_planes": {"a": "plane_a", "b": "plane_b", "c": "plane_c"},
    "line_from_two_points": {"a": "point_a", "b": "point_b"},
    "midpoint_line_planes": {"a": "plane_a", "b": "plane_b"},
    "plane_from_two_lines": {"a": "line_a", "b": "line_b"},
    "intersect_normal_plane": {
        "src": "source_plane",
        "dst": "destination_plane",
    },
}

_MEASURE_V1_KEYS: dict[str, dict[str, str]] = {
    "distance_points": {"a": "point_a", "b": "point_b"},
    "angle_planes": {"a": "plane_a", "b": "plane_b"},
    "angle_lines": {"a": "line_a", "b": "line_b"},
}


def plane_to_json(plane: Plane) -> dict[str, Any]:
    """Serialize a plane to the canonical ``normal`` + ``d`` dict."""
    return {
        "normal": np.asarray(plane.normal, dtype=np.float64).tolist(),
        "d": float(plane.d),
    }


def plane_from_json(rec: dict | list | np.ndarray) -> Plane:
    """Load a plane from ``normal``/``d``, legacy ``abcd``, or a length-4 array."""
    if isinstance(rec, (list, tuple, np.ndarray)):
        return Plane.from_array(rec)
    if not isinstance(rec, dict):
        raise TypeError(f"plane record must be dict or abcd array, got {type(rec).__name__}")
    if "normal" in rec and "d" in rec:
        return Plane(np.asarray(rec["normal"], dtype=np.float64), float(rec["d"]))
    if "abcd" in rec:
        return Plane.from_array(rec["abcd"])
    raise ValueError("plane record needs normal+d or abcd")


def cylinder_to_json(cyl: Cylinder) -> dict[str, Any]:
    """Serialize a cylinder (axis + diameter_mm)."""
    return {
        "point": np.asarray(cyl.point, dtype=np.float64).tolist(),
        "direction": np.asarray(cyl.direction, dtype=np.float64).tolist(),
        "diameter_mm": float(cyl.diameter_mm),
        "diameter_fixed": bool(cyl.diameter_fixed),
    }


def cylinder_from_json(rec: dict) -> Cylinder:
    """Load a cylinder from point/direction/diameter_mm."""
    if not isinstance(rec, dict):
        raise TypeError(f"cylinder record must be dict, got {type(rec).__name__}")
    if "point" not in rec or "direction" not in rec or "diameter_mm" not in rec:
        raise ValueError("cylinder record needs point, direction, diameter_mm")
    return Cylinder(
        point=np.asarray(rec["point"], dtype=np.float64),
        direction=np.asarray(rec["direction"], dtype=np.float64),
        diameter_mm=float(rec["diameter_mm"]),
        diameter_fixed=bool(rec.get("diameter_fixed", False)),
    )


def circle_to_json(cir: Circle) -> dict[str, Any]:
    """Serialize a circle (center + normal + diameter_mm)."""
    return {
        "center": np.asarray(cir.center, dtype=np.float64).tolist(),
        "normal": np.asarray(cir.normal, dtype=np.float64).tolist(),
        "diameter_mm": float(cir.diameter_mm),
        "diameter_fixed": bool(cir.diameter_fixed),
    }


def circle_from_json(rec: dict) -> Circle:
    """Load a circle from center/normal/diameter_mm."""
    if not isinstance(rec, dict):
        raise TypeError(f"circle record must be dict, got {type(rec).__name__}")
    if "center" not in rec or "normal" not in rec or "diameter_mm" not in rec:
        raise ValueError("circle record needs center, normal, diameter_mm")
    return Circle(
        center=np.asarray(rec["center"], dtype=np.float64),
        normal=np.asarray(rec["normal"], dtype=np.float64),
        diameter_mm=float(rec["diameter_mm"]),
        diameter_fixed=bool(rec.get("diameter_fixed", False)),
    )


def _rename_keys(obj: dict, mapping: dict[str, str]) -> dict:
    out = dict(obj)
    for old, new in mapping.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
        elif old in out and new in out:
            out.pop(old)
    return out


def _migrate_construct_step(step: dict) -> dict:
    if not isinstance(step, dict):
        return step
    out = dict(step)
    op = str(out.get("op", ""))
    mapping = _CONSTRUCT_V1_KEYS.get(op)
    if mapping:
        out = _rename_keys(out, mapping)
    return out


def migrate_construct_step(step: dict) -> dict:
    """Normalize one construct step to v2 operand keys."""
    return _migrate_construct_step(step)


def _migrate_measure_spec(spec: dict) -> dict:
    if not isinstance(spec, dict):
        return spec
    out = dict(spec)
    op = str(out.get("op", ""))
    mapping = _MEASURE_V1_KEYS.get(op)
    if mapping:
        out = _rename_keys(out, mapping)
    return out


def migrate_measure_spec(spec: dict) -> dict:
    """Normalize one measure spec to v2 operand keys."""
    return _migrate_measure_spec(spec)


def migrate_recipe(doc: dict) -> dict:
    """Return a deep-copied recipe normalized to ``RECIPE_VERSION``."""
    out = copy.deepcopy(doc)
    version = int(out.get("version", 1))
    if version > RECIPE_VERSION:
        raise ValueError(f"unsupported recipe version {version}")
    construct = out.get("construct")
    if isinstance(construct, list):
        out["construct"] = [
            _migrate_construct_step(s) if isinstance(s, dict) else s for s in construct
        ]
    measures = out.get("measures")
    if isinstance(measures, list):
        out["measures"] = [
            _migrate_measure_spec(m) if isinstance(m, dict) else m for m in measures
        ]
    out["version"] = RECIPE_VERSION
    return out


def migrate_plane_record(rec: dict) -> dict:
    """Normalize a plane entity / fit entry: ``abcd`` → ``normal``/``d``, ``of`` → ``parents``."""
    if not isinstance(rec, dict):
        return rec
    out = dict(rec)
    if "abcd" in out:
        plane = Plane.from_array(out.pop("abcd"))
        out["normal"] = plane.normal.tolist()
        out["d"] = float(plane.d)
    if "of" in out and "parents" not in out:
        out["parents"] = out.pop("of")
    elif "of" in out and "parents" in out:
        out.pop("of")
    return out


def _migrate_entity_map(entities: dict | None) -> dict | None:
    if not isinstance(entities, dict):
        return entities
    out: dict[str, Any] = {}
    for eid, rec in entities.items():
        if not isinstance(rec, dict):
            out[eid] = rec
            continue
        kind = rec.get("kind")
        # Plane records either have abcd or are known plane entities.
        if "abcd" in rec or ("normal" in rec and "d" in rec) or kind == "plane":
            out[eid] = migrate_plane_record(rec)
        else:
            migrated = dict(rec)
            if "of" in migrated and "parents" not in migrated:
                migrated["parents"] = migrated.pop("of")
            elif "of" in migrated:
                migrated.pop("of")
            out[eid] = migrated
    return out


def migrate_geometry(doc: dict) -> dict:
    """Normalize geometry.json (planes/lines/points, aligned, recipe.echo)."""
    out = copy.deepcopy(doc)
    for key in ("planes", "lines", "points"):
        if key in out:
            out[key] = _migrate_entity_map(out.get(key)) or {}
    aligned = out.get("aligned")
    if isinstance(aligned, dict):
        aligned_out = dict(aligned)
        for key in ("planes", "lines", "points"):
            if key in aligned_out:
                aligned_out[key] = _migrate_entity_map(aligned_out.get(key)) or {}
        out["aligned"] = aligned_out
    recipe = out.get("recipe")
    if isinstance(recipe, dict):
        recipe_out = dict(recipe)
        echo = recipe_out.get("echo")
        if isinstance(echo, dict):
            recipe_out["echo"] = migrate_recipe(echo)
        out["recipe"] = recipe_out
    return out


def migrate_group_doc(doc: dict) -> dict:
    """Normalize group_*.json fit.planes[] plane coefficients."""
    out = copy.deepcopy(doc)
    fit = out.get("fit")
    if isinstance(fit, dict):
        fit_out = dict(fit)
        planes = fit_out.get("planes")
        if isinstance(planes, list):
            fit_out["planes"] = [
                migrate_plane_record(p) if isinstance(p, dict) else p for p in planes
            ]
        # cylinders / circles already use diameter_mm; keep as-is if present
        for key in ("cylinders", "circles"):
            items = fit_out.get(key)
            if isinstance(items, list):
                fit_out[key] = [dict(x) if isinstance(x, dict) else x for x in items]
        # Legacy single-plane abcd on fit itself.
        if "abcd" in fit_out:
            plane = Plane.from_array(fit_out.pop("abcd"))
            fit_out["normal"] = plane.normal.tolist()
            fit_out["d"] = float(plane.d)
        out["fit"] = fit_out
    return out


def recipe_needs_migrate(doc: dict) -> bool:
    if int(doc.get("version", 1)) < RECIPE_VERSION:
        return True
    for step in doc.get("construct") or []:
        if not isinstance(step, dict):
            continue
        mapping = _CONSTRUCT_V1_KEYS.get(str(step.get("op", "")))
        if mapping and any(k in step for k in mapping):
            return True
    for spec in doc.get("measures") or []:
        if not isinstance(spec, dict):
            continue
        mapping = _MEASURE_V1_KEYS.get(str(spec.get("op", "")))
        if mapping and any(k in spec for k in mapping):
            return True
    return False


def _entity_map_needs_migrate(entities: dict | None) -> bool:
    if not isinstance(entities, dict):
        return False
    for rec in entities.values():
        if not isinstance(rec, dict):
            continue
        if "abcd" in rec or ("of" in rec and "parents" not in rec):
            return True
    return False


def geometry_needs_migrate(doc: dict) -> bool:
    for key in ("planes", "lines", "points"):
        if _entity_map_needs_migrate(doc.get(key)):
            return True
    aligned = doc.get("aligned")
    if isinstance(aligned, dict):
        for key in ("planes", "lines", "points"):
            if _entity_map_needs_migrate(aligned.get(key)):
                return True
    recipe = doc.get("recipe")
    if isinstance(recipe, dict):
        echo = recipe.get("echo")
        if isinstance(echo, dict) and recipe_needs_migrate(echo):
            return True
    return False


def group_doc_needs_migrate(doc: dict) -> bool:
    fit = doc.get("fit")
    if not isinstance(fit, dict):
        return False
    if "abcd" in fit:
        return True
    planes = fit.get("planes")
    if isinstance(planes, list):
        for p in planes:
            if isinstance(p, dict) and "abcd" in p:
                return True
    return False


def _atomic_write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def migrate_project(
    project_dir: str | Path,
    *,
    recipe_path: str | Path | None = None,
    geometry_path: str | Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Migrate group docs and optional recipe/geometry files. Returns changed paths."""
    root = Path(project_dir)
    changed: list[str] = []

    groups_dir = root / "groups"
    if groups_dir.is_dir():
        for path in sorted(groups_dir.glob("group_*.json")):
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            if not group_doc_needs_migrate(doc):
                continue
            changed.append(str(path))
            if not dry_run:
                _atomic_write_json(path, migrate_group_doc(doc))

    candidates: list[Path] = []
    if recipe_path is not None:
        candidates.append(Path(recipe_path))
    else:
        for name in ("geometry_recipe.json", "recipe.json"):
            p = root / name
            if p.is_file():
                candidates.append(p)
    if geometry_path is not None:
        candidates.append(Path(geometry_path))
    else:
        p = root / "geometry.json"
        if p.is_file():
            candidates.append(p)

    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        # Heuristic: recipe has faces+construct; geometry has planes/lines/points.
        if "faces" in doc or "construct" in doc:
            if not recipe_needs_migrate(doc):
                continue
            changed.append(str(path))
            if not dry_run:
                _atomic_write_json(path, migrate_recipe(doc))
        else:
            if not geometry_needs_migrate(doc):
                continue
            changed.append(str(path))
            if not dry_run:
                _atomic_write_json(path, migrate_geometry(doc))

    return changed
