"""Project directory I/O (the pipeline's on-disk contract).

Naming convention for this package
----------------------------------
* ``load_*`` / ``save_*`` — project-domain objects (settings, groups, caches,
  manifest).
* ``read_*`` / ``write_*`` — raw file formats only (e.g. PLY via ``plyio``).

Layout::

    <project>/
      manifest.json        units, source cloud info, detection params,
                           software version (written on every save)
      settings.json        sectioned picker settings (versioned schema)
      cache/               optional VoxelHashGrid / display-downsample
                           acceleration (not required to reload groups;
                           rebuilt if missing or stale)
      groups/
        group_000.ply          points, double precision
        group_000.json         metadata + fit summary (no pickle anywhere)
        group_000_indices.npy  indices into the source cloud (plain npy)
        group_000_p0_indices.npy  optional: inliers used to fit plane p0
        group_000_p1_indices.npy  optional: inliers used to fit plane p1

Reproducibility rule: everything needed to interpret the groups is baked
into manifest.json / group json at save time; nothing depends on the
mutable settings file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cloudet
from cloudet.fit.picking import PickParams
from cloudet.core.circle import Circle
from cloudet.core.cylinder import Cylinder
from cloudet.core.plane import Plane
from cloudet.core.plyio import write_ply_xyz
from cloudet.project.schema import (
    circle_from_json,
    circle_to_json,
    cylinder_from_json,
    cylinder_to_json,
    migrate_group_doc,
    plane_from_json,
    plane_to_json,
)

__all__ = [
    "SourceInfo",
    "CloudetSettings",
    "FittedPlane",
    "FittedCylinder",
    "FittedCircle",
    "save_group",
    "save_manifest",
    "load_manifest",
    "load_group_indices",
    "load_plane_inlier_indices",
    "plane_inlier_indices_path",
    "load_group_doc",
    "load_group_docs",
    "load_group_fit",
    "load_fitted_plane",
    "load_fitted_cylinder",
    "load_fitted_circle",
    "load_settings",
    "save_settings",
]


@dataclass(frozen=True)
class SourceInfo:
    path: str
    n_points: int
    size_bytes: int | None = None


@dataclass
class ViewSettings:
    """GUI preferences (units mm where applicable).

    Display downsampling only affects rendering; picking, fitting and
    saving always use the full-resolution cloud. Large clouds crash the
    renderer if sent unfiltered, so ``display_max_points`` is a hard
    cap enforced by random subsampling after the optional voxel filter.
    """

    base_point_size: float = 1.0
    active_point_size: float = 6.0
    inactive_point_size: float = 2.5
    display_voxel_size_mm: float = 0.5  # 0 = no voxel filter
    display_max_points: int = 4_000_000  # hard cap per geometry
    # Display-only decimation: "auto" prefers CuPy, then Open3D, else numpy.
    display_downsample_backend: str = "auto"  # auto | numpy | open3d | cupy
    axis_size_mm: float = 100.0
    axis_margin_mm: float = 20.0


@dataclass
class CloudetSettings:
    detection: PickParams = PickParams()
    view: ViewSettings = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.view is None:
            self.view = ViewSettings()


def save_settings(project_dir: str | Path, settings: CloudetSettings) -> Path:
    path = Path(project_dir) / "settings.json"
    doc = {
        "version": 1,
        "units": "mm",
        "detection": asdict(settings.detection),
        "view": asdict(settings.view),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return path


def load_settings(project_dir: str | Path, warn=print) -> CloudetSettings:
    path = Path(project_dir) / "settings.json"
    if not path.exists():
        return CloudetSettings()
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("version", 1) != 1:
        warn(f"settings version {doc.get('version')} is newer than supported (1)")

    from dataclasses import replace

    def build(cls, section: str, data: dict | None = None):
        raw = dict(data if data is not None else doc.get(section, {}))
        known = {f: raw[f] for f in cls.__dataclass_fields__ if f in raw}
        unknown = set(raw) - set(known)
        if unknown:
            warn(f"settings: ignoring unknown keys in [{section}]: {sorted(unknown)}")
        return cls(**known)

    det_raw = dict(doc.get("detection", {}))
    view_raw = dict(doc.get("view", {}))
    # Legacy: compute_backend lived under view; prefer detection if both exist.
    legacy_compute = view_raw.pop("compute_backend", None)
    if "compute_backend" not in det_raw and legacy_compute is not None:
        det_raw["compute_backend"] = legacy_compute

    detection = build(PickParams, "detection", det_raw)
    if detection.ransac_backend == "numpy":
        detection = replace(detection, ransac_backend="seeded")

    return CloudetSettings(
        detection=detection,
        view=build(ViewSettings, "view", view_raw),
    )


def save_manifest(
    project_dir: str | Path,
    source: SourceInfo,
    detection: PickParams,
    n_groups: int,
) -> Path:
    path = Path(project_dir) / "manifest.json"
    doc = {
        "version": 1,
        "units": "mm",
        "source": asdict(source),
        "detection": asdict(detection),
        "n_groups": n_groups,
        "software": {"cloudet": cloudet.__version__},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return path


def load_manifest(project_dir: str | Path) -> dict | None:
    path = Path(project_dir) / "manifest.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _jsonable_fit_summary(fit_summary: dict | None) -> dict | None:
    """Drop GUI-only numpy caches; keep plane/cylinder/circle equations."""
    if not fit_summary:
        return fit_summary
    out: dict = {}

    planes = fit_summary.get("planes")
    if isinstance(planes, list):
        keep = (
            "plane_index",
            "normal",
            "d",
            "abcd",  # accepted on input; rewritten below
            "n_points",
            "status",
            "reasons",
            "bimodal",
            "mad_sigma_mm",
            "threshold_mm",
            "source_plane_index",
            "n_selected",
            "name",
            "inlier_indices_file",
            "inlier_n",
        )
        out_planes = []
        for p in planes:
            if not isinstance(p, dict):
                continue
            entry = {k: p[k] for k in keep if k in p}
            try:
                plane = plane_from_json(entry)
            except (KeyError, TypeError, ValueError):
                continue
            entry.update(plane_to_json(plane))
            entry.pop("abcd", None)
            out_planes.append(entry)
        out["planes"] = out_planes

    cyl_keep = (
        "cylinder_index",
        "point",
        "direction",
        "diameter_mm",
        "diameter_fixed",
        "n_points",
        "status",
        "reasons",
        "mad_sigma_mm",
        "threshold_mm",
        "name",
        "inlier_indices_file",
        "inlier_n",
    )
    cylinders = fit_summary.get("cylinders")
    if isinstance(cylinders, list):
        out_cyl = []
        for c in cylinders:
            if not isinstance(c, dict):
                continue
            entry = {k: c[k] for k in cyl_keep if k in c}
            try:
                cyl = cylinder_from_json(entry)
            except (KeyError, TypeError, ValueError):
                continue
            entry.update(cylinder_to_json(cyl))
            out_cyl.append(entry)
        out["cylinders"] = out_cyl

    cir_keep = (
        "circle_index",
        "center",
        "normal",
        "diameter_mm",
        "diameter_fixed",
        "n_points",
        "status",
        "reasons",
        "mad_sigma_mm",
        "threshold_mm",
        "name",
        "support_plane_index",
        "n_selected",
        "inlier_indices_file",
        "inlier_n",
    )
    circles = fit_summary.get("circles")
    if isinstance(circles, list):
        out_cir = []
        for c in circles:
            if not isinstance(c, dict):
                continue
            entry = {k: c[k] for k in cir_keep if k in c}
            try:
                cir = circle_from_json(entry)
            except (KeyError, TypeError, ValueError):
                continue
            entry.update(circle_to_json(cir))
            out_cir.append(entry)
        out["circles"] = out_cir

    # Pass through legacy / non-primitive fit payloads unchanged.
    if not out:
        return fit_summary
    return out


def save_group(
    project_dir: str | Path,
    group_id: int,
    name: str,
    points: np.ndarray,
    indices: np.ndarray | None,
    coarse_plane: np.ndarray | None,
    clicked: np.ndarray | None,
    color: np.ndarray | None = None,
    detection: PickParams | None = None,
    fit_summary: dict | None = None,
) -> Path:
    """Write one group (ply + json + optional indices.npy). Returns json path.

    When ``fit_summary["planes"]`` carries ``inlier_local`` (indices into
    ``indices``) or ``inlier_source`` (indices into the source cloud), each
    plane is also written as ``group_{id:03d}_p{k}_indices.npy``.
    """
    groups_dir = Path(project_dir) / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)

    ply_name = f"group_{group_id:03d}.ply"
    write_ply_xyz(groups_dir / ply_name, points)

    indices_name = None
    if indices is not None:
        indices_name = f"group_{group_id:03d}_indices.npy"
        np.save(groups_dir / indices_name, np.asarray(indices, dtype=np.int64))

    doc = {
        "version": 1,
        "units": "mm",
        "group_id": int(group_id),
        "name": str(name),
        "num_points": int(len(points)),
        "ply_file": ply_name,
        "indices_file": indices_name,
        "coarse_plane": None if coarse_plane is None else np.asarray(coarse_plane).tolist(),
        "clicked": None if clicked is None else np.asarray(clicked).tolist(),
        "color": None if color is None else np.asarray(color).tolist(),
        "detection": None if detection is None else asdict(detection),
        "fit": _write_plane_inlier_files(project_dir, group_id, indices, fit_summary),
        "software": {"cloudet": cloudet.__version__},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    json_path = groups_dir / f"group_{group_id:03d}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return json_path


def load_group_indices(project_dir: str | Path, group_id: int) -> np.ndarray | None:
    path = Path(project_dir) / "groups" / f"group_{group_id:03d}_indices.npy"
    if not path.exists():
        return None
    return np.load(path)  # plain int64 npy, no pickle


def plane_inlier_indices_path(
    project_dir: str | Path, group_id: int, plane_index: int
) -> Path:
    return (
        Path(project_dir)
        / "groups"
        / f"group_{int(group_id):03d}_p{int(plane_index)}_indices.npy"
    )


def load_plane_inlier_indices(
    project_dir: str | Path, group_id: int, plane_index: int
) -> np.ndarray | None:
    """Indices into the source cloud used to fit one plane (int64 npy)."""
    path = plane_inlier_indices_path(project_dir, group_id, plane_index)
    if not path.exists():
        return None
    return np.load(path)


def _write_plane_inlier_files(
    project_dir: str | Path,
    group_id: int,
    group_indices: np.ndarray | None,
    fit_summary: dict | None,
) -> dict | None:
    """Persist per-plane inliers next to the group; return JSON-safe fit."""
    summary = _jsonable_fit_summary(fit_summary)
    if not summary:
        return summary
    groups_dir = Path(project_dir) / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    gidx = None if group_indices is None else np.asarray(group_indices, dtype=np.int64)
    src_planes = (fit_summary or {}).get("planes") or []
    written: set[str] = set()
    if isinstance(summary.get("planes"), list):
        for i, entry in enumerate(summary["planes"]):
            pi = int(entry.get("plane_index", i))
            src = (
                src_planes[i]
                if i < len(src_planes) and isinstance(src_planes[i], dict)
                else {}
            )
            local = src.get("inlier_local")
            source = src.get("inlier_source")
            path = plane_inlier_indices_path(project_dir, group_id, pi)
            if source is not None:
                idx = np.asarray(source, dtype=np.int64)
            elif local is not None and gidx is not None:
                loc = np.asarray(local, dtype=np.int64)
                if loc.size and (loc.min() < 0 or loc.max() >= len(gidx)):
                    raise ValueError(
                        f"group {group_id} p{pi}: inlier_local out of range "
                        "for group indices"
                    )
                idx = gidx[loc]
            else:
                if path.exists():
                    path.unlink()
                entry.pop("inlier_indices_file", None)
                entry.pop("inlier_n", None)
                continue
            np.save(path, idx)
            written.add(path.name)
            entry["inlier_indices_file"] = path.name
            entry["inlier_n"] = int(len(idx))
            if "n_points" not in entry:
                entry["n_points"] = int(len(idx))
    prefix = f"group_{int(group_id):03d}_p"
    for leftover in groups_dir.glob(f"{prefix}*_indices.npy"):
        if leftover.name not in written:
            leftover.unlink()
    return summary


@dataclass(frozen=True)
class FittedPlane:
    """One fitted plane loaded from a saved group JSON."""

    group_id: int
    group_name: str
    plane_index: int
    plane: Plane
    quality: dict


@dataclass(frozen=True)
class FittedCylinder:
    """One fitted cylinder loaded from a saved group JSON."""

    group_id: int
    group_name: str
    cylinder_index: int
    cylinder: Cylinder
    quality: dict


@dataclass(frozen=True)
class FittedCircle:
    """One fitted circle loaded from a saved group JSON."""

    group_id: int
    group_name: str
    circle_index: int
    circle: Circle
    quality: dict


def load_group_doc(project_dir: str | Path, group_id: int) -> dict | None:
    """Load one ``groups/group_{id:03d}.json`` document, or None if missing."""
    path = Path(project_dir) / "groups" / f"group_{int(group_id):03d}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return migrate_group_doc(json.load(f))


def load_group_docs(project_dir: str | Path) -> list[dict]:
    """Load all ``groups/group_*.json`` documents (sorted by group_id)."""
    groups_dir = Path(project_dir) / "groups"
    if not groups_dir.is_dir():
        raise FileNotFoundError(f"no groups/ directory in {project_dir}")
    docs = []
    for path in sorted(groups_dir.glob("group_*.json")):
        with open(path, encoding="utf-8") as f:
            docs.append(migrate_group_doc(json.load(f)))
    docs.sort(key=lambda d: int(d["group_id"]))
    return docs


def load_group_fit(
    project_dir: str | Path,
    group_id: int,
    group_indices: np.ndarray | None = None,
) -> dict | None:
    """Restore ``fit`` (planes, cylinders, circles) from a saved group JSON.

    Attaches plane inlier arrays when ``group_*_p*_indices.npy`` files exist.
    Returns None when the group has no usable fit entities.
    """
    doc = load_group_doc(project_dir, group_id)
    if not doc:
        return None
    fit = doc.get("fit")
    if not isinstance(fit, dict):
        return None

    gidx = None if group_indices is None else np.asarray(group_indices, dtype=np.int64)
    lookup = None if gidx is None else {int(v): i for i, v in enumerate(gidx)}

    planes: list[dict] = []
    raw_planes = fit.get("planes")
    if isinstance(raw_planes, list):
        for p in raw_planes:
            if not isinstance(p, dict):
                continue
            if "abcd" not in p and not ("normal" in p and "d" in p):
                continue
            entry = dict(p)
            pi = int(entry.get("plane_index", 0))
            src = load_plane_inlier_indices(project_dir, group_id, pi)
            if src is not None:
                src = np.asarray(src, dtype=np.int64)
                entry["inlier_source"] = src
                entry["inlier_n"] = int(len(src))
                if lookup is not None:
                    local = np.array(
                        [lookup[int(s)] for s in src if int(s) in lookup],
                        dtype=np.int64,
                    )
                    if len(local) == len(src):
                        entry["inlier_local"] = local
            planes.append(entry)

    cylinders: list[dict] = []
    for c in fit.get("cylinders") or []:
        if not isinstance(c, dict):
            continue
        try:
            cyl = cylinder_from_json(c)
        except (KeyError, TypeError, ValueError):
            continue
        entry = dict(c)
        entry.update(cylinder_to_json(cyl))
        cylinders.append(entry)

    circles: list[dict] = []
    for c in fit.get("circles") or []:
        if not isinstance(c, dict):
            continue
        try:
            cir = circle_from_json(c)
        except (KeyError, TypeError, ValueError):
            continue
        entry = dict(c)
        entry.update(circle_to_json(cir))
        circles.append(entry)

    if not planes and not cylinders and not circles:
        return None
    out: dict = {"planes": planes}
    if cylinders:
        out["cylinders"] = cylinders
    if circles:
        out["circles"] = circles
    return out


def _planes_from_fit(fit: dict | None, *, group_label: str) -> list[dict]:
    if fit is None:
        raise ValueError(f"{group_label}: no fit summary saved; run Fit then save")
    planes = fit.get("planes")
    if isinstance(planes, list) and planes:
        return planes
    # Legacy flat fit without planes[] cannot drive geometry reduction unless abcd/normal present.
    try:
        plane = plane_from_json(fit)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"{group_label}: fit has no planes[] with normal/d "
            "(re-Fit and save, or use a project with fit.planes entries)"
        ) from e
    return [{
        "plane_index": 0,
        **plane_to_json(plane),
        "status": fit.get("status"),
        "mad_sigma_mm": fit.get("mad_sigma_mm"),
        "threshold_mm": fit.get("threshold_mm"),
        "reasons": fit.get("reasons", []),
        "bimodal": fit.get("bimodal", False),
        "n_points": fit.get("n_points"),
    }]


def load_fitted_plane(
    project_dir: str | Path,
    *,
    name: str | None = None,
    group_id: int | None = None,
    plane_index: int = 0,
) -> FittedPlane:
    """Resolve one fitted plane by group name or id.

    Default ``plane_index=0`` selects the dominant / only plane in a group.
    """
    if (name is None) == (group_id is None):
        raise ValueError("provide exactly one of name= or group_id=")

    docs = load_group_docs(project_dir)
    if name is not None:
        matches = [d for d in docs if str(d.get("name", "")) == name]
        if not matches:
            known = [str(d.get("name", f"G{d['group_id']}")) for d in docs]
            raise KeyError(f"no group named {name!r} (known: {known})")
        if len(matches) > 1:
            raise ValueError(f"multiple groups named {name!r}; use group_id=")
        doc = matches[0]
    else:
        matches = [d for d in docs if int(d["group_id"]) == int(group_id)]
        if not matches:
            raise KeyError(f"no group_id={group_id}")
        doc = matches[0]

    label = f"group {doc.get('name', doc['group_id'])!r}"
    planes = _planes_from_fit(doc.get("fit"), group_label=label)
    try:
        entry = next(p for p in planes if int(p.get("plane_index", 0)) == int(plane_index))
    except StopIteration as e:
        indices = [int(p.get("plane_index", 0)) for p in planes]
        raise KeyError(
            f"{label}: no plane_index={plane_index} (have {indices})"
        ) from e
    try:
        plane = plane_from_json(entry)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{label} plane {plane_index}: missing normal/d (or abcd)") from e

    quality = {
        "status": entry.get("status"),
        "mad_sigma_mm": entry.get("mad_sigma_mm"),
        "threshold_mm": entry.get("threshold_mm"),
        "reasons": entry.get("reasons", []),
        "bimodal": entry.get("bimodal", False),
        "n_points": entry.get("n_points"),
    }
    return FittedPlane(
        group_id=int(doc["group_id"]),
        group_name=str(doc.get("name", f"G{doc['group_id']}")),
        plane_index=int(plane_index),
        plane=plane,
        quality=quality,
    )


def _resolve_group_doc(
    project_dir: str | Path,
    *,
    name: str | None = None,
    group_id: int | None = None,
) -> dict:
    if (name is None) == (group_id is None):
        raise ValueError("provide exactly one of name= or group_id=")
    docs = load_group_docs(project_dir)
    if name is not None:
        matches = [d for d in docs if str(d.get("name", "")) == name]
        if not matches:
            known = [str(d.get("name", f"G{d['group_id']}")) for d in docs]
            raise KeyError(f"no group named {name!r} (known: {known})")
        if len(matches) > 1:
            raise ValueError(f"multiple groups named {name!r}; use group_id=")
        return matches[0]
    matches = [d for d in docs if int(d["group_id"]) == int(group_id)]
    if not matches:
        raise KeyError(f"no group_id={group_id}")
    return matches[0]


def load_fitted_cylinder(
    project_dir: str | Path,
    *,
    name: str | None = None,
    group_id: int | None = None,
    cylinder_index: int = 0,
) -> FittedCylinder:
    """Resolve one fitted cylinder by group name or id."""
    doc = _resolve_group_doc(project_dir, name=name, group_id=group_id)
    label = f"group {doc.get('name', doc['group_id'])!r}"
    fit = doc.get("fit") or {}
    cylinders = fit.get("cylinders")
    if not isinstance(cylinders, list) or not cylinders:
        raise ValueError(f"{label}: no fit.cylinders[]; run cylinder Fit then save")
    try:
        entry = next(
            c for c in cylinders if int(c.get("cylinder_index", 0)) == int(cylinder_index)
        )
    except StopIteration as e:
        indices = [int(c.get("cylinder_index", 0)) for c in cylinders]
        raise KeyError(
            f"{label}: no cylinder_index={cylinder_index} (have {indices})"
        ) from e
    cyl = cylinder_from_json(entry)
    quality = {
        "status": entry.get("status"),
        "mad_sigma_mm": entry.get("mad_sigma_mm"),
        "threshold_mm": entry.get("threshold_mm"),
        "reasons": entry.get("reasons", []),
        "n_points": entry.get("n_points"),
        "diameter_fixed": bool(entry.get("diameter_fixed", False)),
    }
    return FittedCylinder(
        group_id=int(doc["group_id"]),
        group_name=str(doc.get("name", f"G{doc['group_id']}")),
        cylinder_index=int(cylinder_index),
        cylinder=cyl,
        quality=quality,
    )


def load_fitted_circle(
    project_dir: str | Path,
    *,
    name: str | None = None,
    group_id: int | None = None,
    circle_index: int = 0,
) -> FittedCircle:
    """Resolve one fitted circle by group name or id."""
    doc = _resolve_group_doc(project_dir, name=name, group_id=group_id)
    label = f"group {doc.get('name', doc['group_id'])!r}"
    fit = doc.get("fit") or {}
    circles = fit.get("circles")
    if not isinstance(circles, list) or not circles:
        raise ValueError(f"{label}: no fit.circles[]; run circle Fit then save")
    try:
        entry = next(
            c for c in circles if int(c.get("circle_index", 0)) == int(circle_index)
        )
    except StopIteration as e:
        indices = [int(c.get("circle_index", 0)) for c in circles]
        raise KeyError(
            f"{label}: no circle_index={circle_index} (have {indices})"
        ) from e
    cir = circle_from_json(entry)
    quality = {
        "status": entry.get("status"),
        "mad_sigma_mm": entry.get("mad_sigma_mm"),
        "threshold_mm": entry.get("threshold_mm"),
        "reasons": entry.get("reasons", []),
        "n_points": entry.get("n_points"),
        "diameter_fixed": bool(entry.get("diameter_fixed", False)),
    }
    return FittedCircle(
        group_id=int(doc["group_id"]),
        group_name=str(doc.get("name", f"G{doc['group_id']}")),
        circle_index=int(circle_index),
        circle=cir,
        quality=quality,
    )
