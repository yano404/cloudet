"""Project directory I/O (the pipeline's on-disk contract).

Layout::

    <project>/
      manifest.json        units, source cloud info, detection params,
                           software version (written on every save)
      settings.json        sectioned picker settings (versioned schema)
      groups/
        group_000.ply          points, double precision
        group_000.json         metadata + fit summary (no pickle anywhere)
        group_000_indices.npy  indices into the source cloud (plain npy)

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

import detpos
from detpos.picking import PickParams
from detpos.plyio import write_ply_xyz

__all__ = [
    "SourceInfo",
    "PickerSettings",
    "save_group",
    "write_manifest",
    "read_manifest",
    "load_group_indices",
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
    """GUI preferences (units mm where applicable)."""

    base_point_size: float = 1.0
    active_point_size: float = 6.0
    inactive_point_size: float = 2.5
    group_vis_voxel_size_mm: float = 0.0
    axis_size_mm: float = 100.0
    axis_margin_mm: float = 20.0


@dataclass
class PickerSettings:
    detection: PickParams = PickParams()
    view: ViewSettings = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.view is None:
            self.view = ViewSettings()


def save_settings(project_dir: str | Path, settings: PickerSettings) -> Path:
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


def load_settings(project_dir: str | Path, warn=print) -> PickerSettings:
    path = Path(project_dir) / "settings.json"
    if not path.exists():
        return PickerSettings()
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("version", 1) != 1:
        warn(f"settings version {doc.get('version')} is newer than supported (1)")

    def build(cls, section: str):
        data = doc.get(section, {})
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        unknown = set(data) - set(known)
        if unknown:
            warn(f"settings: ignoring unknown keys in [{section}]: {sorted(unknown)}")
        return cls(**known)

    return PickerSettings(
        detection=build(PickParams, "detection"),
        view=build(ViewSettings, "view"),
    )


def write_manifest(
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
        "software": {"detpos": detpos.__version__},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return path


def read_manifest(project_dir: str | Path) -> dict | None:
    path = Path(project_dir) / "manifest.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
    """Write one group (ply + json + optional indices.npy). Returns json path."""
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
        "fit": fit_summary,
        "software": {"detpos": detpos.__version__},
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
