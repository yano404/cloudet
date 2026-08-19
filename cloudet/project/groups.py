"""Group set loading.

A "group" is a plane-candidate point set extracted by the interactive
picker. On disk: ``groups/group_xxx.ply`` + ``group_xxx.json`` with a
top-level ``manifest.json``.

Downstream code only relies on this module's ``GroupInfo`` API.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cloudet.core.plyio import read_ply_xyz

__all__ = ["GroupInfo", "load_groups", "sha256_file"]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class GroupInfo:
    group_id: int
    name: str
    ply_path: Path
    num_points: int
    coarse_plane: np.ndarray | None  # [a, b, c, d] from the picker, may be None
    clicked: np.ndarray | None  # (3,) click position, may be None

    def load_points(self) -> np.ndarray:
        return read_ply_xyz(self.ply_path)

    def sha256(self) -> str:
        return sha256_file(self.ply_path)


def _load_groups(groups_dir: Path) -> list[GroupInfo]:
    groups = []
    for meta_path in sorted(groups_dir.glob("group_*.json")):
        with open(meta_path, encoding="utf-8") as f:
            entry = json.load(f)
        ply_path = groups_dir / entry["ply_file"]
        if not ply_path.exists():
            raise FileNotFoundError(f"missing point cloud for {meta_path.name}: {ply_path}")
        coarse = entry.get("coarse_plane")
        clicked = entry.get("clicked")
        groups.append(
            GroupInfo(
                group_id=int(entry["group_id"]),
                name=str(entry.get("name", f"G{entry['group_id']}")),
                ply_path=ply_path,
                num_points=int(entry["num_points"]),
                coarse_plane=np.asarray(coarse, dtype=np.float64) if coarse else None,
                clicked=np.asarray(clicked, dtype=np.float64) if clicked else None,
            )
        )
    return groups


def load_groups(path: str | Path) -> list[GroupInfo]:
    """Load a group set from ``path``.

    Accepts either a project directory (containing ``groups/``) or a
    groups directory with ``group_*.json`` sidecars.
    """
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")

    if (path / "groups").is_dir():
        path = path / "groups"

    if list(path.glob("group_*.json")):
        return _load_groups(path)
    raise FileNotFoundError(
        f"no group metadata found in {path} (expected group_*.json)"
    )
