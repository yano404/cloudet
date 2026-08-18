"""On-disk caches for the picker's VoxelHashGrid and display downsample.

Derived acceleration only — not part of the project reproducibility
contract. A miss or a corrupt file just rebuilds.

Large arrays are stored as individual ``.npy`` files (optionally memmap'd).
Do not use ``np.savez`` here: it zip-buffers every array in RAM and can
OOM a 60 M-point survey on save or load.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cloudet.neighbors import VoxelHashGrid

__all__ = [
    "CACHE_VERSION",
    "display_cache_path",
    "grid_cache_path",
    "load_display_xyz",
    "load_voxel_grid",
    "save_display_xyz",
    "save_voxel_grid",
    "source_fingerprint",
]

CACHE_VERSION = 3


def grid_cache_path(project_dir: str | Path, cell_size: float) -> Path:
    tag = f"{float(cell_size):.6g}"
    return Path(project_dir) / "cache" / f"voxel_grid_c{tag}"


def display_cache_path(
    project_dir: str | Path,
    *,
    voxel_size: float,
    max_points: int,
    backend: str,
    seed: int = 0,
) -> Path:
    vtag = f"{float(voxel_size):.6g}"
    name = f"display_v{vtag}_n{int(max_points)}_{str(backend).lower()}"
    if int(seed):
        name += f"_s{int(seed)}"
    return Path(project_dir) / "cache" / name


def source_fingerprint(
    source_path: str | Path,
    n_points: int,
    cell_size: float | None = None,
    **extras,
) -> dict:
    path = Path(source_path).expanduser().resolve()
    st = path.stat()
    out: dict = {
        "version": CACHE_VERSION,
        "source_path": str(path),
        "n_points": int(n_points),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    if cell_size is not None:
        out["cell_size"] = float(cell_size)
    for key, value in extras.items():
        if isinstance(value, (bool, np.bool_)):
            out[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[key] = float(value)
        else:
            out[key] = str(value)
    return out


def _canon_fp(doc: dict) -> dict:
    out = {}
    for key, value in doc.items():
        if isinstance(value, (bool, np.bool_)):
            out[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[key] = float(value)
        else:
            out[key] = str(value)
    return out


def _fingerprint_equal(a: dict, b: dict) -> bool:
    return _canon_fp(a) == _canon_fp(b)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".writing.npy")
    with open(tmp, "wb") as f:
        np.save(f, np.asanyarray(arr))
    tmp.replace(path)


def _atomic_write_json(path: Path, doc: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".writing.json")
    tmp.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_meta(directory: Path) -> dict | None:
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _remove_legacy_zips(project_dir: str | Path) -> None:
    """Drop old np.savez caches without loading them (they can be ~1 GB in RAM)."""
    cache = Path(project_dir) / "cache"
    if not cache.is_dir():
        return
    for p in cache.glob("*.npz"):
        try:
            p.unlink()
        except OSError:
            pass


def _display_fingerprint(
    source_path: str | Path,
    n_points: int,
    *,
    voxel_size: float,
    max_points: int,
    backend: str,
    seed: int,
) -> dict:
    return source_fingerprint(
        source_path,
        n_points,
        voxel_size=float(voxel_size),
        max_points=int(max_points),
        backend=str(backend).lower(),
        seed=int(seed),
    )


def load_voxel_grid(
    project_dir: str | Path,
    points: np.ndarray,
    source_path: str | Path,
    cell_size: float,
) -> VoxelHashGrid | None:
    """Return a cached grid if the fingerprint matches; otherwise ``None``."""
    _remove_legacy_zips(project_dir)
    try:
        want = source_fingerprint(source_path, len(points), cell_size)
        directory = grid_cache_path(project_dir, cell_size)
        got = _read_meta(directory)
        if got is None or not _fingerprint_equal(got, want):
            return None
        origin = np.load(directory / "origin.npy")
        dims = np.load(directory / "dims.npy")
        cell_keys = np.load(directory / "cell_keys.npy", mmap_mode="r")
        cell_starts = np.load(directory / "cell_starts.npy", mmap_mode="r")
        order = np.load(directory / "order.npy", mmap_mode="r")
        return VoxelHashGrid.from_arrays(
            points,
            cell_size=float(got["cell_size"]),
            origin=origin,
            dims=dims,
            cell_keys=cell_keys,
            cell_starts=cell_starts,
            order=order,
            validate_range=False,
        )
    except Exception:
        return None


def save_voxel_grid(
    project_dir: str | Path,
    grid: VoxelHashGrid,
    source_path: str | Path,
) -> Path | None:
    """Write ``grid`` next to the project. Returns the directory, or ``None`` on failure."""
    _remove_legacy_zips(project_dir)
    try:
        fp = source_fingerprint(source_path, len(grid.points), grid.cell_size)
        directory = grid_cache_path(project_dir, grid.cell_size)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = grid.index_arrays()
        _atomic_save_npy(directory / "origin.npy", arrays["origin"])
        _atomic_save_npy(directory / "dims.npy", arrays["dims"])
        _atomic_save_npy(directory / "cell_keys.npy", arrays["cell_keys"])
        _atomic_save_npy(directory / "cell_starts.npy", arrays["cell_starts"])
        _atomic_save_npy(directory / "order.npy", arrays["order"])
        _atomic_write_json(directory / "meta.json", fp)
        return directory
    except (OSError, ValueError, TypeError, KeyError):
        return None


def load_display_xyz(
    project_dir: str | Path,
    n_points: int,
    source_path: str | Path,
    *,
    voxel_size: float,
    max_points: int,
    backend: str,
    seed: int = 0,
) -> np.ndarray | None:
    """Return cached display XYZ if the fingerprint matches; otherwise ``None``."""
    _remove_legacy_zips(project_dir)
    try:
        want = _display_fingerprint(
            source_path,
            n_points,
            voxel_size=voxel_size,
            max_points=max_points,
            backend=backend,
            seed=seed,
        )
        directory = display_cache_path(
            project_dir,
            voxel_size=voxel_size,
            max_points=max_points,
            backend=backend,
            seed=seed,
        )
        got = _read_meta(directory)
        if got is None or not _fingerprint_equal(got, want):
            return None
        xyz = np.load(directory / "xyz.npy")
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            return None
        return xyz
    except Exception:
        return None


def save_display_xyz(
    project_dir: str | Path,
    xyz: np.ndarray,
    source_path: str | Path,
    n_points: int,
    *,
    voxel_size: float,
    max_points: int,
    backend: str,
    seed: int = 0,
) -> Path | None:
    """Write display XYZ next to the project. Returns the directory, or ``None`` on failure."""
    _remove_legacy_zips(project_dir)
    try:
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (M, 3)")
        fp = _display_fingerprint(
            source_path,
            n_points,
            voxel_size=voxel_size,
            max_points=max_points,
            backend=backend,
            seed=seed,
        )
        directory = display_cache_path(
            project_dir,
            voxel_size=voxel_size,
            max_points=max_points,
            backend=backend,
            seed=seed,
        )
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_save_npy(directory / "xyz.npy", xyz)
        _atomic_write_json(directory / "meta.json", fp)
        return directory
    except (OSError, ValueError, TypeError, KeyError):
        return None
