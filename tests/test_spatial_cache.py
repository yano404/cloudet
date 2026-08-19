"""Disk cache for VoxelHashGrid (fingerprint hit / miss)."""

import os
from pathlib import Path

import numpy as np

from cloudet.core.plyio import write_ply_xyz
from cloudet.project.spatial_cache import (
    display_cache_path,
    grid_cache_path,
    load_display_xyz,
    load_voxel_grid,
    save_display_xyz,
    save_voxel_grid,
    source_fingerprint,
)
from cloudet.core.neighbors import VoxelHashGrid, display_xyz


def _write_cloud(tmp_path: Path, pts: np.ndarray) -> Path:
    path = tmp_path / "cloud.ply"
    write_ply_xyz(path, pts)
    return path


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-20, 20, size=(3_000, 3))
    ply = _write_cloud(tmp_path, pts)
    project = tmp_path / "proj"
    grid = VoxelHashGrid(pts, cell_size=10.0)
    saved = save_voxel_grid(project, grid, ply)
    assert saved is not None
    assert saved == grid_cache_path(project, 10.0)

    loaded = load_voxel_grid(project, pts, ply, 10.0)
    assert loaded is not None
    center = np.array([0.0, 0.0, 0.0])
    assert np.array_equal(
        np.sort(loaded.radius_indices(center, 7.0)),
        np.sort(grid.radius_indices(center, 7.0)),
    )


def test_fingerprint_n_points_mismatch(tmp_path):
    rng = np.random.default_rng(1)
    pts = rng.uniform(-10, 10, size=(1_000, 3))
    ply = _write_cloud(tmp_path, pts)
    project = tmp_path / "proj"
    save_voxel_grid(project, VoxelHashGrid(pts, cell_size=5.0), ply)
    other = pts[:-10]
    assert load_voxel_grid(project, other, ply, 5.0) is None


def test_fingerprint_cell_size_mismatch(tmp_path):
    rng = np.random.default_rng(2)
    pts = rng.uniform(-10, 10, size=(800, 3))
    ply = _write_cloud(tmp_path, pts)
    project = tmp_path / "proj"
    save_voxel_grid(project, VoxelHashGrid(pts, cell_size=10.0), ply)
    assert load_voxel_grid(project, pts, ply, 4.0) is None
    assert load_voxel_grid(project, pts, ply, 10.0) is not None


def test_fingerprint_mtime_mismatch(tmp_path):
    rng = np.random.default_rng(3)
    pts = rng.uniform(-10, 10, size=(500, 3))
    ply = _write_cloud(tmp_path, pts)
    project = tmp_path / "proj"
    save_voxel_grid(project, VoxelHashGrid(pts, cell_size=8.0), ply)
    st = ply.stat()
    os.utime(ply, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert load_voxel_grid(project, pts, ply, 8.0) is None


def test_corrupt_cache_returns_none(tmp_path):
    rng = np.random.default_rng(5)
    pts = rng.uniform(-5, 5, size=(200, 3))
    ply = _write_cloud(tmp_path, pts)
    project = tmp_path / "proj"
    path = grid_cache_path(project, 10.0)
    path.mkdir(parents=True)
    (path / "meta.json").write_text("{not json", encoding="utf-8")
    assert load_voxel_grid(project, pts, ply, 10.0) is None


def test_missing_cache_returns_none(tmp_path):
    rng = np.random.default_rng(4)
    pts = rng.uniform(-5, 5, size=(100, 3))
    ply = _write_cloud(tmp_path, pts)
    assert load_voxel_grid(tmp_path / "proj", pts, ply, 10.0) is None


def test_source_fingerprint_tracks_file(tmp_path):
    pts = np.zeros((10, 3))
    ply = _write_cloud(tmp_path, pts)
    fp = source_fingerprint(ply, 10, 10.0)
    assert fp["n_points"] == 10
    assert fp["cell_size"] == 10.0
    assert fp["size_bytes"] == ply.stat().st_size
    assert Path(fp["source_path"]) == ply.resolve()


def test_display_xyz_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    pts = rng.uniform(-30, 30, size=(5_000, 3))
    ply = _write_cloud(tmp_path, pts)
    project = tmp_path / "proj"
    kwargs = dict(voxel_size=2.0, max_points=800, backend="numpy")
    xyz = display_xyz(pts, **kwargs)
    saved = save_display_xyz(project, xyz, ply, len(pts), **kwargs)
    assert saved is not None
    assert saved == display_cache_path(project, **kwargs)
    loaded = load_display_xyz(project, len(pts), ply, **kwargs)
    assert loaded is not None
    assert np.array_equal(loaded, xyz)


def test_display_xyz_settings_mismatch(tmp_path):
    rng = np.random.default_rng(7)
    pts = rng.uniform(-10, 10, size=(1_200, 3))
    ply = _write_cloud(tmp_path, pts)
    project = tmp_path / "proj"
    xyz = display_xyz(pts, voxel_size=1.0, max_points=400, backend="numpy")
    save_display_xyz(
        project, xyz, ply, len(pts), voxel_size=1.0, max_points=400, backend="numpy"
    )
    assert load_display_xyz(
        project, len(pts), ply, voxel_size=2.0, max_points=400, backend="numpy"
    ) is None
    assert load_display_xyz(
        project, len(pts), ply, voxel_size=1.0, max_points=200, backend="numpy"
    ) is None
    assert load_display_xyz(
        project, len(pts) - 10, ply, voxel_size=1.0, max_points=400, backend="numpy"
    ) is None
    assert load_display_xyz(
        project, len(pts), ply, voxel_size=1.0, max_points=400, backend="numpy"
    ) is not None
    assert load_display_xyz(
        project, len(pts), ply, voxel_size=1.0, max_points=400, backend="open3d"
    ) is None


def test_legacy_npz_is_not_loaded(tmp_path):
    rng = np.random.default_rng(8)
    pts = rng.uniform(-5, 5, size=(100, 3))
    ply = _write_cloud(tmp_path, pts)
    project = tmp_path / "proj"
    cache = project / "cache"
    cache.mkdir(parents=True)
    bomb = cache / "voxel_grid_c10.npz"
    bomb.write_bytes(b"PK" + b"\x00" * 100)
    assert load_voxel_grid(project, pts, ply, 10.0) is None
    assert not bomb.exists()

