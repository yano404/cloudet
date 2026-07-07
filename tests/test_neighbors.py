import numpy as np
import pytest

from detpos.neighbors import VoxelHashGrid, display_indices, voxel_downsample_indices


def test_radius_matches_bruteforce():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-100, 100, size=(50_000, 3))
    grid = VoxelHashGrid(pts, cell_size=10.0)
    for _ in range(20):
        center = rng.uniform(-100, 100, size=3)
        radius = rng.uniform(1.0, 25.0)
        got = np.sort(grid.radius_indices(center, radius))
        want = np.flatnonzero(np.linalg.norm(pts - center, axis=1) <= radius)
        assert np.array_equal(got, want)


def test_radius_outside_bbox():
    pts = np.zeros((10, 3))
    grid = VoxelHashGrid(pts, cell_size=1.0)
    assert len(grid.radius_indices([100, 100, 100], 5.0)) == 0
    assert len(grid.radius_indices([100, 100, 100], 200.0)) == 10


def test_voxel_downsample():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 10, size=(10_000, 3))
    idx = voxel_downsample_indices(pts, 1.0)
    # at most 10^3 voxels, one point each
    assert len(idx) <= 1000
    # every voxel that has points is represented
    ijk_all = np.floor((pts - pts.min(axis=0)) / 1.0).astype(np.int64)
    ijk_kept = ijk_all[idx]
    assert len(np.unique(ijk_all, axis=0)) == len(np.unique(ijk_kept, axis=0))
    # voxel_size=0 -> identity
    assert np.array_equal(voxel_downsample_indices(pts, 0.0), np.arange(len(pts)))


def test_display_indices_cap():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 1000, size=(200_000, 3))
    idx = display_indices(pts, voxel_size=0.0, max_points=50_000)
    assert len(idx) == 50_000
    assert len(np.unique(idx)) == 50_000
    # deterministic
    idx2 = display_indices(pts, voxel_size=0.0, max_points=50_000)
    assert np.array_equal(idx, idx2)


def test_grid_performance_smoke():
    """Build + queries stay fast on a million points."""
    import time

    rng = np.random.default_rng(3)
    pts = rng.uniform(-500, 500, size=(1_000_000, 3))
    t0 = time.time()
    grid = VoxelHashGrid(pts, cell_size=10.0)
    build = time.time() - t0
    t0 = time.time()
    for _ in range(50):
        grid.radius_indices(rng.uniform(-500, 500, 3), 10.0)
    q = (time.time() - t0) / 50
    assert build < 5.0
    assert q < 0.05
