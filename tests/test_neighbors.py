import numpy as np
import pytest

from cloudet.core.neighbors import VoxelHashGrid, display_indices, voxel_downsample_indices


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


def test_large_radius_slabbed_matches_bruteforce():
    """Query AABB larger than _MAX_QUERY_CELLS must not full-cloud allocate."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-50, 50, size=(80_000, 3)).astype(np.float32)
    grid = VoxelHashGrid(pts, cell_size=2.0)
    assert grid.points.dtype == np.float32
    center = np.zeros(3)
    radius = 40.0  # (40 cells)^3 ≫ 8000
    got = np.sort(grid.radius_indices(center, radius))
    # Chunked reference (same as new bruteforce helper).
    want = np.sort(grid._bruteforce_radius_indices(center, radius))
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


def test_depth_layers_along_ray_separates_overlapping_surfaces():
    from cloudet.core.neighbors import depth_layers_along_ray

    rng = np.random.default_rng(0)
    # Three parallel planes along z, viewed from -z.
    front = np.column_stack([
        rng.uniform(-20, 20, 2_000),
        rng.uniform(-20, 20, 2_000),
        rng.normal(100.0, 0.2, 2_000),
    ])
    mid = np.column_stack([
        rng.uniform(-20, 20, 2_000),
        rng.uniform(-20, 20, 2_000),
        rng.normal(250.0, 0.2, 2_000),
    ])
    back = np.column_stack([
        rng.uniform(-20, 20, 2_000),
        rng.uniform(-20, 20, 2_000),
        rng.normal(400.0, 0.2, 2_000),
    ])
    pts = np.vstack([front, mid, back])
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    layers = depth_layers_along_ray(
        pts, origin, direction, cylinder_radius_mm=5.0, gap_mm=20.0, min_points=50
    )
    assert len(layers) == 3
    depths = [L["depth_mm"] for L in layers]
    assert depths == sorted(depths)
    assert abs(depths[0] - 100) < 5
    assert abs(depths[1] - 250) < 5
    assert abs(depths[2] - 400) < 5
    # Seeds sit on each plane
    assert abs(layers[0]["seed"][2] - 100) < 2
    assert abs(layers[2]["seed"][2] - 400) < 2


def test_front_layer_wins_even_when_sparser_than_the_back():
    """A click must land on the visible surface, not the denser one behind it.

    Point clouds render as isolated dots, so the VTK hit often slips through
    the gaps onto a back face; layer 0 must still be the near surface.
    """
    from cloudet.core.neighbors import depth_layers_along_ray

    rng = np.random.default_rng(1)
    sparse_front = np.column_stack([
        rng.uniform(-10, 10, 120),
        rng.uniform(-10, 10, 120),
        rng.normal(50.0, 0.1, 120),
    ])
    dense_back = np.column_stack([
        rng.uniform(-10, 10, 20_000),
        rng.uniform(-10, 10, 20_000),
        rng.normal(600.0, 0.1, 20_000),
    ])
    pts = np.vstack([sparse_front, dense_back])
    layers = depth_layers_along_ray(
        pts,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        cylinder_radius_mm=6.0,
        gap_mm=20.0,
        min_points=10,
    )
    assert len(layers) == 2
    assert abs(layers[0]["seed"][2] - 50) < 2
    assert layers[0]["n_points"] < layers[1]["n_points"]


def test_points_behind_the_camera_are_ignored():
    from cloudet.core.neighbors import depth_layers_along_ray

    rng = np.random.default_rng(2)
    behind = np.column_stack([
        rng.uniform(-5, 5, 500),
        rng.uniform(-5, 5, 500),
        rng.normal(-200.0, 0.1, 500),
    ])
    ahead = np.column_stack([
        rng.uniform(-5, 5, 500),
        rng.uniform(-5, 5, 500),
        rng.normal(300.0, 0.1, 500),
    ])
    layers = depth_layers_along_ray(
        np.vstack([behind, ahead]),
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        cylinder_radius_mm=6.0,
        gap_mm=20.0,
        min_points=10,
    )
    assert len(layers) == 1
    assert abs(layers[0]["depth_mm"] - 300) < 5


def _oblique_surface(t_start, t_end, step, rng):
    """One flat surface seen so obliquely that decimation spaces it along t."""
    t = np.arange(t_start, t_end, step)
    return np.column_stack([
        rng.normal(0.0, 0.3, len(t)),
        rng.normal(0.0, 0.3, len(t)),
        t,
    ])


def test_adaptive_gap_keeps_one_oblique_surface_whole():
    """A fixed gap shatters a coarsely sampled surface into phantom layers."""
    from cloudet.core.neighbors import depth_layers_along_ray

    rng = np.random.default_rng(7)
    pts = _oblique_surface(100.0, 400.0, 12.0, rng)
    args = (pts, np.zeros(3), np.array([0.0, 0.0, 1.0]))

    shattered = depth_layers_along_ray(
        *args, cylinder_radius_mm=6.0, gap_mm=8.0, min_points=1
    )
    assert len(shattered) == len(pts)

    whole = depth_layers_along_ray(*args, cylinder_radius_mm=6.0, min_points=6)
    assert len(whole) == 1
    assert whole[0]["n_points"] == len(pts)


def test_adaptive_gap_still_splits_genuinely_separated_surfaces():
    from cloudet.core.neighbors import depth_layers_along_ray

    rng = np.random.default_rng(8)
    pts = np.vstack([
        _oblique_surface(100.0, 250.0, 12.0, rng),
        _oblique_surface(900.0, 1050.0, 12.0, rng),
    ])
    layers = depth_layers_along_ray(
        pts,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        cylinder_radius_mm=6.0,
        min_points=6,
    )
    assert len(layers) == 2
    assert layers[1]["depth_mm"] - layers[0]["depth_mm"] > 500


def test_display_xyz_numpy_matches_indices():
    from cloudet.core.neighbors import display_xyz

    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 50, size=(20_000, 3))
    xyz = display_xyz(pts, voxel_size=1.0, max_points=5_000, backend="numpy")
    idx = display_indices(pts, voxel_size=1.0, max_points=5_000)
    assert xyz.shape == (len(idx), 3)
    assert np.allclose(xyz, pts[idx])


def test_resolve_display_backend_auto():
    from cloudet.core.neighbors import resolve_display_backend

    resolved = resolve_display_backend("auto")
    assert resolved in ("numpy", "open3d", "cupy")
    assert resolve_display_backend("numpy") == "numpy"


def test_display_xyz_open3d():
    o3d = pytest.importorskip("open3d")
    from cloudet.core.neighbors import display_xyz, resolve_display_backend

    assert resolve_display_backend("open3d") == "open3d"
    rng = np.random.default_rng(5)
    pts = rng.uniform(0, 20, size=(30_000, 3))
    xyz = display_xyz(pts, voxel_size=1.0, max_points=10_000, backend="open3d")
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert 0 < len(xyz) <= 10_000


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


def test_radius_large_query_uses_slabbed_cells():
    """Fine grids + large radius must not meshgrid millions of cells."""
    import time

    rng = np.random.default_rng(4)
    pts = rng.uniform(-500, 500, size=(500_000, 3))
    grid = VoxelHashGrid(pts, cell_size=1.0)
    t0 = time.perf_counter()
    got = grid.radius_indices([0.0, 0.0, 0.0], 150.0)
    elapsed = time.perf_counter() - t0
    want = np.flatnonzero(np.linalg.norm(pts, axis=1) <= 150.0)
    assert np.array_equal(np.sort(got), np.sort(want))
    assert elapsed < 5.0


def test_from_arrays_roundtrip_queries():
    rng = np.random.default_rng(5)
    pts = rng.uniform(-50, 50, size=(8_000, 3))
    grid = VoxelHashGrid(pts, cell_size=5.0)
    restored = VoxelHashGrid.from_arrays(pts, **grid.index_arrays())
    assert restored.cell_size == grid.cell_size
    assert np.array_equal(restored._order, grid._order)
    center = np.array([1.0, -2.0, 3.0])
    assert np.array_equal(
        np.sort(restored.radius_indices(center, 8.0)),
        np.sort(grid.radius_indices(center, 8.0)),
    )


def test_from_arrays_rejects_stale_origin():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    grid = VoxelHashGrid(pts, cell_size=1.0)
    arrays = grid.index_arrays()
    arrays["origin"] = arrays["origin"] + 10.0
    with pytest.raises(ValueError, match="origin"):
        VoxelHashGrid.from_arrays(pts, **arrays)
