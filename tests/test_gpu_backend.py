"""CuPy GPU backend parity and resolution tests."""

import numpy as np
import pytest

from cloudet.array_backend import (
    GPU_MIN_POINTS,
    cupy_available,
    cupy_unavailable_reason,
    get_context,
    resolve_compute_backend,
)
from cloudet.neighbors import display_indices, resolve_display_backend, voxel_downsample_indices
from cloudet.pipeline import residual_uv_map
from cloudet.plane import Plane, robust_fit_plane, run_ransac

try:
    import cupy as cp

    HAS_CUPY = bool(cp.cuda.is_available())
except ImportError:
    HAS_CUPY = False


SIGMA = 0.03


def _synthetic_plane(n_pts: int = 80_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = np.array([0.0, 0.0, 1.0])
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    uv = rng.uniform(-50, 50, size=(n_pts, 2))
    noise = rng.normal(0, SIGMA, size=n_pts)
    pts = uv[:, :1] * u + uv[:, 1:] * v + noise[:, None] * n
    ghost = pts[:2000] + 0.5 * n
    pts = np.vstack([pts, ghost])
    return pts, Plane(n, 0.0)


def test_resolve_compute_backend_numpy():
    assert resolve_compute_backend("numpy") == "numpy"
    assert resolve_compute_backend("numpy", n_points=1_000_000) == "numpy"


def test_auto_skips_gpu_for_small_clouds():
    if not HAS_CUPY:
        pytest.skip("cupy not available")
    assert resolve_compute_backend("auto", n_points=GPU_MIN_POINTS - 1) == "numpy"
    assert resolve_compute_backend("auto", n_points=GPU_MIN_POINTS) == "cupy"


def test_resolve_display_backend_includes_cupy_when_available():
    resolved = resolve_display_backend("auto")
    if HAS_CUPY:
        assert resolved == "cupy"
    else:
        assert resolved in ("numpy", "open3d")


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_robust_fit_plane_cpu_gpu_parity():
    pts, true = _synthetic_plane()
    init = true
    cpu = robust_fit_plane(
        pts, threshold=0.2, init=init, compute_backend="numpy"
    )
    gpu = robust_fit_plane(
        pts, threshold=0.2, init=init, compute_backend="cupy"
    )
    assert cpu.plane.angle_to(gpu.plane) < 1e-4
    assert cpu.n_inliers == gpu.n_inliers
    assert cpu.converged == gpu.converged


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_residual_uv_map_cpu_gpu_parity():
    pts, plane = _synthetic_plane(n_pts=60_000)
    cpu = residual_uv_map(pts, plane, bins=50, compute_backend="numpy")
    gpu = residual_uv_map(pts, plane, bins=50, compute_backend="cupy")
    filled = cpu["counts"] > 0
    assert np.nanmax(np.abs(cpu["mean"][filled] - gpu["mean"][filled])) < 5 * SIGMA
    assert cpu["n_used"] == gpu["n_used"]


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_voxel_downsample_indices_cpu_gpu_parity():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 100, size=(120_000, 3))
    cpu = voxel_downsample_indices(pts, voxel_size=2.0, compute_backend="numpy")
    gpu = voxel_downsample_indices(pts, voxel_size=2.0, compute_backend="cupy")
    assert len(cpu) == len(gpu)


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_display_indices_cupy_backend():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 50, size=(100_000, 3))
    idx = display_indices(
        pts, voxel_size=1.0, max_points=20_000, compute_backend="cupy"
    )
    assert 0 < len(idx) <= 20_000


def test_forced_cupy_uses_gpu_on_small_clouds():
    if not HAS_CUPY:
        pytest.skip("cupy not available")
    assert resolve_compute_backend("cupy", n_points=1000) == "cupy"


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_ransac_cpu_gpu_parity():
    pts, true = _synthetic_plane(n_pts=60_000)
    kw = dict(threshold=0.2, n_iterations=200, seed=5, compute_backend="numpy")
    p_cpu, m_cpu = run_ransac(pts, **kw)
    p_gpu, m_gpu = run_ransac(pts, **kw, compute_backend="cupy")
    assert p_cpu.angle_to(p_gpu) < 1e-3
    assert m_cpu.sum() == m_gpu.sum()


def test_forced_cupy_without_install_raises():
    if cupy_available():
        pytest.skip("cupy usable")
    with pytest.raises(ImportError, match="cupy"):
        get_context("cupy", n_points=100_000)


def test_cupy_unavailable_reason_when_not_usable():
    if cupy_available():
        pytest.skip("cupy usable")
    reason = cupy_unavailable_reason()
    assert reason
