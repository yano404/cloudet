"""RANSAC backend dispatch and numpy/open3d cross-validation.

The cross-validation tests run only where open3d is installed (e.g. on
the analysis machine); run ``pytest tests/test_backend.py -v`` there to
demonstrate that both backends feed the identical final estimator and
agree at the micrometre level.
"""

import numpy as np
import pytest

from cloudet.core.plane import normalize_ransac_backend, robust_fit_plane, run_ransac

try:
    import open3d  # noqa: F401

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

SIGMA = 0.03


def scene(rng, n_plane=50_000, n_out=20_000):
    pts = np.column_stack([
        rng.uniform(-50, 50, n_plane),
        rng.uniform(-50, 50, n_plane),
        rng.normal(0, SIGMA, n_plane),
    ])
    outliers = rng.uniform(-50, 50, size=(n_out, 3))
    outliers[:, 2] = rng.uniform(1.0, 30.0, n_out)
    return np.vstack([pts, outliers])


def test_normalize_ransac_backend_aliases():
    assert normalize_ransac_backend("numpy") == "seeded"
    assert normalize_ransac_backend("seeded") == "seeded"
    assert normalize_ransac_backend("seeded_cpu") == "seeded_cpu"
    assert normalize_ransac_backend("open3d") == "open3d"
    assert normalize_ransac_backend(None) == "seeded"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown RANSAC backend"):
        run_ransac(np.zeros((10, 3)), 0.1, backend="pcl")


def test_seeded_cpu_and_legacy_numpy_run():
    rng = np.random.default_rng(0)
    pts = scene(rng, n_plane=5_000, n_out=500)
    p_cpu, m_cpu = run_ransac(
        pts, threshold=0.1, n_iterations=200, seed=0, backend="seeded_cpu"
    )
    p_legacy, m_legacy = run_ransac(
        pts,
        threshold=0.1,
        n_iterations=200,
        seed=0,
        backend="numpy",
        compute_backend="numpy",
    )
    assert np.count_nonzero(m_cpu[:5_000]) > 4_000
    assert np.isfinite(p_cpu.d)
    assert np.count_nonzero(m_legacy[:5_000]) > 4_000
    assert np.isfinite(p_legacy.d)


def test_open3d_backend_missing_message():
    if HAS_OPEN3D:
        pytest.skip("open3d installed")
    with pytest.raises(ImportError, match="open3d"):
        run_ransac(np.random.default_rng(0).normal(size=(100, 3)), 0.1, backend="open3d")


@pytest.mark.skipif(not HAS_OPEN3D, reason="open3d not installed")
def test_backends_agree_after_refit():
    """Both backends must give the same final plane within micrometres,
    because the final estimator (robust orthogonal LSQ) is shared."""
    rng = np.random.default_rng(0)
    pts = scene(rng)

    fits = {}
    for backend in ("seeded_cpu", "open3d"):
        seed_plane, _ = run_ransac(
            pts, threshold=0.1, n_iterations=1000, seed=0, backend=backend
        )
        fits[backend] = robust_fit_plane(pts, threshold=0.1, init=seed_plane)

    a, b = fits["seeded_cpu"].plane, fits["open3d"].plane
    assert a.angle_to(b) < 1e-5           # < 10 urad
    assert abs(a.d - b.d) < 1e-3           # < 1 um
    n_a, n_b = fits["seeded_cpu"].n_inliers, fits["open3d"].n_inliers
    assert abs(n_a - n_b) < 0.01 * max(n_a, n_b)


@pytest.mark.skipif(not HAS_OPEN3D, reason="open3d not installed")
def test_open3d_backend_selects_plane():
    rng = np.random.default_rng(1)
    pts = scene(rng)
    plane, mask = run_ransac(
        pts, threshold=0.1, n_iterations=1000, seed=0, backend="open3d"
    )
    assert np.count_nonzero(mask[:50_000]) > 45_000
    assert np.count_nonzero(mask[50_000:]) < 500
