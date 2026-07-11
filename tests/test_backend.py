"""RANSAC backend dispatch and numpy/open3d cross-validation.

The cross-validation tests run only where open3d is installed (e.g. on
the analysis machine); run ``pytest tests/test_backend.py -v`` there to
demonstrate that both backends feed the identical final estimator and
agree at the micrometre level.
"""

import numpy as np
import pytest

from detpos.plane import robust_fit_plane, run_ransac

open3d = pytest.importorskip if False else None  # noqa: F811 (see _has_open3d)

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


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown RANSAC backend"):
        run_ransac(np.zeros((10, 3)), 0.1, backend="pcl")


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
    for backend in ("numpy", "open3d"):
        seed_plane, _ = run_ransac(pts, threshold=0.1, n_iterations=1000,
                                   seed=0, backend=backend)
        fits[backend] = robust_fit_plane(pts, threshold=0.1, init=seed_plane)

    a, b = fits["numpy"].plane, fits["open3d"].plane
    assert a.angle_to(b) < 1e-5           # < 10 urad
    assert abs(a.d - b.d) < 1e-3           # < 1 um
    n_a, n_b = fits["numpy"].n_inliers, fits["open3d"].n_inliers
    assert abs(n_a - n_b) < 0.01 * max(n_a, n_b)


@pytest.mark.skipif(not HAS_OPEN3D, reason="open3d not installed")
def test_open3d_backend_selects_plane():
    rng = np.random.default_rng(1)
    pts = scene(rng)
    plane, mask = run_ransac(pts, threshold=0.1, n_iterations=1000,
                             seed=0, backend="open3d")
    assert np.count_nonzero(mask[:50_000]) > 45_000
    assert np.count_nonzero(mask[50_000:]) < 500
