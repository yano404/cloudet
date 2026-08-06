"""Synthetic-data validation of the plane fitting core.

The tests emulate FARO Quantum-S conditions: coordinates in mm,
point noise sigma = 0.03 mm, patch sizes of order 100 mm.
"""

import numpy as np
import pytest

from cloudet.plane import (
    Plane,
    fit_plane_lsq,
    mad_sigma,
    ransac_plane,
    residual_stats,
    robust_fit_plane,
)

SIGMA = 0.03  # mm, nominal FARO Quantum-S noise


def make_plane_points(
    rng,
    n=200_000,
    extent=100.0,
    sigma=SIGMA,
    normal=(0.3, -0.5, 0.8),
    offset=250.0,
):
    """Points on plane n.x + d = 0 with gaussian perpendicular noise."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    # orthonormal in-plane basis
    a = np.array([1.0, 0.0, 0.0])
    if abs(normal @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, a)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    center = -offset * normal  # point on plane (d = offset)
    uv = rng.uniform(-extent / 2, extent / 2, size=(n, 2))
    noise = rng.normal(0.0, sigma, size=n)
    pts = center + uv[:, :1] * u + uv[:, 1:] * v + noise[:, None] * normal
    return pts, Plane(normal, offset)


def test_fit_exact_plane():
    rng = np.random.default_rng(1)
    pts, true_plane = make_plane_points(rng, n=1000, sigma=0.0)
    fit = fit_plane_lsq(pts)
    assert fit.angle_to(true_plane) < 1e-12
    assert abs(fit.d - true_plane.d) < 1e-9
    assert np.max(fit.distances(pts)) < 1e-9


def test_fit_gaussian_noise():
    rng = np.random.default_rng(2)
    pts, true_plane = make_plane_points(rng)
    fit = fit_plane_lsq(pts)
    # statistical limits: angle ~ sigma/(extent/2 * sqrt(N)) ~ 1.3e-6 rad,
    # offset ~ sigma/sqrt(N) ~ 6.7e-5 mm; allow 10x margin
    assert fit.angle_to(true_plane) < 2e-5
    assert abs(fit.d - true_plane.d) < 7e-4
    stats = residual_stats(fit.signed_distances(pts))
    assert stats["std"] == pytest.approx(SIGMA, rel=0.02)
    assert stats["mad_sigma"] == pytest.approx(SIGMA, rel=0.03)


def test_mad_sigma_truncation_robust():
    """MAD sigma stays close to true sigma even after +-2 sigma truncation,
    where plain std underestimates badly."""
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, SIGMA, size=500_000)
    r_trunc = r[np.abs(r) <= 2 * SIGMA]
    assert np.std(r_trunc) < 0.90 * SIGMA  # plain std is biased
    assert mad_sigma(r_trunc) == pytest.approx(SIGMA, rel=0.10)


def test_robust_fit_symmetric_outliers():
    rng = np.random.default_rng(4)
    pts, true_plane = make_plane_points(rng, n=100_000)
    # 20% uniform slab +-1 mm (like the picker's accumulate window)
    n_out = 20_000
    out_uv = rng.uniform(-50, 50, size=(n_out, 2))
    out_h = rng.uniform(-1.0, 1.0, size=n_out)
    normal = true_plane.normal
    a = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, a); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    center = -true_plane.d * normal
    outliers = center + out_uv[:, :1] * u + out_uv[:, 1:] * v + out_h[:, None] * normal
    all_pts = np.vstack([pts, outliers])

    fit = robust_fit_plane(all_pts, threshold=0.1)
    assert fit.converged
    assert fit.plane.angle_to(true_plane) < 5e-5
    assert abs(fit.plane.d - true_plane.d) < 2e-3


def test_robust_fit_asymmetric_outliers():
    """One-sided contamination (e.g. a second surface 0.5 mm away) must
    not pull the plane: this is where plain LSQ fails."""
    rng = np.random.default_rng(5)
    pts, true_plane = make_plane_points(rng, n=100_000)
    ghost, _ = make_plane_points(rng, n=15_000, offset=250.0 + 0.5)
    all_pts = np.vstack([pts, ghost])

    naive = fit_plane_lsq(all_pts)
    assert abs(naive.d - true_plane.d) > 0.02  # visibly biased

    fit = robust_fit_plane(all_pts, threshold=0.1)
    assert abs(fit.plane.d - true_plane.d) < 3e-3
    assert fit.plane.angle_to(true_plane) < 1e-4


def test_robust_fit_adaptive_threshold():
    rng = np.random.default_rng(6)
    pts, true_plane = make_plane_points(rng, n=100_000)
    ghost, _ = make_plane_points(rng, n=15_000, offset=250.0 + 0.5)
    fit = robust_fit_plane(np.vstack([pts, ghost]), threshold=None, sigma_factor=3.0)
    assert abs(fit.plane.d - true_plane.d) < 3e-3
    # adaptive threshold should settle near 3 sigma
    assert fit.threshold == pytest.approx(3 * SIGMA, rel=0.25)


def test_ransac_selector_and_reproducibility():
    rng = np.random.default_rng(7)
    pts, true_plane = make_plane_points(rng, n=50_000)
    ghost, _ = make_plane_points(rng, n=30_000, offset=250.0 + 5.0)
    all_pts = np.vstack([pts, ghost])

    plane1, mask1 = ransac_plane(all_pts, threshold=0.1, n_iterations=500, seed=42)
    plane2, mask2 = ransac_plane(all_pts, threshold=0.1, n_iterations=500, seed=42)
    assert np.array_equal(mask1, mask2)  # reproducible
    assert plane1.angle_to(plane2) == 0.0

    # RANSAC must select the dominant plane, not the ghost
    assert np.count_nonzero(mask1[:50_000]) > 45_000
    assert np.count_nonzero(mask1[50_000:]) < 1_000

    # full pipeline: RANSAC init + robust refit
    fit = robust_fit_plane(all_pts, threshold=0.1, init=plane1)
    assert abs(fit.plane.d - true_plane.d) < 3e-3


def test_plane_sign_convention():
    p1 = Plane(np.array([0.0, 0.0, 1.0]), 5.0)
    p2 = Plane(np.array([0.0, 0.0, -1.0]), -5.0)
    assert np.allclose(p1.as_array(), p2.as_array())


def test_degenerate_inputs():
    with pytest.raises(ValueError):
        fit_plane_lsq(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        Plane(np.zeros(3), 0.0)
