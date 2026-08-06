"""Residual u–v map tests on synthetic planar patches."""

import numpy as np

from cloudet.pipeline import residual_uv_map
from cloudet.plane import Plane

SIGMA = 0.03


def _clean_plane_patch(n_pts: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = np.array([0.0, 0.0, 1.0])
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    uv = rng.uniform(-50, 50, size=(n_pts, 2))
    noise = rng.normal(0, SIGMA, size=n_pts)
    pts = uv[:, :1] * u + uv[:, 1:] * v + noise[:, None] * n
    return pts, Plane(n, 0.0)


def test_residual_uv_map():
    pts, plane = _clean_plane_patch()
    uv = residual_uv_map(pts, plane, bins=50)
    assert uv["mean"].shape == (50, 50)
    filled = uv["counts"] > 0
    assert np.nanmax(np.abs(uv["mean"][filled])) < 5 * SIGMA


def test_residual_uv_map_aligns_rotated_rectangle():
    """A 45°-rotated rectangle should become axis-aligned in u–v."""
    rng = np.random.default_rng(0)
    # Axis-aligned 80 x 40 mm patch in xy, then rotate 45° about z.
    xs = rng.uniform(-40, 40, size=8000)
    ys = rng.uniform(-20, 20, size=8000)
    pts0 = np.column_stack([xs, ys, rng.normal(0.0, 0.01, size=8000)])
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = pts0 @ R.T
    plane = Plane.from_array([0.0, 0.0, 1.0, 0.0])
    uv = residual_uv_map(pts, plane, bins=40, return_points=True)
    u_span = float(uv["u"].max() - uv["u"].min())
    v_span = float(uv["v"].max() - uv["v"].min())
    long_s, short_s = max(u_span, v_span), min(u_span, v_span)
    assert long_s > 70.0
    assert short_s < 50.0
    assert long_s / short_s > 1.5


def test_residual_uv_map_aligns_despite_density_bias():
    """Uneven sampling should not leave a rectangular face tilted."""
    rng = np.random.default_rng(1)
    # Dense cluster on one side biases PCA away from the true edges.
    n_edge, n_bulk = 6000, 2000
    xs = np.concatenate([
        rng.uniform(-40, -25, size=n_edge),
        rng.uniform(-40, 40, size=n_bulk),
    ])
    ys = np.concatenate([
        rng.uniform(-20, 20, size=n_edge),
        rng.uniform(-20, 20, size=n_bulk),
    ])
    pts0 = np.column_stack([xs, ys, rng.normal(0.0, 0.01, size=n_edge + n_bulk)])
    ang = np.deg2rad(33.0)
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = pts0 @ R.T
    plane = Plane.from_array([0.0, 0.0, 1.0, 0.0])
    uv = residual_uv_map(pts, plane, return_points=True)
    # Corners of the true rectangle in the aligned frame should sit near
    # the AABB corners (small empty corners ⇒ low tilt).
    u_span = float(uv["u"].max() - uv["u"].min())
    v_span = float(uv["v"].max() - uv["v"].min())
    long_s, short_s = max(u_span, v_span), min(u_span, v_span)
    assert long_s > 70.0
    assert short_s < 50.0
    # Occupancy of the AABB should be high if edges are axis-aligned.
    uu, vv = uv["u"], uv["v"]
    nu = ((uu - uu.min()) / max(u_span, 1e-9) * 19).astype(int).clip(0, 19)
    nv = ((vv - vv.min()) / max(v_span, 1e-9) * 19).astype(int).clip(0, 19)
    occ = np.zeros((20, 20), dtype=bool)
    occ[nu, nv] = True
    assert occ.mean() > 0.55
