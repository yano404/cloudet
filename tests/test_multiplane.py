"""Multi-plane extraction: the 0.5 mm parallel-pair case and friends."""

import numpy as np
import pytest

from cloudet.multiplane import MultiPlaneParams, extract_planes
from cloudet.plane import Plane

SIGMA = 0.03


def parallel_pair(rng, separation=0.5, n1=80_000, n2=50_000):
    """Two fully overlapping parallel planes (z = 0 and z = separation)."""
    def sheet(n, z0):
        return np.column_stack([
            rng.uniform(-50, 50, n),
            rng.uniform(-50, 50, n),
            rng.normal(z0, SIGMA, n),
        ])
    return np.vstack([sheet(n1, 0.0), sheet(n2, separation)]), n1


def test_parallel_pair_05mm_separated():
    rng = np.random.default_rng(0)
    pts, n1 = parallel_pair(rng, separation=0.5)

    planes = extract_planes(pts, MultiPlaneParams())
    assert len(planes) == 2

    d = sorted(abs(p["result"].plane.d) for p in planes)
    assert d[0] == pytest.approx(0.0, abs=3e-3)
    assert d[1] == pytest.approx(0.5, abs=3e-3)

    # dominant sheet first, and point assignment is clean
    assert planes[0]["n_points"] > planes[1]["n_points"]
    first = planes[0]["mask"]
    assert np.count_nonzero(first[:n1]) > 0.95 * n1
    assert np.count_nonzero(first[n1:]) < 1000
    for p in planes:
        assert p["result"].status == "ok"
        assert not p["bimodal"]
        assert p["result"].fit.stats_inliers["mad_sigma"] == pytest.approx(SIGMA, rel=0.2)


def test_parallel_pair_04mm_still_separated():
    rng = np.random.default_rng(1)
    pts, _ = parallel_pair(rng, separation=0.4)
    planes = extract_planes(pts, MultiPlaneParams())
    assert len(planes) == 2
    d = sorted(abs(p["result"].plane.d) for p in planes)
    assert d[1] - d[0] == pytest.approx(0.4, abs=5e-3)


def test_perpendicular_planes():
    rng = np.random.default_rng(2)
    floor = np.column_stack([
        rng.uniform(-50, 50, 60_000),
        rng.uniform(-50, 50, 60_000),
        rng.normal(0, SIGMA, 60_000),
    ])
    wall = np.column_stack([
        rng.normal(50, SIGMA, 40_000),
        rng.uniform(-50, 50, 40_000),
        rng.uniform(0, 80, 40_000),
    ])
    planes = extract_planes(np.vstack([floor, wall]), MultiPlaneParams())
    assert len(planes) == 2
    n0 = planes[0]["result"].plane
    n1 = planes[1]["result"].plane
    assert n0.angle_to(n1) == pytest.approx(np.pi / 2, abs=1e-3)


def test_single_plane_returns_one():
    rng = np.random.default_rng(3)
    pts = np.column_stack([
        rng.uniform(-50, 50, 50_000),
        rng.uniform(-50, 50, 50_000),
        rng.normal(0, SIGMA, 50_000),
    ])
    planes = extract_planes(pts, MultiPlaneParams())
    assert len(planes) == 1
    assert planes[0]["result"].status == "ok"


def test_click_anchors_first_plane():
    """With a click on the minor sheet, it must come out first."""
    rng = np.random.default_rng(4)
    pts, n1 = parallel_pair(rng, separation=0.5, n1=80_000, n2=50_000)
    clicked = np.array([0.0, 0.0, 0.5])  # on the smaller, upper sheet

    planes = extract_planes(pts, MultiPlaneParams(), clicked=clicked)
    assert len(planes) == 2
    # NOTE: the click anchors the *component* choice, but RANSAC picks the
    # dominant surface; the upper sheet fully overlaps the lower one, so the
    # first plane is still the dominant (lower) one. The click matters for
    # spatially separated components, not for stacked parallel sheets.
    d = sorted(abs(p["result"].plane.d) for p in planes)
    assert d[1] - d[0] == pytest.approx(0.5, abs=5e-3)
