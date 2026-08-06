"""Tests for main plane component extraction on rough groups."""

import numpy as np
import pytest

from cloudet.mainplane import MainPlaneParams, _label_components, extract_main_plane
from cloudet.plane import Plane


def _label_components_reference(occupied):
    """Per-cell BFS, kept only as an oracle for the vectorised labeller."""
    from collections import deque

    labels = np.zeros(occupied.shape, dtype=np.int32)
    current = 0
    nrows, ncols = occupied.shape
    for i in range(nrows):
        for j in range(ncols):
            if occupied[i, j] and labels[i, j] == 0:
                current += 1
                q = deque([(i, j)])
                labels[i, j] = current
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if (
                            0 <= yy < nrows and 0 <= xx < ncols
                            and occupied[yy, xx] and labels[yy, xx] == 0
                        ):
                            labels[yy, xx] = current
                            q.append((yy, xx))
    return labels


def _same_partition(a, b):
    """Labels may be numbered differently; compare the induced partitions."""
    if not np.array_equal(a > 0, b > 0):
        return False
    pairs = set(zip(a[a > 0].tolist(), b[b > 0].tolist()))
    return len({p[0] for p in pairs}) == len(pairs) == len({p[1] for p in pairs})


def test_label_components_matches_bfs_reference():
    rng = np.random.default_rng(0)
    for density in (0.15, 0.45, 0.8):
        occ = rng.random((40, 55)) < density
        assert _same_partition(_label_components(occ), _label_components_reference(occ))


def test_label_components_edge_cases():
    empty = np.zeros((5, 5), dtype=bool)
    assert _label_components(empty).max() == 0

    full = np.ones((4, 6), dtype=bool)
    assert _label_components(full).max() == 1

    # 4-connectivity: diagonal touch is NOT one component
    diag = np.zeros((3, 3), dtype=bool)
    diag[0, 0] = diag[1, 1] = True
    assert _label_components(diag).max() == 2

SIGMA = 0.03


def patch(rng, normal, offset, u_range, v_range, n, sigma=SIGMA):
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0.0, 0.0])
    if abs(normal @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, a); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    uu = rng.uniform(*u_range, size=n)
    vv = rng.uniform(*v_range, size=n)
    noise = rng.normal(0, sigma, size=n)
    center = -offset * normal
    return center + uu[:, None] * u + vv[:, None] * v + noise[:, None] * normal, u, v, center


def test_coplanar_distant_patch_removed():
    """Slab contamination: a second, coplanar but spatially distant patch
    must be excluded from the main component."""
    rng = np.random.default_rng(0)
    main, u, v, center = patch(rng, (0, 0, 1), 100.0, (-50, 50), (-50, 50), 50_000)
    far, *_ = patch(rng, (0, 0, 1), 100.0, (200, 260), (-50, 50), 20_000)  # same plane!
    pts = np.vstack([main, far])
    clicked = center  # click in the middle of the main patch

    res = extract_main_plane(pts, clicked=clicked)
    assert res.status == "ok", res.reasons
    # all main-patch points in, far-patch points out
    assert np.count_nonzero(res.main_mask[:50_000]) > 45_000
    assert np.count_nonzero(res.main_mask[50_000:]) == 0
    assert res.fit.stats_inliers["mad_sigma"] == pytest.approx(SIGMA, rel=0.15)


def test_click_selects_smaller_component():
    """If the user clicked the smaller of two coplanar patches, honour it."""
    rng = np.random.default_rng(1)
    small, _, _, center = patch(rng, (0, 0, 1), 100.0, (-25, 25), (-25, 25), 15_000)
    big, *_ = patch(rng, (0, 0, 1), 100.0, (100, 260), (-80, 80), 60_000)
    pts = np.vstack([small, big])

    res = extract_main_plane(pts, clicked=center)
    assert np.count_nonzero(res.main_mask[:15_000]) > 13_000
    assert np.count_nonzero(res.main_mask[15_000:]) == 0


def test_ghost_surface_rejected():
    rng = np.random.default_rng(2)
    main, _, _, center = patch(rng, (0.3, -0.5, 0.8), 250.0, (-50, 50), (-50, 50), 60_000)
    ghost, *_ = patch(rng, (0.3, -0.5, 0.8), 250.5, (-50, 50), (-50, 50), 12_000)
    pts = np.vstack([main, ghost])
    n = np.array([0.3, -0.5, 0.8])
    true = Plane(n / np.linalg.norm(n), 250.0)  # d applies to the unit normal

    res = extract_main_plane(pts, clicked=center)
    assert res.status == "ok", res.reasons
    assert abs(res.plane.d - true.d) < 3e-3
    assert res.plane.angle_to(true) < 1e-4


def test_curved_surface_flagged():
    """A cylinder-like surface is not a plane: must not silently return ok."""
    rng = np.random.default_rng(3)
    R = 500.0  # mm, gentle curvature -> sagitta over 100mm is ~2.5mm
    xx = rng.uniform(-50, 50, size=80_000)
    yy = rng.uniform(-50, 50, size=80_000)
    zz = R - np.sqrt(R**2 - xx**2) + rng.normal(0, SIGMA, size=80_000)
    pts = np.column_stack([xx, yy, zz])

    res = extract_main_plane(pts)
    assert res.status in ("suspect", "fail")
    assert res.reasons


def test_no_click_uses_largest_component():
    rng = np.random.default_rng(4)
    big, *_ = patch(rng, (0, 0, 1), 100.0, (-60, 60), (-60, 60), 60_000)
    small, *_ = patch(rng, (0, 0, 1), 100.0, (200, 240), (-20, 20), 8_000)
    pts = np.vstack([big, small])

    res = extract_main_plane(pts)
    assert np.count_nonzero(res.main_mask[:60_000]) > 54_000
    assert np.count_nonzero(res.main_mask[60_000:]) == 0


def test_bounded_threshold_does_not_diverge():
    """Non-planar blob: adaptive threshold must stay <= ceiling (G11 case)."""
    rng = np.random.default_rng(5)
    pts = rng.uniform(-30, 30, size=(50_000, 3))  # a solid box, not a plane
    params = MainPlaneParams(min_points=100)
    res = extract_main_plane(pts, params=params)
    assert res.fit.threshold <= params.max_threshold_mm + 1e-12
    assert res.status == "fail"
