"""Synthetic tests for cylinder and circle fitting (diameter-based API)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cloudet.core.circle import Circle, robust_fit_circle
from cloudet.core.cylinder import Cylinder, robust_fit_cylinder
from cloudet.fit.picking import PickParams
from cloudet.project import (
    SourceInfo,
    circle_from_json,
    circle_to_json,
    cylinder_from_json,
    cylinder_to_json,
    load_fitted_circle,
    load_fitted_cylinder,
    save_group,
    save_manifest,
)


def _cylinder_cloud(
    *,
    axis_point=None,
    direction=None,
    diameter_mm: float = 80.0,
    n: int = 800,
    length: float = 200.0,
    noise: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p0 = np.zeros(3) if axis_point is None else np.asarray(axis_point, dtype=np.float64)
    u = np.array([0.0, 0.0, 1.0]) if direction is None else np.asarray(direction, dtype=np.float64)
    u = u / np.linalg.norm(u)
    # Orthonormal frame
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    r = 0.5 * diameter_mm
    t = rng.uniform(-0.5 * length, 0.5 * length, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    pts = (
        p0
        + t[:, None] * u
        + (r * np.cos(theta))[:, None] * e1
        + (r * np.sin(theta))[:, None] * e2
    )
    pts += rng.normal(0.0, noise, size=pts.shape)
    return pts


def _circle_cloud(
    *,
    center=None,
    normal=None,
    diameter_mm: float = 50.0,
    n: int = 400,
    noise: float = 0.05,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    nrm = np.array([0.0, 0.0, 1.0]) if normal is None else np.asarray(normal, dtype=np.float64)
    nrm = nrm / np.linalg.norm(nrm)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(nrm, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    r = 0.5 * diameter_mm
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    # Slight radial jitter around the ring
    rr = r + rng.normal(0.0, noise, size=n)
    pts = c + rr[:, None] * np.cos(theta)[:, None] * e1 + rr[:, None] * np.sin(theta)[:, None] * e2
    pts += rng.normal(0.0, 0.5 * noise, size=pts.shape) * nrm
    return pts


def test_robust_fit_cylinder_free_diameter():
    true_d = 80.0
    pts = _cylinder_cloud(diameter_mm=true_d, seed=42)
    res = robust_fit_cylinder(pts, threshold=2.0, seed=0, ransac_iterations=400)
    assert res.status in ("ok", "suspect")
    assert res.n_inliers > 0.7 * len(pts)
    assert abs(res.cylinder.diameter_mm - true_d) < 1.0
    assert abs(abs(res.cylinder.direction[2]) - 1.0) < 0.05
    assert not res.cylinder.diameter_fixed


def test_robust_fit_cylinder_fixed_diameter():
    true_d = 60.0
    pts = _cylinder_cloud(diameter_mm=true_d, seed=7)
    res = robust_fit_cylinder(
        pts,
        threshold=2.0,
        seed=1,
        diameter_mm=true_d,
        diameter_fixed=True,
        ransac_iterations=400,
    )
    assert res.cylinder.diameter_fixed
    assert res.cylinder.diameter_mm == pytest.approx(true_d)
    assert abs(abs(res.cylinder.direction[2]) - 1.0) < 0.05
    assert res.n_inliers > 0.7 * len(pts)


def test_robust_fit_cylinder_wrong_fixed_diameter_is_worse():
    true_d = 80.0
    pts = _cylinder_cloud(diameter_mm=true_d, seed=3)
    good = robust_fit_cylinder(
        pts, threshold=2.0, seed=2, diameter_mm=true_d, diameter_fixed=True
    )
    bad = robust_fit_cylinder(
        pts, threshold=2.0, seed=2, diameter_mm=40.0, diameter_fixed=True
    )
    assert good.stats_inliers.get("mad_sigma", 1.0) < bad.stats_inliers.get(
        "mad_sigma", 0.0
    ) or good.n_inliers > bad.n_inliers


def test_robust_fit_cylinder_adaptive_threshold():
    """GUI cylinder path uses threshold=None (adaptive after RANSAC boot)."""
    true_d = 80.0
    pts = _cylinder_cloud(n=400, seed=5)
    res = robust_fit_cylinder(
        pts, threshold=None, seed=0, ransac_iterations=400
    )
    assert abs(res.cylinder.diameter_mm - true_d) < 3.0
    assert res.n_inliers > 100


def test_robust_fit_cylinder_tilted_axis_sampling():
    """Axis is sampled each RANSAC iter (not PCA-locked), so tilted ducts work."""
    true_d = 70.0
    direction = np.array([0.3, 0.2, 1.0])
    direction = direction / np.linalg.norm(direction)
    pts = _cylinder_cloud(
        diameter_mm=true_d, direction=direction, n=600, seed=9, noise=0.08
    )
    rng = np.random.default_rng(9)
    flange = pts.mean(axis=0) + np.column_stack([
        rng.uniform(-60, 60, 200),
        rng.uniform(-60, 60, 200),
        np.full(200, 120.0),
    ])
    mixed = np.vstack([pts, flange])
    res = robust_fit_cylinder(
        mixed,
        threshold=2.0,
        seed=0,
        diameter_mm=true_d,
        diameter_fixed=True,
        ransac_iterations=800,
    )
    assert abs(res.cylinder.diameter_mm - true_d) < 1e-6
    assert abs(float(res.cylinder.direction @ direction)) > 0.92
    assert res.n_inliers > 0.5 * len(pts)


def test_pick_cylinder_region_shell_filters_interior():
    from cloudet.core.cylinder import distances_to_axis
    from cloudet.fit.picking import pick_cylinder_region

    true_d = 80.0
    pts = _cylinder_cloud(n=500, seed=3, diameter_mm=true_d)
    clicked = pts[10]
    rng = np.random.default_rng(3)
    inside = rng.normal(0.0, 2.0, size=(150, 3))
    outside = pts.mean(axis=0) + np.array([120.0, 0.0, 0.0]) + rng.normal(
        0.0, 3.0, size=(150, 3)
    )
    cloud = np.vstack([pts, inside, outside])
    d = np.linalg.norm(cloud - clicked, axis=1)
    ball = np.flatnonzero(d <= 100.0)
    idx, plane = pick_cylinder_region(
        cloud, clicked, ball, diameter_mm=true_d
    )
    assert plane.normal.shape == (3,)
    assert len(idx) < len(ball)
    assert len(idx) >= 50
    radial = distances_to_axis(cloud[idx], np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert float(np.median(radial)) == pytest.approx(0.5 * true_d, abs=8.0)


def test_pick_cylinder_region_keeps_float32_cloud():
    """Regression: must not promote the full survey cloud to float64."""
    from cloudet.fit.picking import pick_cylinder_region

    true_d = 80.0
    pts = _cylinder_cloud(n=400, seed=4, diameter_mm=true_d).astype(np.float32)
    cloud = np.ascontiguousarray(pts)
    clicked = cloud[0].astype(np.float64)
    ball = np.arange(len(cloud), dtype=np.int64)
    _idx, _plane = pick_cylinder_region(
        cloud, clicked, ball, diameter_mm=true_d, max_ball_points=200
    )
    assert cloud.dtype == np.float32
    assert cloud.flags["OWNDATA"]


def test_distances_to_axis_matches_cross():
    from cloudet.core.cylinder import distances_to_axis

    rng = np.random.default_rng(0)
    pts = rng.normal(size=(2000, 3))
    p = np.array([0.1, -0.2, 0.3])
    u = np.array([0.2, 0.8, -0.3])
    u = u / np.linalg.norm(u)
    got = distances_to_axis(pts, p, u)
    ref = np.linalg.norm(np.cross(pts - p, u), axis=1)
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-12)


def test_cylinder_from_three_points_free_and_fixed():
    from cloudet.core.cylinder import cylinder_from_three_points

    r = 40.0
    # Equilateral-ish points on z=0 circle, axis +Z.
    a = np.array([r, 0.0, 0.0])
    b = np.array([-0.5 * r, 0.5 * np.sqrt(3) * r, 0.0])
    c = np.array([-0.5 * r, -0.5 * np.sqrt(3) * r, 0.0])
    free = cylinder_from_three_points(a, b, c)
    assert free is not None
    assert abs(free.diameter_mm - 2.0 * r) < 1e-6
    assert abs(abs(free.direction[2]) - 1.0) < 1e-9
    assert np.linalg.norm(free.point[:2]) < 1e-6

    fixed = cylinder_from_three_points(
        a, b, c, diameter_mm=100.0, diameter_fixed=True
    )
    assert fixed is not None
    assert fixed.diameter_fixed
    assert fixed.diameter_mm == pytest.approx(100.0)
    assert abs(abs(fixed.direction[2]) - 1.0) < 1e-9

    assert cylinder_from_three_points(a, a + [1, 0, 0], a + [2, 0, 0]) is None


def test_geometric_refine_beats_algebraic_on_short_arc():
    """Short circumferential arc: geometric ρ−r refine recovers center better."""
    from cloudet.core.cylinder import refine_cylinder, refine_cylinder_geometric

    true_d = 80.0
    r = 0.5 * true_d
    rng = np.random.default_rng(21)
    # ~70° arc only (algebraic circle bias is severe here).
    theta = rng.uniform(-0.6, 0.6, size=300)
    z = rng.uniform(-30.0, 30.0, size=300)
    pts = np.column_stack([
        r * np.cos(theta) + rng.normal(0, 0.15, size=300),
        r * np.sin(theta) + rng.normal(0, 0.15, size=300),
        z,
    ])
    # Biased init: axis OK, center shifted outward (typical algebraic failure).
    init = Cylinder(
        point=np.array([15.0, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        diameter_mm=true_d,
        diameter_fixed=True,
    )
    alg = refine_cylinder(
        pts, init, diameter_mm=true_d, diameter_fixed=True, update_direction=False
    )
    geo = refine_cylinder_geometric(
        pts, init, diameter_mm=true_d, diameter_fixed=True, lock_direction=True
    )
    err_alg = float(np.linalg.norm(alg.point[:2]))
    err_geo = float(np.linalg.norm(geo.point[:2]))
    assert err_geo < err_alg
    assert err_geo < 5.0


def test_robust_fit_with_init_uses_geometric_polish():
    true_d = 80.0
    pts = _cylinder_cloud(n=500, seed=22, diameter_mm=true_d, noise=0.2)
    # Perturb init slightly.
    init = Cylinder(
        point=np.array([2.0, -1.5, 0.0]),
        direction=np.array([0.02, 0.0, 1.0]),
        diameter_mm=true_d,
        diameter_fixed=True,
    )
    res = robust_fit_cylinder(
        pts,
        threshold=1.0,
        init=init,
        diameter_mm=true_d,
        diameter_fixed=True,
        max_iterations=12,
    )
    assert abs(float(res.cylinder.direction @ np.array([0.0, 0.0, 1.0]))) > 0.98
    assert float(np.linalg.norm(res.cylinder.point[:2])) < 2.0
    assert res.cylinder.diameter_mm == pytest.approx(true_d)


def test_robust_fit_with_init_locks_direction_on_rim():
    """Short circular rim: PCA would pick an in-plane axis; init must win."""
    true_u = np.array([0.0, 0.0, 1.0])
    true_d = 80.0
    r = 0.5 * true_d
    rng = np.random.default_rng(9)
    theta = rng.uniform(0, 2 * np.pi, size=400)
    pts = np.column_stack([
        r * np.cos(theta) + rng.normal(0, 0.3, size=400),
        r * np.sin(theta) + rng.normal(0, 0.3, size=400),
        rng.normal(0, 0.4, size=400),
    ])
    init = Cylinder(
        point=np.zeros(3),
        direction=true_u,
        diameter_mm=true_d,
        diameter_fixed=True,
    )
    res = robust_fit_cylinder(
        pts,
        threshold=2.0,
        init=init,
        diameter_mm=true_d,
        diameter_fixed=True,
        max_iterations=8,
    )
    assert abs(float(res.cylinder.direction @ true_u)) > 0.98
    assert res.cylinder.diameter_mm == pytest.approx(true_d)


def test_pick_cylinder_region_from_cylinder_uses_seed_axis():
    from cloudet.core.cylinder import Cylinder, distances_to_axis
    from cloudet.fit.picking import pick_cylinder_region_from_cylinder

    true_d = 80.0
    pts = _cylinder_cloud(n=600, seed=5, diameter_mm=true_d)
    cyl = Cylinder(
        point=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        diameter_mm=true_d,
        diameter_fixed=True,
    )
    ball = np.arange(len(pts), dtype=np.int64)
    idx, plane = pick_cylinder_region_from_cylinder(
        pts, ball, cyl, anchor=np.zeros(3)
    )
    assert plane.normal.shape == (3,)
    assert len(idx) >= 50
    radial = distances_to_axis(pts[idx], cyl.point, cyl.direction)
    assert float(np.median(radial)) == pytest.approx(0.5 * true_d, abs=6.0)


def test_resolve_cylinder_shell_mm_auto_and_override():
    from cloudet.fit.picking import resolve_cylinder_shell_mm

    hw, hl = resolve_cylinder_shell_mm(40.0)
    assert hw == pytest.approx(max(6.0, 0.15 * 40.0))
    assert hl == pytest.approx(max(40.0, 40.0))
    hw2, hl2 = resolve_cylinder_shell_mm(
        40.0, shell_half_width_mm=12.0, axial_half_length_mm=100.0
    )
    assert hw2 == pytest.approx(12.0)
    assert hl2 == pytest.approx(100.0)
    # 0 means auto
    hw3, _ = resolve_cylinder_shell_mm(40.0, shell_half_width_mm=0.0)
    assert hw3 == pytest.approx(hw)


def test_residual_cylinder_map_bins_radial():
    from cloudet.core.cylinder import Cylinder
    from cloudet.fit.pipeline import residual_cylinder_map

    true_d = 80.0
    pts = _cylinder_cloud(n=800, seed=6, diameter_mm=true_d, length=100.0)
    cyl = Cylinder(
        point=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        diameter_mm=true_d,
    )
    mp = residual_cylinder_map(pts, cyl, bins=40)
    assert mp["kind"] == "cylinder"
    assert mp["mean"].shape == (40, 40)
    assert mp["n_used"] == len(pts)
    # Noise-free cloud → residuals near zero.
    assert float(np.nanmedian(np.abs(mp["r"]))) < 1.0
    assert mp["u_edges"][0] >= -0.5 * true_d * np.pi - 1.0
    assert mp["v_edges"][-1] - mp["v_edges"][0] > 50.0


def test_robust_fit_circle_free_and_fixed():
    true_d = 50.0
    pts = _circle_cloud(diameter_mm=true_d, seed=11)
    free = robust_fit_circle(pts, threshold=0.5, seed=0, ransac_iterations=300)
    assert abs(free.circle.diameter_mm - true_d) < 1.5
    assert abs(abs(free.circle.normal[2]) - 1.0) < 0.1

    fixed = robust_fit_circle(
        pts,
        threshold=0.5,
        seed=0,
        diameter_mm=true_d,
        diameter_fixed=True,
        ransac_iterations=300,
    )
    assert fixed.circle.diameter_fixed
    assert fixed.circle.diameter_mm == pytest.approx(true_d)
    assert np.linalg.norm(fixed.circle.center[:2]) < 1.0


def test_robust_fit_circle_locks_provided_plane():
    """UV / Groups face: center must stay on the given plane (no plane re-fit)."""
    from cloudet.core.plane import Plane

    true_d = 50.0
    # Tilted support plane; points have small out-of-plane noise so a re-fit
    # would drift the normal if it were allowed.
    n = np.array([0.2, -0.1, 1.0], dtype=np.float64)
    n = n / np.linalg.norm(n)
    plane = Plane(n, -3.0)
    origin = -plane.d * plane.normal
    # Build an orthonormal UV on the locked plane.
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    rng = np.random.default_rng(7)
    theta = rng.uniform(0, 2 * np.pi, size=180)
    r = 0.5 * true_d
    xy = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    pts = origin + xy[:, 0:1] * u + xy[:, 1:2] * v
    pts = pts + rng.normal(0.0, 0.15, size=pts.shape) * n  # out-of-plane only
    pts = pts + rng.normal(0.0, 0.05, size=pts.shape)  # tiny isotropic

    res = robust_fit_circle(
        pts,
        threshold=0.8,
        seed=0,
        plane=plane,
        diameter_mm=true_d,
        diameter_fixed=True,
        ransac_iterations=400,
    )
    assert abs(float(res.circle.normal @ plane.normal)) > 0.999
    assert abs(float(plane.signed_distances(res.circle.center.reshape(1, 3))[0])) < 1e-9
    assert res.circle.diameter_mm == pytest.approx(true_d)


def test_recipe_binds_cylinder_and_circle(tmp_path):
    from cloudet.reduction import ReductionSession

    params = PickParams()
    cyl_pts = _cylinder_cloud(n=200, seed=0)
    cir_pts = _circle_cloud(n=200, seed=1)
    cyl = Cylinder(
        point=[0, 0, 0],
        direction=[0, 0, 1],
        diameter_mm=80.0,
        diameter_fixed=True,
    )
    cir = Circle(
        center=[0, 0, 0],
        normal=[0, 0, 1],
        diameter_mm=50.0,
        diameter_fixed=True,
    )
    save_group(
        tmp_path,
        0,
        "bore",
        cyl_pts,
        None,
        None,
        None,
        detection=params,
        fit_summary={
            "cylinders": [
                {
                    "cylinder_index": 0,
                    **cylinder_to_json(cyl),
                    "n_points": 200,
                    "status": "ok",
                }
            ]
        },
    )
    save_group(
        tmp_path,
        1,
        "hole",
        cir_pts,
        None,
        None,
        None,
        detection=params,
        fit_summary={
            "circles": [
                {
                    "circle_index": 0,
                    **circle_to_json(cir),
                    "n_points": 200,
                    "status": "ok",
                }
            ]
        },
    )
    save_manifest(tmp_path, SourceInfo(path="s.ply", n_points=400), params, n_groups=2)
    recipe = {
        "version": 2,
        "units": "mm",
        "faces": {
            "bore_axis": {
                "from": "group",
                "name": "bore",
                "kind": "cylinder",
                "diameter_mm": 80.0,
                "diameter_fixed": True,
            },
            "hole_center": {
                "from": "group",
                "name": "hole",
                "kind": "circle",
                "diameter_mm": 50.0,
                "diameter_fixed": True,
            },
        },
        "construct": [],
        "export": ["bore_axis", "hole_center"],
    }
    sess = ReductionSession.from_recipe(recipe, project_dir=tmp_path)
    assert sess.kind_of("bore_axis") == "line"
    assert sess.kind_of("hole_center") == "point"
    assert sess._face_specs["bore_axis"]["diameter_mm"] == 80.0
    assert sess._face_specs["hole_center"]["diameter_fixed"] is True
    assert abs(abs(sess.line("bore_axis").direction[2]) - 1.0) < 1e-9


def test_cylinder_circle_json_roundtrip_and_group_save(tmp_path):
    cyl = Cylinder(
        point=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        diameter_mm=80.0,
        diameter_fixed=True,
    )
    cir = Circle(
        center=[1.0, 2.0, 3.0],
        normal=[0.0, 0.0, 1.0],
        diameter_mm=50.0,
        diameter_fixed=False,
    )
    assert cylinder_from_json(cylinder_to_json(cyl)).diameter_mm == 80.0
    assert circle_from_json(circle_to_json(cir)).diameter_mm == 50.0

    pts = _cylinder_cloud(n=100, seed=0)
    params = PickParams()
    fit = {
        "planes": [],
        "cylinders": [
            {
                "cylinder_index": 0,
                **cylinder_to_json(cyl),
                "n_points": 100,
                "status": "ok",
                "reasons": [],
                "mad_sigma_mm": 0.05,
                "threshold_mm": 0.5,
            }
        ],
        "circles": [
            {
                "circle_index": 0,
                **circle_to_json(cir),
                "n_points": 50,
                "status": "ok",
                "reasons": [],
                "mad_sigma_mm": 0.04,
                "threshold_mm": 0.5,
            }
        ],
    }
    save_group(
        tmp_path,
        0,
        "bore",
        pts,
        None,
        None,
        None,
        detection=params,
        fit_summary=fit,
    )
    save_manifest(tmp_path, SourceInfo(path="s.ply", n_points=100), params, n_groups=1)
    loaded_c = load_fitted_cylinder(tmp_path, name="bore")
    assert loaded_c.cylinder.diameter_mm == pytest.approx(80.0)
    assert loaded_c.cylinder.diameter_fixed
    loaded_r = load_fitted_circle(tmp_path, name="bore")
    assert loaded_r.circle.diameter_mm == pytest.approx(50.0)
    doc = json.loads((tmp_path / "groups" / "group_000.json").read_text())
    assert "cylinders" in doc["fit"]
    assert "circles" in doc["fit"]

    from cloudet.project.store import load_group_fit

    restored = load_group_fit(tmp_path, 0)
    assert restored is not None
    assert len(restored.get("cylinders") or []) == 1
    assert len(restored.get("circles") or []) == 1
    assert restored["cylinders"][0]["diameter_mm"] == pytest.approx(80.0)
    assert restored["circles"][0]["diameter_mm"] == pytest.approx(50.0)


def test_load_group_fit_cylinder_only_without_planes(tmp_path):
    """Cylinder-only groups must reload even when fit.planes is empty."""
    from cloudet.project.store import load_group_fit

    pts = _cylinder_cloud(n=80, seed=3)
    params = PickParams()
    cyl = Cylinder(
        point=[0, 0, 0],
        direction=[0, 0, 1],
        diameter_mm=80.0,
        diameter_fixed=True,
    )
    save_group(
        tmp_path,
        1,
        "duct",
        pts,
        None,
        None,
        None,
        detection=params,
        fit_summary={
            "planes": [],
            "cylinders": [
                {
                    "cylinder_index": 0,
                    **cylinder_to_json(cyl),
                    "n_points": 80,
                    "status": "ok",
                }
            ],
        },
    )
    restored = load_group_fit(tmp_path, 1)
    assert restored is not None
    assert restored.get("planes") == []
    assert len(restored["cylinders"]) == 1
    assert "circles" not in restored


def test_load_fitted_circle_by_index(tmp_path):
    from cloudet.core.circle import Circle
    from cloudet.project.schema import circle_to_json
    from cloudet.project.store import load_fitted_circle, save_group, save_manifest
    from cloudet.project import SourceInfo
    from cloudet.fit.picking import PickParams

    pts = np.zeros((30, 3), dtype=np.float64)
    pts[:, 0] = np.linspace(-10, 10, 30)
    cir0 = Circle(center=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], diameter_mm=40.0)
    cir1 = Circle(center=[100.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], diameter_mm=55.0)
    fit = {
        "planes": [],
        "circles": [
            {"circle_index": 0, **circle_to_json(cir0), "n_points": 10, "status": "ok"},
            {"circle_index": 1, **circle_to_json(cir1), "n_points": 12, "status": "ok"},
        ],
    }
    save_group(
        tmp_path, 0, "face", pts, None, None, None,
        detection=PickParams(), fit_summary=fit,
    )
    save_manifest(tmp_path, SourceInfo(path="s.ply", n_points=30), PickParams(), n_groups=1)
    r0 = load_fitted_circle(tmp_path, name="face", circle_index=0)
    r1 = load_fitted_circle(tmp_path, name="face", circle_index=1)
    assert r0.circle.diameter_mm == pytest.approx(40.0)
    assert r1.circle.diameter_mm == pytest.approx(55.0)
    assert r1.circle_index == 1


def test_cylinder_circle_inlier_indices_roundtrip(tmp_path):
    """Cylinder / circle inliers persist like plane ``*_p*_indices.npy``."""
    from cloudet.project.store import (
        load_circle_inlier_indices,
        load_cylinder_inlier_indices,
        load_group_fit,
    )

    pts = _cylinder_cloud(n=120, seed=9)
    group_idx = np.arange(500, 620, dtype=np.int64)
    local_cyl = np.array([2, 5, 9, 11, 40], dtype=np.int64)
    local_cir = np.array([1, 3, 7], dtype=np.int64)
    cyl = Cylinder(
        point=[0, 0, 0],
        direction=[0, 0, 1],
        diameter_mm=80.0,
        diameter_fixed=True,
    )
    cir = Circle(
        center=[0, 0, 0],
        normal=[0, 0, 1],
        diameter_mm=50.0,
        diameter_fixed=True,
    )
    params = PickParams()
    save_group(
        tmp_path,
        3,
        "mixed",
        pts,
        group_idx,
        None,
        None,
        detection=params,
        fit_summary={
            "planes": [],
            "cylinders": [
                {
                    "cylinder_index": 0,
                    **cylinder_to_json(cyl),
                    "n_points": 5,
                    "status": "ok",
                    "inlier_local": local_cyl,
                }
            ],
            "circles": [
                {
                    "circle_index": 0,
                    **circle_to_json(cir),
                    "n_points": 3,
                    "status": "ok",
                    "support_plane_index": 0,
                    "inlier_local": local_cir,
                }
            ],
        },
    )
    src_cyl = load_cylinder_inlier_indices(tmp_path, 3, 0)
    src_cir = load_circle_inlier_indices(tmp_path, 3, 0)
    assert src_cyl is not None and src_cir is not None
    assert np.array_equal(src_cyl, group_idx[local_cyl])
    assert np.array_equal(src_cir, group_idx[local_cir])

    doc = json.loads((tmp_path / "groups" / "group_003.json").read_text())
    assert doc["fit"]["cylinders"][0]["inlier_indices_file"] == (
        "group_003_cyl0_indices.npy"
    )
    assert doc["fit"]["circles"][0]["inlier_indices_file"] == (
        "group_003_cir0_indices.npy"
    )
    assert "inlier_local" not in doc["fit"]["cylinders"][0]

    restored = load_group_fit(tmp_path, 3, group_idx)
    assert restored is not None
    assert np.array_equal(restored["cylinders"][0]["inlier_local"], local_cyl)
    assert np.array_equal(restored["circles"][0]["inlier_local"], local_cir)
    assert np.array_equal(
        restored["cylinders"][0]["inlier_source"], group_idx[local_cyl]
    )

    # Drop circle and re-save: leftover cir npy is removed; cyl kept.
    save_group(
        tmp_path,
        3,
        "mixed",
        pts,
        group_idx,
        None,
        None,
        detection=params,
        fit_summary={
            "planes": [],
            "cylinders": [
                doc["fit"]["cylinders"][0] | {"inlier_source": src_cyl}
            ],
        },
    )
    assert (tmp_path / "groups" / "group_003_cyl0_indices.npy").exists()
    assert not (tmp_path / "groups" / "group_003_cir0_indices.npy").exists()
