"""Tests for click-driven extraction and project I/O (GUI-independent)."""

import json

import numpy as np
import pytest

from cloudet.groups import load_groups
from cloudet.picking import PickParams, pick_plane_region
from cloudet.project import (
    PickerSettings,
    SourceInfo,
    load_group_indices,
    load_settings,
    read_manifest,
    save_group,
    save_settings,
    write_manifest,
)

SIGMA = 0.03


def make_scene(rng):
    """Two coplanar patches (z=0, separated in x) and a perpendicular wall."""
    def plane_patch(x0, x1, y0, y1, n):
        p = np.column_stack([
            rng.uniform(x0, x1, n),
            rng.uniform(y0, y1, n),
            rng.normal(0, SIGMA, n),
        ])
        return p

    main = plane_patch(-50, 50, -50, 50, 40_000)
    far = plane_patch(150, 250, -50, 50, 30_000)  # coplanar, disconnected
    wall = np.column_stack([
        rng.normal(0, SIGMA, 20_000) + 60.0,
        rng.uniform(-50, 50, 20_000),
        rng.uniform(0, 100, 20_000),
    ])
    return np.vstack([main, far, wall])


def brute_neighbors(points, center, radius):
    d = np.linalg.norm(points - np.asarray(center), axis=1)
    return np.flatnonzero(d <= radius)


def test_pick_connected_only():
    rng = np.random.default_rng(0)
    pts = make_scene(rng)
    clicked = np.array([0.0, 0.0, 0.0])
    nb = brute_neighbors(pts, clicked, 10.0)

    idx, plane = pick_plane_region(pts, clicked, nb, PickParams())
    # z=0 plane recovered
    assert abs(abs(plane.normal[2]) - 1.0) < 1e-3
    # main patch in, far coplanar patch out, wall out
    assert np.count_nonzero(idx < 40_000) > 38_000
    in_far = np.count_nonzero((idx >= 40_000) & (idx < 70_000))
    assert in_far == 0
    # wall contributes at most its bottom edge within 1 mm of z=0
    assert np.count_nonzero(idx >= 70_000) < 1_000


def test_pick_without_connectivity_sweeps_far_patch():
    """Without connect and without progressive expand, the slab keeps far coplanar points."""
    rng = np.random.default_rng(1)
    pts = make_scene(rng)
    clicked = np.array([0.0, 0.0, 0.0])
    nb = brute_neighbors(pts, clicked, 10.0)

    idx, _ = pick_plane_region(
        pts, clicked, nb, PickParams(connect=False, expand_step_mm=0.0)
    )
    assert np.count_nonzero((idx >= 40_000) & (idx < 70_000)) > 25_000


def test_pick_too_few_neighbors_raises():
    rng = np.random.default_rng(2)
    pts = make_scene(rng)
    with pytest.raises(ValueError, match="too few neighbor"):
        pick_plane_region(pts, [0, 0, 0], np.arange(10), PickParams())


def test_max_inplane_radius_limits_extent():
    rng = np.random.default_rng(4)
    pts = make_scene(rng)
    clicked = np.array([0.0, 0.0, 0.0])
    nb = brute_neighbors(pts, clicked, 10.0)
    idx, _ = pick_plane_region(
        pts,
        clicked,
        nb,
        PickParams(
            expand_step_mm=0.0,
            max_inplane_radius_mm=20.0,
            accumulate_threshold_mm=1.0,
        ),
    )
    sel = pts[idx]
    assert np.all(np.hypot(sel[:, 0], sel[:, 1]) <= 20.0 + 1.0)
    assert len(idx) < 25_000


def test_progressive_refine_recovers_from_tilted_seed():
    """A mildly tilted local neighbourhood should not diagonal-cut a large face."""
    from cloudet.picking import _progressive_refine_plane
    from cloudet.plane import Plane

    rng = np.random.default_rng(5)
    n = 60_000
    pts = np.column_stack([
        rng.uniform(-120, 120, n),
        rng.uniform(-120, 120, n),
        rng.normal(0, SIGMA, n),
    ])
    clicked = np.array([0.0, 0.0, 0.0])
    # ~3 deg tilt about y: normal ~ (sin θ, 0, cos θ)
    theta = np.deg2rad(3.0)
    bad = Plane(np.array([np.sin(theta), 0.0, np.cos(theta)]), 0.0)
    params = PickParams(
        accumulate_threshold_mm=0.5,
        expand_step_mm=30.0,
        max_expand_rounds=20,
        local_radius_mm=15.0,
        cell_size_mm=5.0,
    )
    refined, _R = _progressive_refine_plane(pts, clicked, bad, params)
    true_n = np.array([0.0, 0.0, 1.0])
    bad_ang = bad.angle_to(Plane(true_n, 0.0))
    good_ang = refined.angle_to(Plane(true_n, 0.0))
    assert good_ang < 0.4 * bad_ang

    idx, _ = pick_plane_region(
        pts,
        clicked,
        brute_neighbors(pts, clicked, 15.0),
        PickParams(
            local_radius_mm=15.0,
            min_neighbor_points=200,
            min_local_inliers=100,
            accumulate_threshold_mm=0.5,
            expand_step_mm=30.0,
            max_expand_rounds=20,
        ),
    )
    assert len(idx) > 50_000


def test_corner_click_covers_whole_face_without_diagonal_cut():
    """Clicking near a corner must still grab the whole face, not a band.

    The seed neighbourhood at a corner is the worst case for a tilted local
    normal; a thin slab then selects a diagonal strip across the face.
    """
    rng = np.random.default_rng(11)
    n_face = 400_000
    face = np.column_stack([
        rng.uniform(-400, 400, n_face),
        rng.uniform(-400, 400, n_face),
        rng.normal(0, SIGMA, n_face),
    ])
    n_wall = 60_000
    wall = np.column_stack([
        rng.normal(150.0, SIGMA, n_wall),
        rng.uniform(-400, 400, n_wall),
        rng.uniform(0, 200, n_wall),
    ])
    n_far = 40_000
    far = np.column_stack([  # coplanar but spatially disconnected
        rng.uniform(900, 1200, n_far),
        rng.uniform(-150, 150, n_far),
        rng.normal(0, SIGMA, n_far),
    ])
    pts = np.vstack([face, wall, far])
    clicked = np.array([-380.0, -380.0, 0.0])

    idx, plane = pick_plane_region(
        pts,
        clicked,
        brute_neighbors(pts, clicked, 15.0),
        PickParams(
            local_radius_mm=15.0, accumulate_threshold_mm=1.0, expand_step_mm=25.0
        ),
    )

    assert abs(abs(plane.normal[2]) - 1.0) < 1e-4
    n_from_face = int(np.count_nonzero(idx < n_face))
    assert n_from_face > 0.95 * n_face  # whole face, not a diagonal band
    assert int(np.count_nonzero(idx >= n_face + n_wall)) == 0  # far patch excluded

    sel = pts[idx]
    assert sel[:, 0].min() < -390 and sel[:, 0].max() > 390
    assert sel[:, 1].min() < -390 and sel[:, 1].max() > 390


@pytest.mark.parametrize("tilt_deg", [2.0, 10.0, 20.0])
def test_accumulate_refit_recovers_whole_face_from_tilted_seed(tilt_deg):
    """One pass with a tilted seed yields a band; refitting recovers the face."""
    from cloudet.picking import _accumulate_with_refit, _select_candidates
    from cloudet.plane import Plane

    rng = np.random.default_rng(7)
    n_face = 300_000
    face = np.column_stack([
        rng.uniform(-300, 300, n_face),
        rng.uniform(-300, 300, n_face),
        rng.normal(0, SIGMA, n_face),
    ])
    n_far = 40_000
    far = np.column_stack([
        rng.uniform(900, 1100, n_far),
        rng.uniform(-100, 100, n_far),
        rng.normal(0, SIGMA, n_far),
    ])
    pts = np.vstack([face, far])
    clicked = np.array([-280.0, -280.0, 0.0])
    params = PickParams(local_radius_mm=15.0, accumulate_threshold_mm=1.0)

    theta = np.deg2rad(tilt_deg)
    seed = Plane(np.array([np.sin(theta), 0.0, np.cos(theta)]), 0.0)

    band = _select_candidates(pts, seed, clicked, params, inplane_radius_mm=None)
    assert len(band) < 0.5 * n_face  # a single pass really does cut a band

    idx, plane = _accumulate_with_refit(pts, clicked, seed, params)
    assert len(idx) > 0.95 * n_face
    assert int(np.count_nonzero(idx >= n_face)) == 0  # disconnected patch excluded
    assert np.degrees(plane.angle_to(Plane(np.array([0.0, 0.0, 1.0]), 0.0))) < 0.05


def test_legacy_oneshot_still_available():
    """expand_step_mm=0 keeps a single full-slab accumulate (old behaviour)."""
    rng = np.random.default_rng(0)
    pts = make_scene(rng)
    clicked = np.array([0.0, 0.0, 0.0])
    nb = brute_neighbors(pts, clicked, 10.0)
    idx, plane = pick_plane_region(
        pts, clicked, nb, PickParams(expand_step_mm=0.0)
    )
    assert abs(abs(plane.normal[2]) - 1.0) < 1e-3
    assert np.count_nonzero(idx < 40_000) > 38_000


def test_project_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    pts = rng.uniform(-10, 10, size=(5_000, 3))
    indices = rng.choice(100_000, size=5_000, replace=False).astype(np.int64)
    params = PickParams(accumulate_threshold_mm=2.5)

    save_group(
        tmp_path, 0, "TopFace", pts, indices,
        coarse_plane=[0, 0, 1, -5], clicked=[1, 2, 3],
        color=[0.9, 0.25, 0.25], detection=params,
        fit_summary={"status": "ok", "mad_sigma_mm": 0.031},
    )
    write_manifest(
        tmp_path, SourceInfo(path="/data/scan.ply", n_points=100_000), params, n_groups=1
    )

    # loads through the same loader the fit pipeline uses
    groups = load_groups(tmp_path)
    assert len(groups) == 1
    g = groups[0]
    assert g.name == "TopFace"
    assert g.num_points == 5_000
    back = g.load_points()
    assert np.array_equal(back, pts)

    idx = load_group_indices(tmp_path, 0)
    assert np.array_equal(idx, indices)

    m = read_manifest(tmp_path)
    assert m["units"] == "mm"
    assert m["source"]["n_points"] == 100_000
    assert m["detection"]["accumulate_threshold_mm"] == 2.5

    # detection params are baked into the group json too
    doc = json.loads((tmp_path / "groups" / "group_000.json").read_text())
    assert doc["detection"]["accumulate_threshold_mm"] == 2.5
    assert doc["fit"]["status"] == "ok"


def test_settings_roundtrip_and_unknown_keys(tmp_path):
    s = PickerSettings()
    s.detection = PickParams(local_radius_mm=15.0)
    save_settings(tmp_path, s)

    loaded = load_settings(tmp_path, warn=lambda *_: None)
    assert loaded.detection.local_radius_mm == 15.0
    assert loaded.view.axis_size_mm == 100.0

    # unknown key -> warned, not crashed; typo does not silently apply
    doc = json.loads((tmp_path / "settings.json").read_text())
    doc["detection"]["local_raduis_mm"] = 99.0  # typo
    (tmp_path / "settings.json").write_text(json.dumps(doc))
    warnings = []
    loaded = load_settings(tmp_path, warn=warnings.append)
    assert loaded.detection.local_radius_mm == 15.0
    assert any("local_raduis_mm" in w for w in warnings)


def test_settings_missing_file_defaults(tmp_path):
    s = load_settings(tmp_path)
    assert s.detection == PickParams()
