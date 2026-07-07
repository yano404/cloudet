"""Tests for click-driven extraction and project I/O (GUI-independent)."""

import json

import numpy as np
import pytest

from detpos.groups import load_groups
from detpos.picking import PickParams, pick_plane_region
from detpos.project import (
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
    """Documents the legacy behaviour: connect=False keeps the slab."""
    rng = np.random.default_rng(1)
    pts = make_scene(rng)
    clicked = np.array([0.0, 0.0, 0.0])
    nb = brute_neighbors(pts, clicked, 10.0)

    idx, _ = pick_plane_region(pts, clicked, nb, PickParams(connect=False))
    assert np.count_nonzero((idx >= 40_000) & (idx < 70_000)) > 25_000


def test_pick_too_few_neighbors_raises():
    rng = np.random.default_rng(2)
    pts = make_scene(rng)
    with pytest.raises(ValueError, match="too few neighbor"):
        pick_plane_region(pts, [0, 0, 0], np.arange(10), PickParams())


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
