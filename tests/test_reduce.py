"""Project fit loading and declarative geometry reduction."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cloudet.picking import PickParams
from cloudet.plane import Plane
from cloudet.project import (
    SourceInfo,
    load_fitted_plane,
    load_group_docs,
    save_group,
    write_manifest,
)
from cloudet.reduce import load_recipe, run_reduction, write_geometry_json


def _fit_planes(*abcd_list, statuses=None):
    statuses = statuses or ["ok"] * len(abcd_list)
    return {
        "planes": [
            {
                "plane_index": i,
                "abcd": list(abcd),
                "n_points": 1000,
                "status": statuses[i],
                "reasons": [],
                "bimodal": False,
                "mad_sigma_mm": 0.05,
                "threshold_mm": 0.15,
            }
            for i, abcd in enumerate(abcd_list)
        ]
    }


def _make_project(tmp_path):
    """Three axis-aligned faces: x=-50, y=-20, z=100 (as Hesse abcd)."""
    params = PickParams()
    faces = [
        (0, "tracker_left", Plane(np.array([1.0, 0.0, 0.0]), 50.0)),   # x=-50
        (1, "tracker_front", Plane(np.array([0.0, 1.0, 0.0]), 20.0)),  # y=-20
        (2, "target", Plane(np.array([0.0, 0.0, 1.0]), -100.0)),       # z=100
    ]
    for gid, name, plane in faces:
        pts = np.zeros((10, 3), dtype=np.float64)
        save_group(
            tmp_path,
            gid,
            name,
            pts,
            indices=np.arange(10, dtype=np.int64),
            coarse_plane=plane.as_array(),
            clicked=[0.0, 0.0, 0.0],
            detection=params,
            fit_summary=_fit_planes(plane.as_array()),
        )
    write_manifest(
        tmp_path,
        SourceInfo(path="/data/scan.ply", n_points=100),
        params,
        n_groups=3,
    )
    return tmp_path


def test_load_fitted_plane_by_name(tmp_path):
    _make_project(tmp_path)
    fitted = load_fitted_plane(tmp_path, name="target")
    assert fitted.group_name == "target"
    assert fitted.plane_index == 0
    assert fitted.quality["status"] == "ok"
    assert abs(fitted.plane.signed_distances(np.array([[0.0, 0.0, 100.0]]))[0]) < 1e-12


def test_load_fitted_plane_missing_fit(tmp_path):
    params = PickParams()
    pts = np.zeros((5, 3))
    save_group(
        tmp_path, 0, "Bare", pts, None, None, None, detection=params, fit_summary=None
    )
    with pytest.raises(ValueError, match="no fit"):
        load_fitted_plane(tmp_path, name="Bare")


def test_load_fitted_plane_legacy_without_abcd(tmp_path):
    params = PickParams()
    pts = np.zeros((5, 3))
    save_group(
        tmp_path,
        0,
        "Legacy",
        pts,
        None,
        None,
        None,
        detection=params,
        fit_summary={"status": "ok", "mad_sigma_mm": 0.03},
    )
    with pytest.raises(ValueError, match="planes"):
        load_fitted_plane(tmp_path, name="Legacy")


def test_load_group_docs(tmp_path):
    _make_project(tmp_path)
    docs = load_group_docs(tmp_path)
    assert [d["name"] for d in docs] == ["tracker_left", "tracker_front", "target"]


def test_reduction_beam_on_target(tmp_path):
    project = _make_project(tmp_path)
    recipe = {
        "version": 1,
        "units": "mm",
        "faces": {
            "tracker_left": {"from": "group", "name": "tracker_left"},
            "tracker_front": {"from": "group", "name": "tracker_front"},
            "target": {"from": "group", "name": "target"},
        },
        "construct": [
            {"id": "left_in", "op": "offset", "of": "tracker_left", "distance_mm": 50.0},
            {"id": "front_in", "op": "offset", "of": "tracker_front", "distance_mm": 20.0},
            {"id": "beam_axis", "op": "intersect_planes", "a": "left_in", "b": "front_in"},
            {
                "id": "beam_on_target",
                "op": "intersect_line_plane",
                "line": "beam_axis",
                "plane": "target",
            },
        ],
        "export": ["beam_axis", "beam_on_target"],
    }
    result = run_reduction(project, recipe)
    assert np.allclose(result.points["beam_on_target"]["xyz"], [0.0, 0.0, 100.0])
    assert np.allclose(
        np.abs(result.lines["beam_axis"]["direction"]), [0.0, 0.0, 1.0]
    )
    assert result.planes["tracker_left"]["provenance"] == "scanned"
    assert result.planes["left_in"]["provenance"] == "offset"
    assert result.lines["beam_axis"]["provenance"] == "intersection"

    out = write_geometry_json(tmp_path / "geometry.json", result)
    doc = json.loads(out.read_text())
    assert doc["version"] == 1
    assert doc["units"] == "mm"
    assert doc["export"] == ["beam_axis", "beam_on_target"]
    assert "sha256" in doc["recipe"]


def test_load_recipe_and_unknown_op(tmp_path):
    project = _make_project(tmp_path)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "version": 1,
                "units": "mm",
                "faces": {"target": {"from": "group", "name": "target"}},
                "construct": [{"id": "x", "op": "nope"}],
            }
        )
    )
    recipe = load_recipe(recipe_path)
    with pytest.raises(ValueError, match="unknown op"):
        run_reduction(project, recipe)


def test_multiplane_index(tmp_path):
    params = PickParams()
    p0 = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    p1 = Plane(np.array([0.0, 0.0, 1.0]), -5.0)
    save_group(
        tmp_path,
        0,
        "Stack",
        np.zeros((4, 3)),
        None,
        None,
        None,
        detection=params,
        fit_summary=_fit_planes(p0.as_array(), p1.as_array()),
    )
    fitted = load_fitted_plane(tmp_path, name="Stack", plane_index=1)
    assert abs(fitted.plane.d - (-5.0)) < 1e-12 or abs(fitted.plane.d - 5.0) < 1e-12
    # z=5 plane: after sign convention n=+z, d=-5
    assert fitted.plane.d == pytest.approx(-5.0)


def test_reduction_session_records_recipe():
    from cloudet.reduce import ReductionSession

    sess = ReductionSession()
    left = Plane(np.array([1.0, 0.0, 0.0]), 50.0)
    front = Plane(np.array([0.0, 1.0, 0.0]), 20.0)
    target = Plane(np.array([0.0, 0.0, 1.0]), -100.0)
    sess.bind_scanned("tracker_left", left, group_name="G0", group_id=0)
    sess.bind_scanned("tracker_front", front, group_name="G1", group_id=1)
    sess.bind_scanned("target", target, group_name="G2", group_id=2)
    sess.offset("left_in", "tracker_left", 50.0)
    sess.offset("front_in", "tracker_front", 20.0)
    sess.intersect_planes("beam_axis", "left_in", "front_in")
    sess.intersect_line_plane("beam_on_target", "beam_axis", "target")

    recipe = sess.to_recipe(export=["beam_axis", "beam_on_target"])
    assert recipe["faces"]["tracker_left"]["name"] == "G0"
    assert recipe["construct"][0]["op"] == "offset"
    assert recipe["export"] == ["beam_axis", "beam_on_target"]

    result = sess.to_result(source_project="/tmp/proj")
    assert np.allclose(result.points["beam_on_target"]["xyz"], [0.0, 0.0, 100.0])
    assert sess.unique_id("offset").startswith("offset_")
