"""Project fit loading and declarative geometry reduction."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cloudet.fit.picking import PickParams
from cloudet.core.plane import Plane
from cloudet.project import (
    SourceInfo,
    load_fitted_plane,
    load_group_docs,
    save_group,
    save_manifest,
)
from cloudet.reduction import load_recipe, run_reduction, write_geometry_json


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
    save_manifest(
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
    assert result.aligned is None
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
    from cloudet.reduction import ReductionSession

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


def test_reduction_session_line_from_point_normal():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    wall = Plane(np.array([1.0, 0.0, 0.0]), 50.0)   # x = -50
    front = Plane(np.array([0.0, 1.0, 0.0]), 20.0)  # y = -20
    target = Plane(np.array([0.0, 0.0, 1.0]), -100.0)
    sess.bind_scanned("wall", wall, group_name="G0", group_id=0)
    sess.bind_scanned("front", front, group_name="G1", group_id=1)
    sess.bind_scanned("target", target, group_name="G2", group_id=2)
    sess.intersect_three_planes("corner", "wall", "front", "target")
    sess.line_from_point_normal("n_axis", "corner", "target")

    line = sess.line("n_axis")
    assert np.allclose(line.point, sess.point("corner"))
    assert np.allclose(np.abs(line.direction), [0.0, 0.0, 1.0])
    recipe = sess.to_recipe()
    assert recipe["construct"][-1]["op"] == "line_from_point_normal"
    assert recipe["construct"][-1]["point"] == "corner"
    assert recipe["construct"][-1]["plane"] == "target"


def test_reduction_session_line_from_two_points():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    wall = Plane(np.array([1.0, 0.0, 0.0]), 50.0)
    front = Plane(np.array([0.0, 1.0, 0.0]), 20.0)
    target = Plane(np.array([0.0, 0.0, 1.0]), -100.0)
    sess.bind_scanned("wall", wall, group_name="G0", group_id=0)
    sess.bind_scanned("front", front, group_name="G1", group_id=1)
    sess.bind_scanned("target", target, group_name="G2", group_id=2)
    sess.intersect_three_planes("p0", "wall", "front", "target")
    sess.offset("target_out", "target", 80.0)
    sess.intersect_three_planes("p1", "wall", "front", "target_out")
    sess.line_from_two_points("chord", "p0", "p1")

    line = sess.line("chord")
    assert np.allclose(np.abs(line.direction), [0.0, 0.0, 1.0])
    assert np.allclose(sess.anchors["chord"], 0.5 * (sess.point("p0") + sess.point("p1")))
    recipe = sess.to_recipe()
    step = recipe["construct"][-1]
    assert step["op"] == "line_from_two_points"
    assert step["a"] == "p0"
    assert step["b"] == "p1"
    sess.rename("p0", "origin")
    live = [s for s in sess.to_recipe()["construct"] if s["id"] == "chord"][0]
    assert live["a"] == "origin"
    with pytest.raises(ValueError, match="must differ"):
        sess.line_from_two_points("bad", "p1", "p1")


def test_reduction_session_intersect_normal_plane():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    src = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    dst = Plane(np.array([0.0, 0.0, 1.0]), -100.0)
    sess.bind_scanned("src", src, group_name="G0", group_id=0)
    sess.bind_scanned("dst", dst, group_name="G1", group_id=1)
    sess.intersect_normal_plane("hit", "src", "dst")
    assert np.allclose(sess.point("hit"), [0.0, 0.0, 100.0])
    rec = sess.record_of("hit")
    assert rec["op"] == "intersect_normal_plane"
    step = [s for s in sess.to_recipe()["construct"] if s["id"] == "hit"][0]
    assert step["src"] == "src"
    assert step["dst"] == "dst"
    with pytest.raises(ValueError, match="must differ"):
        sess.intersect_normal_plane("bad", "src", "src")


def test_intersect_normal_plane_uses_src_overlay_anchor():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    src = Plane(np.array([1.0, 0.0, 0.0]), -1000.0)  # x = 1000
    dst = Plane(np.array([1.0, 0.0, 0.0]), -1100.0)  # x = 1100
    sess.bind_scanned(
        "src",
        src,
        group_name="G0",
        group_id=0,
        anchor=np.array([1000.0, 5000.0, 3000.0]),
    )
    sess.bind_scanned("dst", dst, group_name="G1", group_id=1)
    sess.intersect_normal_plane("hit", "src", "dst")
    assert np.allclose(sess.point("hit"), [1100.0, 5000.0, 3000.0])


def test_reduction_session_midpoint_line_planes():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    left = Plane(np.array([1.0, 0.0, 0.0]), 50.0)
    front = Plane(np.array([0.0, 1.0, 0.0]), 20.0)
    z0 = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    z10 = Plane(np.array([0.0, 0.0, 1.0]), -10.0)
    sess.bind_scanned("left", left, group_name="G0", group_id=0)
    sess.bind_scanned("front", front, group_name="G1", group_id=1)
    sess.bind_scanned("z0", z0, group_name="G2", group_id=2)
    sess.bind_scanned("z10", z10, group_name="G3", group_id=3)
    sess.intersect_planes("axis", "left", "front")
    sess.midpoint_line_planes("mid", "axis", "z0", "z10")
    assert np.allclose(sess.point("mid"), [-50.0, -20.0, 5.0])
    assert np.allclose(sess.anchors["mid"], sess.point("mid"))
    rec = sess.record_of("mid")
    assert rec["op"] == "midpoint_line_planes"
    assert np.allclose(rec["ends"][0][2], 0.0) or np.allclose(rec["ends"][1][2], 0.0)
    recipe = sess.to_recipe()
    step = recipe["construct"][-1]
    assert step["op"] == "midpoint_line_planes"
    assert step["line"] == "axis"
    sess.rename("z0", "near")
    live = [s for s in sess.to_recipe()["construct"] if s["id"] == "mid"][0]
    assert live["a"] == "near"
    with pytest.raises(ValueError, match="must differ"):
        sess.midpoint_line_planes("bad", "axis", "z10", "z10")


def test_reduction_session_rename_and_remove():
    from cloudet.reduction import ReductionSession

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
    sess.intersect_line_plane("hit", "beam_axis", "target")

    sess.rename("tracker_left", "left_wall")
    assert "left_wall" in sess.ids()
    assert "tracker_left" not in sess.ids()
    recipe = sess.to_recipe()
    assert "left_wall" in recipe["faces"]
    assert recipe["construct"][0]["of"] == "left_wall"

    sess.rename("beam_axis", "axis")
    live = [s for s in sess.to_recipe()["construct"] if s["id"] == "hit"][0]
    assert live["line"] == "axis"

    gone = sess.remove("left_in")
    assert "left_in" in gone
    assert "axis" in gone
    assert "hit" in gone
    assert "front_in" in sess.ids()
    assert "left_wall" in sess.ids()
    with pytest.raises(ValueError, match="already exists"):
        sess.rename("target", "front_in")


def test_overlay_mm_survives_rename_and_drops_on_remove():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    left = Plane(np.array([1.0, 0.0, 0.0]), 50.0)
    sess.bind_scanned("tracker_left", left, group_name="G0", group_id=0)
    assert sess.overlay_mm("tracker_left") == 200.0
    sess.set_overlay_mm("tracker_left", 450.0)
    assert sess.overlay_mm("tracker_left") == 450.0
    sess.rename("tracker_left", "left_wall")
    assert "tracker_left" not in sess.display_mm
    assert sess.overlay_mm("left_wall") == 450.0
    recipe = sess.to_recipe()
    assert "display_mm" not in recipe
    sess.remove("left_wall")
    assert sess.display_mm == {}
    sess.bind_scanned("again", left, group_name="G0", group_id=0)
    sess.display_default_mm["plane"] = 80.0
    assert sess.overlay_mm("again") == 80.0
    sess.set_overlay_mm("again", 120.0)
    sess.clear_overlay_mm("again")
    assert sess.overlay_mm("again") == 80.0
    front = Plane(np.array([0.0, 1.0, 0.0]), 20.0)
    sess.bind_scanned("front", front, group_name="G1", group_id=1)
    sess.intersect_planes("axis", "again", "front")
    assert sess.overlay_width_mm("axis") == 1.0
    sess.set_overlay_width_mm("axis", 2.5)
    sess.rename("axis", "beam")
    assert sess.overlay_width_mm("beam") == 2.5
    sess.remove("beam")
    assert "beam" not in sess.display_width_mm


def test_bind_scanned_replay_updates_downstream():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    wall = Plane(np.array([1.0, 0.0, 0.0]), 0.0)
    sess.bind_scanned("wall", wall, group_name="G0", group_id=0)
    sess.offset("wall_in", "wall", 50.0)
    assert abs(sess.plane("wall_in").d - (-50.0)) < 1e-6

    wall2 = Plane(np.array([1.0, 0.0, 0.0]), 10.0)
    sess.bind_scanned("wall", wall2, group_name="G0", group_id=0)
    assert abs(sess.plane("wall_in").d - (-40.0)) < 1e-6


def test_preview_construct_step_does_not_mutate_session():
    from cloudet.reduction import ReductionSession, preview_construct_step

    sess = ReductionSession()
    wall = Plane(np.array([1.0, 0.0, 0.0]), 0.0)
    sess.bind_scanned("wall", wall, group_name="G0", group_id=0)
    preview = preview_construct_step(
        sess,
        {"id": "tmp_off", "op": "offset", "of": "wall", "distance_mm": 12.0},
    )
    assert preview.kind == "plane"
    assert preview.plane is not None
    assert abs(preview.plane.d - (-12.0)) < 1e-6
    assert "tmp_off" not in sess.ids()


def test_preview_construct_step_midpoint_segment_ends():
    from cloudet.reduction import ReductionSession, preview_construct_step

    sess = ReductionSession()
    left = Plane(np.array([1.0, 0.0, 0.0]), 50.0)
    front = Plane(np.array([0.0, 1.0, 0.0]), 20.0)
    z0 = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    z10 = Plane(np.array([0.0, 0.0, 1.0]), -10.0)
    sess.bind_scanned("left", left, group_name="G0", group_id=0)
    sess.bind_scanned("front", front, group_name="G1", group_id=1)
    sess.bind_scanned("z0", z0, group_name="G2", group_id=2)
    sess.bind_scanned("z10", z10, group_name="G3", group_id=3)
    sess.intersect_planes("axis", "left", "front")
    preview = preview_construct_step(
        sess,
        {
            "id": "mid",
            "op": "midpoint_line_planes",
            "line": "axis",
            "a": "z0",
            "b": "z10",
        },
    )
    assert preview.kind == "point"
    assert preview.segment_ends is not None
    assert np.allclose(preview.point, [-50.0, -20.0, 5.0])
    assert np.allclose(preview.segment_ends[0][2], 0.0)
    assert np.allclose(preview.segment_ends[1][2], 10.0)
    assert "mid" not in sess.ids()


def _beam_recipe():
    return {
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


def test_session_apply_recipe_roundtrip(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    recipe = _beam_recipe()
    sess = ReductionSession.from_recipe(recipe, project_dir=project)
    assert np.allclose(sess.point("beam_on_target"), [0.0, 0.0, 100.0])
    assert np.allclose(np.abs(sess.line("beam_axis").direction), [0.0, 0.0, 1.0])
    assert "tracker_left" in sess.anchors
    assert np.allclose(sess.anchors["tracker_left"], [0.0, 0.0, 0.0])

    out = sess.to_recipe()
    assert out["faces"]["tracker_left"]["name"] == "tracker_left"
    assert out["construct"][0]["op"] == "offset"
    assert "left_in" in out["export"]

    sess2 = ReductionSession.from_recipe(out, project_dir=project)
    assert np.allclose(sess2.point("beam_on_target"), sess.point("beam_on_target"))
    assert sess2.ids() == sess.ids()


def test_session_apply_recipe_keeps_existing_on_failure(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=project)
    before = list(sess.ids())
    bad = {
        "version": 1,
        "units": "mm",
        "faces": {"target": {"from": "group", "name": "target"}},
        "construct": [{"id": "x", "op": "nope"}],
    }
    with pytest.raises(ValueError, match="unknown op"):
        sess.apply_recipe(bad, project_dir=project)
    assert sess.ids() == before


def test_replace_construct_step_rebuilds_dependents(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=project)
    sess.offset("extra", "target", 5.0)
    extra_before = sess.plane("extra").as_array().copy()
    sess.set_overlay_mm("beam_axis", 123.0)
    sess.visible["beam_on_target"] = False
    mid = sess.add_measure({
        "id": "to_left",
        "op": "distance_point_plane",
        "point": "beam_on_target",
        "plane": "tracker_left",
    })
    assert sess.evaluate_measure(sess.measures[0])["value"] == pytest.approx(50.0)

    sess.replace_construct_step(
        "left_in",
        {"id": "left_in", "op": "offset", "of": "tracker_left", "distance_mm": 40.0},
    )

    assert sess.ids() == [
        "tracker_left",
        "tracker_front",
        "target",
        "left_in",
        "front_in",
        "beam_axis",
        "beam_on_target",
        "extra",
    ]
    assert sess.construct_step("left_in")["distance_mm"] == 40.0
    assert np.allclose(sess.point("beam_on_target"), [-10.0, 0.0, 100.0])
    assert np.allclose(sess.plane("extra").as_array(), extra_before)
    assert sess.overlay_mm("beam_axis") == pytest.approx(123.0)
    assert sess.visible["beam_on_target"] is False
    assert sess.measures[0]["id"] == mid
    assert sess.evaluate_measure(sess.measures[0])["value"] == pytest.approx(40.0)


def test_replace_construct_step_rejects_later_operand(tmp_path):
    from cloudet.reduction import ReductionSession

    sess = ReductionSession.from_recipe(
        _beam_recipe(), project_dir=_make_project(tmp_path)
    )
    before = sess.to_recipe()
    with pytest.raises(ValueError, match="operand not available"):
        sess.replace_construct_step(
            "left_in",
            {"id": "left_in", "op": "offset", "of": "beam_axis", "distance_mm": 1.0},
        )
    with pytest.raises(ValueError, match="cannot reference itself"):
        sess.replace_construct_step(
            "left_in",
            {"id": "left_in", "op": "offset", "of": "left_in", "distance_mm": 1.0},
        )
    with pytest.raises(ValueError, match="scanned face"):
        sess.replace_construct_step(
            "tracker_left",
            {"id": "tracker_left", "op": "offset", "of": "target", "distance_mm": 1.0},
        )
    assert sess.to_recipe()["construct"] == before["construct"]
    assert np.allclose(sess.point("beam_on_target"), [0.0, 0.0, 100.0])


def test_replace_construct_step_rejects_op_change(tmp_path):
    from cloudet.reduction import ReductionSession

    sess = ReductionSession.from_recipe(
        _beam_recipe(), project_dir=_make_project(tmp_path)
    )
    with pytest.raises(ValueError, match="cannot change op"):
        sess.replace_construct_step(
            "left_in",
            {
                "id": "left_in",
                "op": "intersect_planes",
                "a": "tracker_left",
                "b": "tracker_front",
            },
        )
    assert sess.kind_of("left_in") == "plane"


def test_replace_construct_step_rollback_on_failure(tmp_path):
    from cloudet.reduction import ReductionSession

    sess = ReductionSession.from_recipe(
        _beam_recipe(), project_dir=_make_project(tmp_path)
    )
    sess.set_overlay_mm("beam_axis", 50.0)
    before_ids = list(sess.ids())
    before_point = sess.point("beam_on_target").copy()
    before_step = sess.construct_step("left_in")

    with pytest.raises(ValueError, match="parallel"):
        sess.replace_construct_step(
            "left_in",
            {
                "id": "left_in",
                "op": "offset",
                "of": "tracker_front",
                "distance_mm": 20.0,
            },
        )

    assert sess.ids() == before_ids
    assert sess.construct_step("left_in") == before_step
    assert np.allclose(sess.point("beam_on_target"), before_point)
    assert sess.overlay_mm("beam_axis") == pytest.approx(50.0)


def test_session_apply_recipe_missing_face(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    recipe = {
        "version": 1,
        "units": "mm",
        "faces": {"nope": {"from": "group", "name": "does_not_exist"}},
    }
    with pytest.raises(KeyError, match="does_not_exist"):
        ReductionSession.from_recipe(recipe, project_dir=project)


def test_apply_recipe_rejects_geometry_json():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    with pytest.raises(ValueError, match="geometry.json"):
        sess.apply_recipe(
            {"version": 1, "units": "mm", "planes": {"a": {}}},
            project_dir=".",
        )


def test_apply_recipe_bind_face_fallback(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    seen: list[str] = []

    def bind(alias, spec):
        seen.append(alias)
        return None

    sess = ReductionSession.from_recipe(
        _beam_recipe(), project_dir=project, bind_face=bind
    )
    assert seen == ["tracker_left", "tracker_front", "target"]
    assert "beam_on_target" in sess.ids()


def test_recipe_frame_roundtrip(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": True,
    }
    sess = ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))
    assert sess.frame_spec == {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": True,
    }
    out = sess.to_recipe()
    assert out["frame"] == sess.frame_spec
    assert all(s.get("op") != "frame" for s in out["construct"])

    sess2 = ReductionSession.from_recipe(out, project_dir=_make_project(tmp_path))
    assert sess2.frame_spec == sess.frame_spec
    assert np.allclose(sess2.point("beam_on_target"), sess.point("beam_on_target"))
    # export uses rigid_frame() from frame_spec (Align Z view pose not required).
    survey = sess.to_result()
    assert survey.aligned is None
    assert sess.rigid_frame() is not None
    from cloudet.reduction import export_reduction_result

    aligned = export_reduction_result(
        sess, source_project=str(tmp_path), aligned_frame=sess.rigid_frame()
    )
    assert aligned.aligned is not None
    assert aligned.frame is not None
    assert "beam_on_target" in aligned.aligned["points"]


def test_recipe_frame_yaw_roundtrip(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["construct"].append(
        {
            "id": "horiz",
            "op": "intersect_planes",
            "a": "tracker_left",
            "b": "target",
        }
    )
    recipe["frame"] = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
        "yaw_line": "horiz",
        "yaw_to": "x",
    }
    sess = ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))
    assert sess.frame_spec["yaw_line"] == "horiz"
    assert sess.frame_spec["yaw_to"] == "x"
    frame = sess.rigid_frame()
    assert np.allclose(np.abs(frame.apply_direction(sess.line("beam_axis").direction)), [0.0, 0.0, 1.0])
    d = frame.apply_direction(sess.line("horiz").direction)
    assert abs(d[1]) < 1e-9
    assert abs(d[2]) < 1e-9
    assert abs(abs(d[0]) - 1.0) < 1e-9
    out = sess.to_recipe()
    assert out["frame"]["yaw_line"] == "horiz"
    assert out["frame"]["yaw_to"] == "x"
    sess.rename("horiz", "h")
    assert sess.frame_spec["yaw_line"] == "h"
    sess.remove("h")
    assert sess.frame_spec["axis"] == "beam_axis"
    assert "yaw_line" not in sess.frame_spec


def test_recipe_frame_yaw_plane_roundtrip(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "yaw_plane": "tracker_left",
        "yaw_to": "x",
    }
    sess = ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))
    assert sess.frame_spec["yaw_plane"] == "tracker_left"
    frame = sess.rigid_frame()
    n = frame.apply_direction(sess.plane("tracker_left").normal)
    assert abs(n[1]) < 1e-9
    assert abs(n[2]) < 1e-9
    assert n[0] == pytest.approx(1.0)
    out = sess.to_recipe()
    assert out["frame"]["yaw_plane"] == "tracker_left"
    assert "yaw_line" not in out["frame"]


def test_recipe_frame_yaw_line_and_plane_exclusive(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "yaw_line": "beam_axis",
        "yaw_plane": "tracker_left",
        "yaw_to": "x",
    }
    with pytest.raises(ValueError, match="not both"):
        ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))


def test_recipe_frame_yaw_to_needs_reference(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "yaw_to": "x",
    }
    with pytest.raises(ValueError, match="yaw_line or yaw_plane"):
        ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))


def test_recipe_frame_yaw_wrong_kind(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "yaw_line": "target",
        "yaw_to": "y",
    }
    with pytest.raises(ValueError, match="must be a line"):
        ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))


def test_recipe_frame_optional(tmp_path):
    from cloudet.reduction import ReductionSession

    sess = ReductionSession.from_recipe(
        _beam_recipe(), project_dir=_make_project(tmp_path)
    )
    assert sess.frame_spec is None
    assert "frame" not in sess.to_recipe()


def test_recipe_frame_invalid_object():
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = "beam_axis"
    with pytest.raises(ValueError, match="recipe.frame must be an object"):
        ReductionSession().apply_recipe(recipe, project_dir=".")


def test_recipe_frame_incomplete():
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = {"axis": "beam_axis"}
    with pytest.raises(ValueError, match="axis and origin"):
        ReductionSession().apply_recipe(recipe, project_dir=".")


def test_recipe_frame_unknown_id(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = {"axis": "nope", "origin": "beam_on_target"}
    with pytest.raises(KeyError, match="frame.axis"):
        ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))


def test_recipe_frame_wrong_kind(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = {"axis": "target", "origin": "beam_on_target"}
    with pytest.raises(ValueError, match="must be a line"):
        ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))


def test_recipe_frame_rename_and_remove():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    left = Plane(np.array([1.0, 0.0, 0.0]), 50.0)
    front = Plane(np.array([0.0, 1.0, 0.0]), 20.0)
    target = Plane(np.array([0.0, 0.0, 1.0]), -100.0)
    spare = Plane(np.array([0.0, 0.0, 1.0]), 10.0)
    sess.bind_scanned("tracker_left", left, group_name="G0", group_id=0)
    sess.bind_scanned("tracker_front", front, group_name="G1", group_id=1)
    sess.bind_scanned("target", target, group_name="G2", group_id=2)
    sess.bind_scanned("spare", spare, group_name="G3", group_id=3)
    sess.offset("left_in", "tracker_left", 50.0)
    sess.offset("front_in", "tracker_front", 20.0)
    sess.intersect_planes("beam_axis", "left_in", "front_in")
    sess.intersect_line_plane("beam_on_target", "beam_axis", "target")
    sess.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }

    sess.rename("beam_axis", "axis")
    assert sess.frame_spec["axis"] == "axis"
    sess.rename("beam_on_target", "hit")
    assert sess.frame_spec["origin"] == "hit"
    sess.remove("spare")
    assert sess.frame_spec == {"axis": "axis", "origin": "hit", "flip_z": False}
    sess.remove("axis")
    assert sess.frame_spec is None


def test_run_reduction_writes_aligned_when_recipe_has_frame(tmp_path):
    recipe = _beam_recipe()
    recipe["frame"] = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": True,
    }
    result = run_reduction(_make_project(tmp_path), recipe)
    assert np.allclose(result.points["beam_on_target"]["xyz"], [0.0, 0.0, 100.0])
    assert result.aligned is not None
    assert np.allclose(result.aligned["points"]["beam_on_target"]["xyz"], [0.0, 0.0, 0.0])
    assert np.allclose(
        np.abs(result.aligned["lines"]["beam_axis"]["direction"]), [0.0, 0.0, 1.0]
    )
    assert result.frame["kind"] == "aligned"
    assert result.frame["axis"] == "beam_axis"
    assert result.frame["flip_z"] is True
    assert result.recipe["echo"]["frame"]["axis"] == "beam_axis"


def test_reduce_cli_writes_aligned(tmp_path):
    from cloudet.cli import main

    project = _make_project(tmp_path)
    recipe = _beam_recipe()
    recipe["frame"] = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    out = tmp_path / "geometry.json"
    assert main([
        "reduce",
        str(project),
        "--recipe",
        str(recipe_path),
        "-o",
        str(out),
    ]) == 0
    doc = json.loads(out.read_text())
    assert np.allclose(doc["points"]["beam_on_target"]["xyz"], [0.0, 0.0, 100.0])
    assert np.allclose(doc["aligned"]["points"]["beam_on_target"]["xyz"], [0.0, 0.0, 0.0])
    assert np.allclose(
        np.abs(doc["aligned"]["lines"]["beam_axis"]["direction"]), [0.0, 0.0, 1.0]
    )
    assert doc["frame"]["kind"] == "aligned"
    assert doc["frame"]["origin"] == "beam_on_target"


def test_session_measures_roundtrip(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["measures"] = [
        {
            "id": "hit_height",
            "op": "distance_point_plane",
            "point": "beam_on_target",
            "plane": "target",
        },
        {
            "id": "wall_angle",
            "op": "angle_planes",
            "a": "tracker_left",
            "b": "tracker_front",
        },
    ]
    sess = ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))
    hit = sess.evaluate_measure(sess.measures[0])
    assert hit["value"] == pytest.approx(0.0)
    assert hit["unit"] == "mm"
    ang = sess.evaluate_measure(sess.measures[1])
    assert ang["value"] == pytest.approx(90.0)
    assert ang["unit"] == "deg"

    out = sess.to_recipe()
    assert out["measures"][0]["id"] == "hit_height"
    result = sess.to_result()
    assert result.measures[0]["value"] == pytest.approx(0.0)

    sess.rename("beam_on_target", "hit")
    assert sess.measures[0]["point"] == "hit"
    sess.remove("tracker_left")
    assert sess.measures == []


def test_add_measure_and_cli_export(tmp_path):
    from cloudet.reduction import ReductionSession, run_reduction

    project = _make_project(tmp_path)
    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=project)
    with pytest.raises(ValueError, match="must differ"):
        sess.add_measure({
            "id": "bad",
            "op": "distance_points",
            "a": "beam_on_target",
            "b": "beam_on_target",
        })
    mid = sess.add_measure({
        "op": "distance_point_plane",
        "point": "beam_on_target",
        "plane": "target",
    })
    assert mid == "dplane_1"
    recipe = sess.to_recipe()
    result = run_reduction(project, recipe)
    assert result.measures[0]["id"] == "dplane_1"
    assert result.measures[0]["value"] == pytest.approx(0.0)


def test_construct_plane_from_plane_point(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=project)
    pt = sess.point("beam_on_target")
    sess.plane_from_plane_point("shifted", "target", "beam_on_target")
    plane = sess.plane("shifted")
    assert np.allclose(plane.normal, sess.plane("target").normal)
    assert abs(plane.signed_distances(pt.reshape(1, 3))[0]) < 1e-9


def test_construct_plane_from_line_point(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=project)
    sess.plane_from_line_point("wall", "beam_axis", "beam_on_target")
    plane = sess.plane("wall")
    assert np.allclose(np.abs(plane.normal), [0.0, 0.0, 1.0])
    assert abs(plane.signed_distances(sess.point("beam_on_target").reshape(1, 3))[0]) < 1e-9


def test_construct_plane_from_two_lines(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    recipe = _beam_recipe()
    recipe["construct"].append({
        "id": "horiz",
        "op": "intersect_planes",
        "a": "left_in",
        "b": "target",
    })
    sess = ReductionSession.from_recipe(recipe, project_dir=project)
    sess.plane_from_two_lines("face", "beam_axis", "horiz")
    plane = sess.plane("face")
    assert abs(plane.signed_distances(sess.point("beam_on_target").reshape(1, 3))[0]) < 1e-9


def test_construct_rotate_plane_about_line(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=project)
    sess.line_from_point_normal("edge_x", "beam_on_target", "left_in")
    sess.rotate_plane_about_line("tilted", "target", "edge_x", 90.0)
    plane = sess.plane("tilted")
    assert np.allclose(plane.normal, [0.0, 1.0, 0.0], atol=1e-9)
    out = sess.to_recipe()
    step = next(s for s in out["construct"] if s["id"] == "tilted")
    assert step["op"] == "rotate_plane_about_line"
    assert step["angle_deg"] == pytest.approx(90.0)


def test_construct_rotate_about_aligned_axis(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=project)
    sess.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    assert "aligned.x" in sess.available_aligned_axis_ids()
    axis = sess.line("aligned.x")
    assert np.allclose(axis.point, sess.point("beam_on_target"))
    assert np.allclose(axis.direction, [1.0, 0.0, 0.0])
    sess.rotate_plane_about_line("tilted", "target", "aligned.x", 90.0)
    plane = sess.plane("tilted")
    assert np.allclose(np.abs(plane.normal), [0.0, 1.0, 0.0], atol=1e-9)
    step = next(s for s in sess.to_recipe()["construct"] if s["id"] == "tilted")
    assert step["line"] == "aligned.x"


def test_apply_recipe_construct_can_use_aligned_axis(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    recipe["construct"].append(
        {
            "id": "tilted",
            "op": "rotate_plane_about_line",
            "plane": "target",
            "line": "aligned.x",
            "angle_deg": 90.0,
        }
    )
    sess = ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))
    assert np.allclose(np.abs(sess.plane("tilted").normal), [0.0, 1.0, 0.0], atol=1e-9)


def test_reserved_aligned_ids_rejected(tmp_path):
    from cloudet.reduction import ReductionSession

    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=_make_project(tmp_path))
    for eid in ("aligned.x", "aligned.origin", "aligned.xy"):
        with pytest.raises(ValueError, match="reserved"):
            sess.offset(eid, "target", 1.0)


def test_construct_uses_aligned_origin_and_planes(tmp_path):
    from cloudet.reduction import ReductionSession

    project = _make_project(tmp_path)
    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=project)
    sess.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    origin = sess.point("aligned.origin")
    assert np.allclose(origin, sess.point("beam_on_target"))
    xy = sess.plane("aligned.xy")
    yz = sess.plane("aligned.yz")
    zx = sess.plane("aligned.zx")
    assert np.allclose(xy.normal, [0.0, 0.0, 1.0])
    assert np.allclose(yz.normal, [1.0, 0.0, 0.0])
    assert np.allclose(zx.normal, [0.0, 1.0, 0.0])
    assert abs(xy.signed_distances(origin.reshape(1, 3))[0]) < 1e-12
    sess.offset("above_xy", "aligned.xy", 10.0)
    assert np.allclose(sess.plane("above_xy").normal, [0.0, 0.0, 1.0])
    sess.line_from_point_normal("from_origin", "aligned.origin", "target")
    line = sess.line("from_origin")
    assert np.allclose(line.point, origin)
    assert np.allclose(np.abs(line.direction), [0.0, 0.0, 1.0])
    sess.intersect_line_plane("on_xy", "beam_axis", "aligned.xy")
    assert np.allclose(sess.point("on_xy"), origin)


def test_apply_recipe_construct_can_use_aligned_plane(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["frame"] = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    recipe["construct"].append(
        {
            "id": "above_xy",
            "op": "offset",
            "of": "aligned.xy",
            "distance_mm": 10.0,
        }
    )
    sess = ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))
    assert np.allclose(sess.plane("above_xy").normal, [0.0, 0.0, 1.0])
    origin = sess.point("aligned.origin")
    d = abs(sess.plane("above_xy").signed_distances(origin.reshape(1, 3))[0])
    assert d == pytest.approx(10.0)


def test_remove_frame_axis_drops_aligned_plane_dependents(tmp_path):
    from cloudet.reduction import ReductionSession

    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=_make_project(tmp_path))
    sess.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    sess.offset("above_xy", "aligned.xy", 10.0)
    removed = sess.remove("beam_axis")
    assert "above_xy" in removed
    assert "above_xy" not in sess.ids()
    assert sess.frame_spec is None


def test_remove_frame_axis_drops_aligned_axis_dependents(tmp_path):
    from cloudet.reduction import ReductionSession

    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=_make_project(tmp_path))
    sess.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    sess.rotate_plane_about_line("tilted", "target", "aligned.x", 90.0)
    removed = sess.remove("beam_axis")
    assert "tilted" in removed
    assert "tilted" not in sess.ids()
    assert sess.frame_spec is None


def test_replay_keeps_aligned_axis_visibility(tmp_path):
    from cloudet.reduction import ReductionSession

    sess = ReductionSession.from_recipe(
        _beam_recipe(), project_dir=_make_project(tmp_path)
    )
    sess.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    sess.visible["aligned.x"] = False
    sess.replace_construct_step(
        "left_in",
        {
            "id": "left_in",
            "op": "offset",
            "of": "tracker_left",
            "distance_mm": 51.0,
        },
    )
    assert sess.visible.get("aligned.x") is False


def test_replay_warns_when_frame_becomes_invalid():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    left = Plane(np.array([1.0, 0.0, 0.0]), 50.0)
    front = Plane(np.array([0.0, 1.0, 0.0]), 20.0)
    sess.bind_scanned("left", left, group_name="G0", group_id=0)
    sess.bind_scanned("front", front, group_name="G1", group_id=1)
    sess.intersect_planes("axis", "left", "front")
    sess.frame_spec = {"axis": "axis", "origin": "missing", "flip_z": False}
    wall = Plane(np.array([1.0, 0.0, 0.0]), 0.0)
    sess.bind_scanned("left", wall, group_name="G0", group_id=0)
    assert sess.frame_spec is None
    assert any("dropped frame" in w for w in sess.replay_warnings)


def test_replay_warns_when_measure_becomes_invalid(tmp_path):
    from cloudet.reduction import ReductionSession

    sess = ReductionSession.from_recipe(_beam_recipe(), project_dir=_make_project(tmp_path))
    sess.measures.append(
        {
            "id": "d1",
            "op": "distance_points",
            "a": "beam_on_target",
            "b": "missing_point",
        }
    )
    wall = Plane(np.array([1.0, 0.0, 0.0]), 50.0)
    sess.bind_scanned("tracker_left", wall, group_name="G0", group_id=0)
    assert not any(m["id"] == "d1" for m in sess.measures)
    assert any("dropped measure 'd1'" in w for w in sess.replay_warnings)


def test_check_recipe_rejects_unknown_construct_op():
    from cloudet.reduction.session import _check_recipe

    recipe = {
        "version": 1,
        "units": "mm",
        "faces": {"wall": {"from": "group", "name": "wall"}},
        "construct": [{"id": "x", "op": "not_real", "a": "wall"}],
    }
    with pytest.raises(ValueError, match="unknown op"):
        _check_recipe(recipe)


def test_check_recipe_rejects_duplicate_construct_id():
    from cloudet.reduction.session import _check_recipe

    recipe = {
        "version": 1,
        "units": "mm",
        "faces": {"wall": {"from": "group", "name": "wall"}},
        "construct": [
            {"id": "same", "op": "offset", "of": "wall", "distance_mm": 1.0},
            {"id": "same", "op": "offset", "of": "wall", "distance_mm": 2.0},
        ],
    }
    with pytest.raises(ValueError, match="duplicate id"):
        _check_recipe(recipe)


def test_apply_recipe_validates_construct_operand_kind(tmp_path):
    from cloudet.reduction import ReductionSession

    recipe = _beam_recipe()
    recipe["construct"].append(
        {"id": "bad", "op": "intersect_planes", "a": "beam_on_target", "b": "target"}
    )
    with pytest.raises(ValueError, match="must be a plane"):
        ReductionSession.from_recipe(recipe, project_dir=_make_project(tmp_path))


def test_build_frame_spec_yaw_exclusive():
    from cloudet.reduction import build_frame_spec, normalize_frame_spec

    spec = build_frame_spec(
        axis="axis",
        origin="origin",
        flip_z=True,
        yaw_to="x",
        yaw_kind="plane",
        yaw_ref="p1",
    )
    assert spec == {
        "axis": "axis",
        "origin": "origin",
        "flip_z": True,
        "yaw_to": "x",
        "yaw_plane": "p1",
    }
    assert "yaw_line" not in spec
    roundtrip = normalize_frame_spec(
        {
            "axis": "axis",
            "origin": "origin",
            "flip_z": False,
            "yaw_line": "line1",
            "yaw_to": "y",
        }
    )
    assert roundtrip["yaw_line"] == "line1"
    assert "yaw_plane" not in roundtrip


def test_session_rotate_point_about_line():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    top = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    front = Plane(np.array([0.0, 1.0, 0.0]), 0.0)
    sess.bind_scanned("top", top, group_name="top", group_id=0)
    sess.bind_scanned("front", front, group_name="front", group_id=1)
    side = Plane(np.array([1.0, 0.0, 0.0]), 0.0)
    sess.bind_scanned("side", side, group_name="side", group_id=2)
    sess.apply_step({"id": "p1", "op": "intersect_three_planes",
                      "a": "top", "b": "front", "c": "side"})
    sess.apply_step({"id": "ax", "op": "intersect_planes",
                      "a": "top", "b": "front"})
    sess.rotate_point_about_line("p2", "p1", "ax", 90.0)
    assert sess.kind_of("p2") == "point"
    pt = sess.point("p2")
    assert pt.shape == (3,)


def test_session_rotate_line_about_line():
    from cloudet.reduction import ReductionSession

    sess = ReductionSession()
    top = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    front = Plane(np.array([0.0, 1.0, 0.0]), 0.0)
    side = Plane(np.array([1.0, 0.0, 0.0]), 0.0)
    sess.bind_scanned("top", top, group_name="top", group_id=0)
    sess.bind_scanned("front", front, group_name="front", group_id=1)
    sess.bind_scanned("side", side, group_name="side", group_id=2)
    sess.apply_step({"id": "ax1", "op": "intersect_planes",
                      "a": "top", "b": "front"})
    sess.apply_step({"id": "ax2", "op": "intersect_planes",
                      "a": "top", "b": "side"})
    sess.rotate_line_about_line("ax3", "ax1", "ax2", 45.0)
    assert sess.kind_of("ax3") == "line"
