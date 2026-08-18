"""View-frame alignment: map a chosen axis to global +Z."""

import numpy as np
import pytest

from cloudet.frame import (
    RigidFrame,
    result_in_frame,
    rotation_mapping_to_z,
    transform_record,
    with_aligned_copy,
)
from cloudet.geometry import Line
from cloudet.plane import Plane
from cloudet.reduce import ReductionSession


def test_rotation_x_axis_to_z():
    r = rotation_mapping_to_z([1.0, 0.0, 0.0])
    assert np.allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0])
    # Smallest rotation keeps +Y.
    assert np.allclose(r @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose(r @ r.T, np.eye(3))
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_rotation_already_z_is_identity():
    r = rotation_mapping_to_z([0.0, 0.0, 4.0])
    assert np.allclose(r, np.eye(3))


def test_rotation_antiparallel_z():
    r = rotation_mapping_to_z([0.0, 0.0, -1.0])
    assert np.allclose(r @ np.array([0.0, 0.0, -1.0]), [0.0, 0.0, 1.0])
    assert np.allclose(r @ r.T, np.eye(3))
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_align_z_moves_origin_and_axis():
    origin = np.array([10.0, 20.0, 30.0])
    direction = np.array([0.0, 1.0, 0.0])
    frame = RigidFrame.align_z(direction, origin, axis_id="ax", origin_id="o")
    assert np.allclose(frame.apply_points(origin), [0.0, 0.0, 0.0])
    along = origin + 5.0 * direction / np.linalg.norm(direction)
    p = frame.apply_points(along)
    assert abs(p[0]) < 1e-12
    assert abs(p[1]) < 1e-12
    assert p[2] == pytest.approx(5.0)
    back = frame.inverse_points(p)
    assert np.allclose(back, along)


def test_flip_z_reverses_axis():
    origin = np.zeros(3)
    frame = RigidFrame.align_z([0.0, 0.0, 1.0], origin, flip_z=True)
    p = frame.apply_points([0.0, 0.0, 4.0])
    assert np.allclose(p, [0.0, 0.0, -4.0])


def test_yaw_line_to_plus_x():
    origin = np.zeros(3)
    frame = RigidFrame.align_z(
        [0.0, 0.0, 1.0],
        origin,
        yaw_direction=[0.0, 1.0, 0.0],
        yaw_to="x",
        yaw_id="horiz",
    )
    assert np.allclose(frame.apply_direction([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
    assert np.allclose(frame.apply_direction([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0])
    assert frame.yaw_id == "horiz"
    assert frame.yaw_to == "x"
    assert frame.to_dict()["yaw_line"] == "horiz"


def test_yaw_line_to_plus_y():
    origin = np.zeros(3)
    frame = RigidFrame.align_z(
        [0.0, 0.0, 1.0],
        origin,
        yaw_direction=[1.0, 0.0, 0.0],
        yaw_to="y",
    )
    assert np.allclose(frame.apply_direction([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose(frame.apply_direction([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])


def test_yaw_plane_normal_to_plus_x():
    origin = np.zeros(3)
    frame = RigidFrame.align_z(
        [0.0, 0.0, 1.0],
        origin,
        yaw_direction=[1.0, 0.0, 0.0],
        yaw_to="x",
        yaw_id="wall",
        yaw_kind="plane",
    )
    assert frame.yaw_kind == "plane"
    assert np.allclose(frame.apply_direction([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
    assert np.allclose(frame.apply_direction([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
    assert frame.to_dict()["yaw_plane"] == "wall"


def test_yaw_plane_parallel_to_z_errors():
    with pytest.raises(ValueError, match="parallel to Z"):
        RigidFrame.align_z(
            [0.0, 0.0, 1.0],
            np.zeros(3),
            yaw_direction=[0.0, 0.0, 1.0],
            yaw_to="x",
            yaw_kind="plane",
        )


def test_yaw_keeps_axis_on_z_when_axis_was_x():
    origin = np.array([1.0, 2.0, 3.0])
    frame = RigidFrame.align_z(
        [1.0, 0.0, 0.0],
        origin,
        yaw_direction=[0.0, 1.0, 0.0],
        yaw_to="y",
    )
    assert np.allclose(frame.apply_direction([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0])
    d = frame.apply_direction([0.0, 1.0, 0.0])
    assert abs(d[0]) < 1e-12
    assert d[1] == pytest.approx(1.0)
    assert abs(d[2]) < 1e-12


def test_plane_incidence_preserved():
    plane = Plane(np.array([1.0, 0.0, 0.0]), -5.0)  # x = 5
    origin = np.array([5.0, 1.0, 2.0])
    frame = RigidFrame.align_z([1.0, 0.0, 0.0], origin)
    pts = np.array([[5.0, 0.0, 0.0], [5.0, 3.0, -1.0], [5.0, -2.0, 8.0]])
    assert np.allclose(plane.signed_distances(pts), 0.0)
    plane2 = frame.apply_plane(plane)
    pts2 = frame.apply_points(pts)
    assert np.allclose(plane2.signed_distances(pts2), 0.0, atol=1e-12)


def test_line_through_origin_becomes_z_axis():
    origin = np.array([1.0, 2.0, 3.0])
    line = Line(origin, np.array([2.0, 0.0, 0.0]))
    frame = RigidFrame.align_z(line.direction, origin)
    line2 = frame.apply_line(line)
    assert np.allclose(np.abs(line2.direction), [0.0, 0.0, 1.0])
    # A point on the original line maps onto the z-axis.
    p = frame.apply_points(origin + 7.0 * line.direction)
    assert abs(p[0]) < 1e-12
    assert abs(p[1]) < 1e-12


def test_transform_record_and_result_in_frame():
    sess = ReductionSession()
    sess.bind_scanned(
        "left", Plane(np.array([1.0, 0.0, 0.0]), 0.0), group_name="G0", group_id=0
    )
    sess.bind_scanned(
        "front", Plane(np.array([0.0, 1.0, 0.0]), 0.0), group_name="G1", group_id=1
    )
    sess.bind_scanned(
        "target", Plane(np.array([0.0, 0.0, 1.0]), -10.0), group_name="G2", group_id=2
    )
    sess.intersect_planes("axis", "left", "front")
    sess.intersect_line_plane("hit", "axis", "target")

    origin = np.array(sess.point("hit"), dtype=np.float64)
    direction = np.array(sess.line("axis").direction, dtype=np.float64)
    frame = RigidFrame.align_z(direction, origin, axis_id="axis", origin_id="hit")

    rec = transform_record("point", sess.record_of("hit"), frame)
    assert np.allclose(rec["xyz"], [0.0, 0.0, 0.0])

    survey = sess.to_result(source_project="/tmp/proj")
    aligned = result_in_frame(survey, frame)
    assert aligned.frame["kind"] == "aligned"
    assert aligned.frame["axis"] == "axis"
    assert np.allclose(aligned.points["hit"]["xyz"], [0.0, 0.0, 0.0])
    assert np.allclose(np.abs(aligned.lines["axis"]["direction"]), [0.0, 0.0, 1.0])
    # Recipe echo and the live session stay in survey coordinates.
    assert aligned.recipe["echo"]["construct"][-1]["id"] == "hit"
    assert "frame" not in survey.to_dict()
    assert aligned.to_dict()["frame"]["origin"] == "hit"
    assert np.allclose(sess.point("hit"), [0.0, 0.0, 10.0])
    assert np.allclose(survey.points["hit"]["xyz"], [0.0, 0.0, 10.0])


def test_with_aligned_copy_keeps_survey_and_adds_aligned():
    sess = ReductionSession()
    sess.bind_scanned(
        "left", Plane(np.array([1.0, 0.0, 0.0]), 0.0), group_name="G0", group_id=0
    )
    sess.bind_scanned(
        "front", Plane(np.array([0.0, 1.0, 0.0]), 0.0), group_name="G1", group_id=1
    )
    sess.bind_scanned(
        "target", Plane(np.array([0.0, 0.0, 1.0]), -10.0), group_name="G2", group_id=2
    )
    sess.intersect_planes("axis", "left", "front")
    sess.intersect_line_plane("hit", "axis", "target")
    survey = sess.to_result(source_project="/tmp/proj")
    frame = RigidFrame.align_z(
        sess.line("axis").direction,
        sess.point("hit"),
        axis_id="axis",
        origin_id="hit",
    )
    both = with_aligned_copy(survey, frame)
    doc = both.to_dict()
    assert np.allclose(doc["points"]["hit"]["xyz"], [0.0, 0.0, 10.0])
    assert np.allclose(doc["aligned"]["points"]["hit"]["xyz"], [0.0, 0.0, 0.0])
    assert np.allclose(np.abs(doc["aligned"]["lines"]["axis"]["direction"]), [0.0, 0.0, 1.0])
    assert doc["frame"]["kind"] == "aligned"
    assert doc["frame"]["origin"] == "hit"
    assert "aligned" not in survey.to_dict()


def test_transform_record_rewrites_segment_ends():
    origin = np.array([1.0, 2.0, 3.0])
    frame = RigidFrame.align_z([0.0, 0.0, 1.0], origin)
    rec = transform_record(
        "point",
        {
            "xyz": origin.tolist(),
            "ends": [(origin + [0, 0, -4]).tolist(), (origin + [0, 0, 4]).tolist()],
            "through": origin.tolist(),
            "provenance": "constructed",
        },
        frame,
    )
    assert np.allclose(rec["xyz"], [0.0, 0.0, 0.0])
    assert np.allclose(rec["through"], [0.0, 0.0, 0.0])
    assert np.allclose(rec["ends"], [[0.0, 0.0, -4.0], [0.0, 0.0, 4.0]])
    assert rec["provenance"] == "constructed"
