"""Exact constructive-geometry tests (no noise)."""

import numpy as np
import pytest

from cloudet.geometry import (
    Line,
    intersect_line_plane,
    intersect_normal_plane,
    intersect_planes,
    intersect_three_planes,
    line_from_point_normal,
    line_from_two_points,
    midpoint_line_planes,
    offset_plane,
)
from cloudet.plane import Plane


def test_offset_plane_moves_along_normal():
    # z = 0  →  n=(0,0,1), d=0
    p = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    q = offset_plane(p, 5.0)
    # New plane should be z = 5  →  n·x + d = z - 5 = 0 → d = -5
    assert np.allclose(q.normal, [0, 0, 1])
    assert q.d == pytest.approx(-5.0)
    assert p.signed_distances(np.array([[0.0, 0.0, 5.0]]))[0] == pytest.approx(5.0)
    assert q.signed_distances(np.array([[0.0, 0.0, 5.0]]))[0] == pytest.approx(0.0)
    # Original points now sit at -distance on the offset plane.
    assert q.signed_distances(np.array([[0.0, 0.0, 0.0]]))[0] == pytest.approx(-5.0)


def test_offset_negative_distance():
    p = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    q = offset_plane(p, -12.0)
    assert q.d == pytest.approx(12.0)
    assert q.signed_distances(np.array([[0.0, 0.0, -12.0]]))[0] == pytest.approx(0.0)


def test_intersect_planes_axis_aligned():
    # x = 1 and y = 2 → line (1, 2, t) direction (0,0,1)
    p1 = Plane(np.array([1.0, 0.0, 0.0]), -1.0)  # x - 1 = 0
    p2 = Plane(np.array([0.0, 1.0, 0.0]), -2.0)  # y - 2 = 0
    line = intersect_planes(p1, p2)
    assert np.allclose(np.abs(line.direction), [0, 0, 1])
    # Point on line must satisfy both planes.
    assert abs(p1.signed_distances(line.point.reshape(1, 3))[0]) < 1e-12
    assert abs(p2.signed_distances(line.point.reshape(1, 3))[0]) < 1e-12
    assert np.allclose(line.point[:2], [1.0, 2.0])


def test_intersect_planes_parallel_raises():
    p1 = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    p2 = Plane(np.array([0.0, 0.0, 1.0]), -5.0)
    with pytest.raises(ValueError, match="parallel"):
        intersect_planes(p1, p2)


def test_intersect_line_plane_known_point():
    line = Line(np.array([1.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    plane = Plane(np.array([0.0, 0.0, 1.0]), -10.0)  # z = 10
    pt = intersect_line_plane(line, plane)
    assert np.allclose(pt, [1.0, 2.0, 10.0])


def test_intersect_line_plane_parallel_raises():
    line = Line(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    plane = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    with pytest.raises(ValueError, match="parallel"):
        intersect_line_plane(line, plane)


def test_intersect_three_planes_corner():
    # Unit cube corner at (1, 2, 3)
    px = Plane(np.array([1.0, 0.0, 0.0]), -1.0)
    py = Plane(np.array([0.0, 1.0, 0.0]), -2.0)
    pz = Plane(np.array([0.0, 0.0, 1.0]), -3.0)
    pt = intersect_three_planes(px, py, pz)
    assert np.allclose(pt, [1.0, 2.0, 3.0])


def test_intersect_three_planes_degenerate_raises():
    p1 = Plane(np.array([1.0, 0.0, 0.0]), 0.0)
    p2 = Plane(np.array([1.0, 0.0, 0.0]), -1.0)
    p3 = Plane(np.array([0.0, 1.0, 0.0]), 0.0)
    with pytest.raises(ValueError, match="unique point"):
        intersect_three_planes(p1, p2, p3)


def test_beam_axis_and_target_hit():
    """Offset tracker walls → virtual beam axis ∩ target."""
    left = Plane(np.array([1.0, 0.0, 0.0]), 50.0)   # x = -50
    front = Plane(np.array([0.0, 1.0, 0.0]), 20.0)  # y = -20
    left_in = offset_plane(left, 50.0)    # → x = 0
    front_in = offset_plane(front, 20.0)  # → y = 0
    assert abs(left_in.d) < 1e-12
    assert abs(front_in.d) < 1e-12

    axis = intersect_planes(left_in, front_in)
    assert np.allclose(np.abs(axis.direction), [0, 0, 1])
    assert np.allclose(axis.point[:2], [0.0, 0.0])

    target = Plane(np.array([0.0, 0.0, 1.0]), -100.0)  # z = 100
    hit = intersect_line_plane(axis, target)
    assert np.allclose(hit, [0.0, 0.0, 100.0])


def test_intersect_normal_plane_default_through():
    src = Plane(np.array([0.0, 0.0, 1.0]), 0.0)  # z = 0
    # Destination: x + z = 10 → after normalisation n=(1,0,1)/√2, d=-10/√2
    dst = Plane(np.array([1.0, 0.0, 1.0]), -10.0)
    # Default through = origin projected onto src = (0,0,0); ray along +z.
    # Hit when z = 10 → (0, 0, 10).
    pt = intersect_normal_plane(src, dst)
    assert np.allclose(pt, [0.0, 0.0, 10.0])

    # Parallel faces: normal ray is orthogonal to the destination → intersects.
    dst_par = Plane(np.array([0.0, 0.0, 1.0]), -7.0)  # z = 7
    pt2 = intersect_normal_plane(src, dst_par)
    assert np.allclose(pt2, [0.0, 0.0, 7.0])

    # Truly parallel ray: src normal lies in destination plane.
    src_x = Plane(np.array([1.0, 0.0, 0.0]), 0.0)  # x = 0
    dst_z = Plane(np.array([0.0, 0.0, 1.0]), -7.0)  # z = 7
    with pytest.raises(ValueError, match="parallel"):
        intersect_normal_plane(src_x, dst_z)


def test_intersect_normal_plane_explicit_through():
    src = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    dst = Plane(np.array([1.0, 0.0, 1.0]), -10.0)  # x + z = 10
    pt = intersect_normal_plane(src, dst, through=np.array([3.0, 4.0, 0.0]))
    assert np.allclose(pt, [3.0, 4.0, 7.0])  # 3 + z = 10 → z = 7


def test_line_from_point_normal():
    plane = Plane(np.array([0.0, 0.0, 1.0]), -100.0)  # z = 100
    through = np.array([12.0, -3.0, 50.0])
    line = line_from_point_normal(through, plane)
    assert np.allclose(line.point, through)
    assert np.allclose(np.abs(line.direction), [0.0, 0.0, 1.0])
    # Same direction from a raw vector.
    line2 = line_from_point_normal(through, np.array([0.0, 0.0, 2.0]))
    assert np.allclose(np.abs(line2.direction), [0.0, 0.0, 1.0])


def test_line_from_two_points():
    a = np.array([10.0, 20.0, 0.0])
    b = np.array([10.0, 20.0, 50.0])
    line = line_from_two_points(a, b)
    assert np.allclose(line.point, a)
    assert np.allclose(np.abs(line.direction), [0.0, 0.0, 1.0])
    # Both points lie on the line.
    t = (b - line.point) @ line.direction
    assert np.allclose(line.point_at(t), b)
    with pytest.raises(ValueError, match="coincide"):
        line_from_two_points(a, a)


def test_midpoint_line_planes():
    line = Line(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    z0 = Plane(np.array([0.0, 0.0, 1.0]), 0.0)     # z = 0
    z10 = Plane(np.array([0.0, 0.0, 1.0]), -10.0)  # z = 10
    mid = midpoint_line_planes(line, z0, z10)
    assert np.allclose(mid, [0.0, 0.0, 5.0])
    # Order of planes does not matter.
    assert np.allclose(midpoint_line_planes(line, z10, z0), mid)
    with pytest.raises(ValueError, match="parallel"):
        midpoint_line_planes(
            line,
            Plane(np.array([1.0, 0.0, 0.0]), 0.0),
            z10,
        )


def test_plane_patch_and_line_segment_helpers():
    from cloudet.geometry import line_segment_points, plane_patch_corners

    p = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
    corners = plane_patch_corners(p, center=np.array([10.0, 20.0, 5.0]), size_mm=100.0)
    assert corners.shape == (4, 3)
    assert np.allclose(corners[:, 2], 0.0)  # projected onto z=0
    line = Line(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    seg = line_segment_points(line, half_length_mm=50.0)
    assert np.allclose(seg, [[0, 0, -50], [0, 0, 50]])
