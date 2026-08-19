"""Form↔step metadata in reduction_ops."""

from __future__ import annotations

import pytest

from cloudet.reduction_ops import (
    GUI_BIND_PAGE_INDEX,
    GUI_PAGE_INDEX,
    REDUCTION_OP_BY_GUI,
    REDUCTION_OPS,
    build_construct_step,
    form_values_from_step,
)


def test_build_construct_step_offset():
    op = REDUCTION_OP_BY_GUI["offset"]
    step = build_construct_step(
        op,
        "wall_in",
        operand_values={"rd_offset_plane": "wall"},
        scalar_values={"rd_offset_spin": 12.5},
    )
    assert step == {
        "id": "wall_in",
        "op": "offset",
        "of": "wall",
        "distance_mm": 12.5,
    }


def test_build_construct_step_requires_operands():
    op = REDUCTION_OP_BY_GUI["intersect_planes"]
    with pytest.raises(ValueError, match="2 planes"):
        build_construct_step(
            op,
            "axis",
            operand_values={"rd_p2_a": "a", "rd_p2_b": None},
            scalar_values={},
        )


def test_build_construct_step_distinct_operands():
    op = REDUCTION_OP_BY_GUI["intersect_planes"]
    with pytest.raises(ValueError, match="2 planes"):
        build_construct_step(
            op,
            "axis",
            operand_values={"rd_p2_a": "same", "rd_p2_b": "same"},
            scalar_values={},
        )


def test_form_values_from_step_roundtrip():
    op = REDUCTION_OP_BY_GUI["rotate_plane_about_line"]
    step = build_construct_step(
        op,
        "rot",
        operand_values={"rd_rot_plane": "p1", "rd_rot_line": "axis"},
        scalar_values={"rd_rot_angle": 45.0},
    )
    gui_key, operands, scalars = form_values_from_step(step)
    assert gui_key == "rotate_plane_about_line"
    assert operands == {"rd_rot_plane": "p1", "rd_rot_line": "axis"}
    assert scalars == {"rd_rot_angle": 45.0}


def test_form_values_from_step_unknown_op():
    gui_key, operands, scalars = form_values_from_step({"op": "unknown", "id": "x"})
    assert gui_key is None
    assert operands == {}
    assert scalars == {}


def test_build_construct_step_intersect_normal_plane():
    op = REDUCTION_OP_BY_GUI["intersect_normal_plane"]
    step = build_construct_step(
        op,
        "hit",
        operand_values={"rd_np_src": "wall", "rd_np_dst": "target"},
        scalar_values={},
    )
    assert step == {
        "id": "hit",
        "op": "intersect_normal_plane",
        "src": "wall",
        "dst": "target",
    }
    gui_key, operands, scalars = form_values_from_step(step)
    assert gui_key == "intersect_normal_plane"
    assert operands == {"rd_np_src": "wall", "rd_np_dst": "target"}
    assert scalars == {}


def test_page_index_follows_tuple_order():
    assert GUI_PAGE_INDEX["bind"] == GUI_BIND_PAGE_INDEX == 0
    for i, op in enumerate(REDUCTION_OPS, start=1):
        assert GUI_PAGE_INDEX[op.gui_key] == i
        assert op.operands
        for field in op.operands:
            assert field.label
            assert field.widget.startswith("rd_")
