"""Form↔step metadata in reduction_ops."""

from __future__ import annotations

import pytest

from cloudet.reduction_ops import (
    REDUCTION_OP_BY_GUI,
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
