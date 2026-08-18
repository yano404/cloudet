"""Shared Reduction operation metadata.

Centralizes GUI labels, recipe op names, construct operand schemas, and
measure definitions so form↔step translation stays in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperandField:
    step_key: str
    kind: str  # plane | line | point
    widget: str


@dataclass(frozen=True)
class ScalarField:
    step_key: str
    widget: str


@dataclass(frozen=True)
class ReductionOpDef:
    gui_key: str
    recipe_op: str
    menu_label: str
    apply_label: str
    page_index: int
    id_prefix: str
    result_kind: str  # plane | line | point
    operands: tuple[OperandField, ...] = ()
    scalars: tuple[ScalarField, ...] = ()
    missing_msg: str = ""
    operands_must_differ: bool = False


@dataclass(frozen=True)
class MeasureOperandDef:
    key: str
    kind: str  # plane | line | point
    label: str


@dataclass(frozen=True)
class MeasureOpDef:
    op: str
    menu_label: str
    operands: tuple[MeasureOperandDef, MeasureOperandDef]


REDUCTION_OPS: tuple[ReductionOpDef, ...] = (
    ReductionOpDef(
        "offset",
        "offset",
        "Offset plane",
        "Apply offset",
        1,
        "offset",
        "plane",
        operands=(OperandField("of", "plane", "rd_offset_plane"),),
        scalars=(ScalarField("distance_mm", "rd_offset_spin"),),
        missing_msg="Offset needs a plane",
    ),
    ReductionOpDef(
        "intersect_planes",
        "intersect_planes",
        "Intersect 2 planes → axis",
        "Create axis",
        2,
        "axis",
        "line",
        operands=(
            OperandField("a", "plane", "rd_p2_a"),
            OperandField("b", "plane", "rd_p2_b"),
        ),
        missing_msg="Intersect planes needs 2 planes",
        operands_must_differ=True,
    ),
    ReductionOpDef(
        "intersect_line_plane",
        "intersect_line_plane",
        "Line ∩ plane → point",
        "Create point",
        3,
        "point",
        "point",
        operands=(
            OperandField("line", "line", "rd_lp_line"),
            OperandField("plane", "plane", "rd_lp_plane"),
        ),
        missing_msg="Line ∩ plane needs a line and a plane",
    ),
    ReductionOpDef(
        "intersect_three",
        "intersect_three_planes",
        "3 planes → corner",
        "Create corner",
        4,
        "corner",
        "point",
        operands=(
            OperandField("a", "plane", "rd_p3_a"),
            OperandField("b", "plane", "rd_p3_b"),
            OperandField("c", "plane", "rd_p3_c"),
        ),
        missing_msg="3 planes → point needs 3 planes",
    ),
    ReductionOpDef(
        "line_from_point_normal",
        "line_from_point_normal",
        "Point + normal → axis",
        "Create axis",
        5,
        "axis",
        "line",
        operands=(
            OperandField("point", "point", "rd_pn_point"),
            OperandField("plane", "plane", "rd_pn_plane"),
        ),
        missing_msg="Point + normal needs a point and a plane",
    ),
    ReductionOpDef(
        "line_from_two_points",
        "line_from_two_points",
        "2 points → axis",
        "Create axis",
        6,
        "axis",
        "line",
        operands=(
            OperandField("a", "point", "rd_pp_a"),
            OperandField("b", "point", "rd_pp_b"),
        ),
        missing_msg="2 points → axis needs two points",
        operands_must_differ=True,
    ),
    ReductionOpDef(
        "midpoint_line_planes",
        "midpoint_line_planes",
        "Line ∩ 2 planes → midpoint",
        "Create midpoint",
        7,
        "mid",
        "point",
        operands=(
            OperandField("line", "line", "rd_mp_line"),
            OperandField("a", "plane", "rd_mp_a"),
            OperandField("b", "plane", "rd_mp_b"),
        ),
        missing_msg="midpoint needs 1 axis and 2 planes",
        operands_must_differ=True,
    ),
    ReductionOpDef(
        "plane_from_plane_point",
        "plane_from_plane_point",
        "Plane + point → parallel plane",
        "Create plane",
        8,
        "plane",
        "plane",
        operands=(
            OperandField("plane", "plane", "rd_pp_plane"),
            OperandField("point", "point", "rd_pp_point"),
        ),
        missing_msg="plane + point → plane needs a plane and a point",
    ),
    ReductionOpDef(
        "plane_from_line_point",
        "plane_from_line_point",
        "Line + point → plane",
        "Create plane",
        9,
        "plane",
        "plane",
        operands=(
            OperandField("line", "line", "rd_lpp_line"),
            OperandField("point", "point", "rd_lpp_point"),
        ),
        missing_msg="line + point → plane needs an axis and a point",
    ),
    ReductionOpDef(
        "plane_from_two_lines",
        "plane_from_two_lines",
        "2 lines → plane",
        "Create plane",
        10,
        "plane",
        "plane",
        operands=(
            OperandField("a", "line", "rd_l2p_a"),
            OperandField("b", "line", "rd_l2p_b"),
        ),
        missing_msg="2 lines → plane needs two axes",
        operands_must_differ=True,
    ),
    ReductionOpDef(
        "rotate_plane_about_line",
        "rotate_plane_about_line",
        "Rotate plane about axis",
        "Rotate plane",
        11,
        "plane",
        "plane",
        operands=(
            OperandField("plane", "plane", "rd_rot_plane"),
            OperandField("line", "line", "rd_rot_line"),
        ),
        scalars=(ScalarField("angle_deg", "rd_rot_angle"),),
        missing_msg="rotate plane needs a plane and an axis",
    ),
    ReductionOpDef(
        "intersect_normal_plane",
        "intersect_normal_plane",
        "Normal ∩ plane → point",
        "Create point",
        12,
        "point",
        "point",
        operands=(
            OperandField("src", "plane", "rd_np_src"),
            OperandField("dst", "plane", "rd_np_dst"),
        ),
        missing_msg="Normal ∩ plane needs a source plane and a destination plane",
        operands_must_differ=True,
    ),
)

MEASURE_OPS: tuple[MeasureOpDef, ...] = (
    MeasureOpDef(
        "distance_points",
        "Distance (point - point)",
        (
            MeasureOperandDef("a", "point", "Point A"),
            MeasureOperandDef("b", "point", "Point B"),
        ),
    ),
    MeasureOpDef(
        "distance_point_plane",
        "Distance (point - plane)",
        (
            MeasureOperandDef("point", "point", "Point"),
            MeasureOperandDef("plane", "plane", "Plane"),
        ),
    ),
    MeasureOpDef(
        "distance_point_line",
        "Distance (point - line)",
        (
            MeasureOperandDef("point", "point", "Point"),
            MeasureOperandDef("line", "line", "Line"),
        ),
    ),
    MeasureOpDef(
        "angle_planes",
        "Angle (plane - plane)",
        (
            MeasureOperandDef("a", "plane", "Plane A"),
            MeasureOperandDef("b", "plane", "Plane B"),
        ),
    ),
    MeasureOpDef(
        "angle_lines",
        "Angle (line - line)",
        (
            MeasureOperandDef("a", "line", "Line A"),
            MeasureOperandDef("b", "line", "Line B"),
        ),
    ),
    MeasureOpDef(
        "angle_line_plane",
        "Angle (line - plane)",
        (
            MeasureOperandDef("line", "line", "Line"),
            MeasureOperandDef("plane", "plane", "Plane"),
        ),
    ),
)

# bind is GUI-only import action (not a construct op)
GUI_BIND_OP_KEY = "bind"
GUI_BIND_MENU_LABEL = "Import plane from Groups"
GUI_BIND_APPLY_LABEL = "Import plane"
GUI_BIND_PAGE_INDEX = 0

GUI_TO_RECIPE_OP = {op.gui_key: op.recipe_op for op in REDUCTION_OPS}
RECIPE_TO_GUI_OP = {op.recipe_op: op.gui_key for op in REDUCTION_OPS}
GUI_MENU_ITEMS = [(GUI_BIND_MENU_LABEL, GUI_BIND_OP_KEY)] + [
    (op.menu_label, op.gui_key) for op in REDUCTION_OPS
]
GUI_APPLY_LABELS = {GUI_BIND_OP_KEY: GUI_BIND_APPLY_LABEL} | {
    op.gui_key: op.apply_label for op in REDUCTION_OPS
}
GUI_PAGE_INDEX = {GUI_BIND_OP_KEY: GUI_BIND_PAGE_INDEX} | {
    op.gui_key: op.page_index for op in REDUCTION_OPS
}
GUI_ID_PREFIX = {op.gui_key: op.id_prefix for op in REDUCTION_OPS}
GUI_RESULT_KIND = {op.gui_key: op.result_kind for op in REDUCTION_OPS}

REDUCTION_OP_BY_GUI = {op.gui_key: op for op in REDUCTION_OPS}
REDUCTION_OP_BY_RECIPE = {op.recipe_op: op for op in REDUCTION_OPS}
MEASURE_OP_BY_KEY = {op.op: op for op in MEASURE_OPS}
MEASURE_MENU_ITEMS = [(op.menu_label, op.op) for op in MEASURE_OPS]
MEASURE_OPERAND_FIELDS = {
    op.op: tuple((operand.key, operand.kind) for operand in op.operands)
    for op in MEASURE_OPS
}


def build_construct_step(
    op_def: ReductionOpDef,
    entity_id: str,
    *,
    operand_values: dict[str, str | None],
    scalar_values: dict[str, float],
) -> dict:
    """Build a recipe construct step from widget-keyed operand/scalar values."""
    step: dict = {"id": str(entity_id), "op": op_def.recipe_op}
    resolved: list[str] = []
    for field in op_def.operands:
        val = operand_values.get(field.widget)
        if not val:
            raise ValueError(op_def.missing_msg or f"{op_def.gui_key} missing {field.step_key}")
        step[field.step_key] = str(val)
        resolved.append(str(val))
    if op_def.operands_must_differ and len(set(resolved)) < len(resolved):
        raise ValueError(op_def.missing_msg or "operands must differ")
    for field in op_def.scalars:
        step[field.step_key] = float(scalar_values.get(field.widget, 0.0))
    return step


def form_values_from_step(
    step: dict,
) -> tuple[str | None, dict[str, str | None], dict[str, float]]:
    """Map a construct step to GUI widget values."""
    gui_key = RECIPE_TO_GUI_OP.get(str(step.get("op") or ""))
    if gui_key is None:
        return None, {}, {}
    op_def = REDUCTION_OP_BY_GUI[gui_key]
    operands = {
        field.widget: step.get(field.step_key) for field in op_def.operands
    }
    scalars = {
        field.widget: float(step.get(field.step_key, 0.0)) for field in op_def.scalars
    }
    return gui_key, operands, scalars
