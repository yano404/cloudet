"""Shared Reduction operation metadata.

This module centralizes operation definitions used by the GUI and recipe bridge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReductionOpDef:
    gui_key: str
    recipe_op: str
    menu_label: str
    apply_label: str
    page_index: int
    id_prefix: str
    result_kind: str  # plane | line | point


REDUCTION_OPS: tuple[ReductionOpDef, ...] = (
    ReductionOpDef("offset", "offset", "Offset plane", "Apply offset", 1, "offset", "plane"),
    ReductionOpDef("intersect_planes", "intersect_planes", "Intersect 2 planes → axis", "Create axis", 2, "axis", "line"),
    ReductionOpDef("intersect_line_plane", "intersect_line_plane", "Line ∩ plane → point", "Create point", 3, "point", "point"),
    ReductionOpDef("intersect_three", "intersect_three_planes", "3 planes → corner", "Create corner", 4, "corner", "point"),
    ReductionOpDef("line_from_point_normal", "line_from_point_normal", "Point + normal → axis", "Create axis", 5, "axis", "line"),
    ReductionOpDef("line_from_two_points", "line_from_two_points", "2 points → axis", "Create axis", 6, "axis", "line"),
    ReductionOpDef("midpoint_line_planes", "midpoint_line_planes", "Line ∩ 2 planes → midpoint", "Create midpoint", 7, "mid", "point"),
    ReductionOpDef("plane_from_plane_point", "plane_from_plane_point", "Plane + point → parallel plane", "Create plane", 8, "plane", "plane"),
    ReductionOpDef("plane_from_line_point", "plane_from_line_point", "Line + point → plane", "Create plane", 9, "plane", "plane"),
    ReductionOpDef("plane_from_two_lines", "plane_from_two_lines", "2 lines → plane", "Create plane", 10, "plane", "plane"),
    ReductionOpDef("rotate_plane_about_line", "rotate_plane_about_line", "Rotate plane about axis", "Rotate plane", 11, "plane", "plane"),
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
