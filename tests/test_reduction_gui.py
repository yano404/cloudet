"""Reduction GUI helpers: aligned combos, update mode, export."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("PySide6")

if "pyvista" not in sys.modules:
    sys.modules["pyvista"] = MagicMock()

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QMainWindow

from cloudet.plane import Plane
from cloudet.reduce import ReductionSession, export_reduction_result
from cloudet.reduction_ops import GUI_PAGE_INDEX, MEASURE_MENU_ITEMS, REDUCTION_OPS
from cloudet.ui.app_common import AppCommonMixin
from cloudet.ui.frame_mixin import FrameMixin
from cloudet.ui.constants import RD_ALIGNED
from cloudet.ui.reduction_mixin import ReductionMixin


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


def _combo_data_ids(combo: QComboBox) -> list[str | None]:
    return [combo.itemData(i) for i in range(combo.count())]


def _beam_session() -> ReductionSession:
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
    return sess


class _ReductionGuiHarness(QMainWindow, AppCommonMixin, FrameMixin, ReductionMixin):
    """Minimal window with Reduction/Frame widgets, no 3D view."""

    def __init__(self, session: ReductionSession | None = None):
        super().__init__()
        self.project_dir = Path("/tmp/cloudet-reduction-gui-test")
        self._reduction = session or _beam_session()
        self._rd_loading_step = False
        self._rd_form_entity_id = None
        self._view_frame = None
        self.groups: list[dict] = []
        self.status_messages: list[str] = []
        self._reduction_actor_names: list[str] = []
        self._reduction_measure_actor_names: list[str] = []
        self._rd_offset_sync = False
        self._status_default = "Ready"
        self._build_reduction_dock()

    def _status(self, message: str) -> None:
        self.status_messages.append(message)

    def _guard(self, fn, *, busy: bool = True):
        return fn()

    def _refresh_frame_overlay(self) -> None:
        return

    def _refresh_reduction_actors(self, *, render: bool = True) -> None:
        return

    def _reduction_update_live_preview(self) -> None:
        return

    def _reduction_update_selection_label(self) -> None:
        return

    def _refresh_reduction_entity_overlays(self, _ids) -> None:
        return

    def _clear_reduction_preview(self) -> None:
        return

    def _reduction_refresh_view(self) -> None:
        return

    def _sync_reduction_entity_actions(self) -> None:
        return

    def _reduction_sync_size_controls_from_selection(self) -> None:
        return

    def _set_combo(self, combo: QComboBox, eid: str | None) -> None:
        idx = combo.findData(eid) if eid else -1
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _select_tree_entity(self, eid: str) -> None:
        self._refresh_reduction_tree()
        for i in range(self.rd_tree.topLevelItemCount()):
            item = self.rd_tree.topLevelItem(i)
            if str(item.data(0, Qt.UserRole)) == eid:
                self.rd_tree.setCurrentItem(item)
                item.setSelected(True)
                break

    def _build_export_result(self):
        self._reduction_capture_frame_spec()
        want_aligned = bool(
            getattr(self, "rd_export_frame_cb", None)
            and self.rd_export_frame_cb.isChecked()
        )
        aligned_frame = self._reduction.rigid_frame() if want_aligned else None
        return export_reduction_result(
            self._reduction,
            source_project=str(self.project_dir),
            aligned_frame=aligned_frame,
        )


@pytest.fixture
def gui(qapp, tmp_path):
    harness = _ReductionGuiHarness()
    harness.project_dir = tmp_path
    yield harness
    harness.close()


def test_line_combo_includes_aligned_axes_when_frame_set(gui):
    gui._reduction.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    gui._reduction_fill_combo(gui.rd_rot_line, kind="line")
    ids = _combo_data_ids(gui.rd_rot_line)
    assert "aligned.x" in ids
    assert "aligned.y" in ids
    assert "aligned.z" in ids


def test_plane_and_point_combos_include_aligned_when_frame_set(gui):
    gui._reduction.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    gui._reduction_fill_combo(gui.rd_rot_plane, kind="plane")
    plane_ids = _combo_data_ids(gui.rd_rot_plane)
    assert "aligned.xy" in plane_ids
    assert "aligned.yz" in plane_ids
    assert "aligned.zx" in plane_ids
    combo = QComboBox()
    gui._reduction_fill_combo(combo, kind="point")
    point_ids = _combo_data_ids(combo)
    assert "aligned.origin" in point_ids


def test_frame_axis_combo_omits_aligned_axes(gui):
    gui._reduction.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    gui._reduction_fill_combo(
        gui.rd_frame_axis, kind="line", include_aligned=False
    )
    ids = {x for x in _combo_data_ids(gui.rd_frame_axis) if x}
    assert not ids.intersection({"aligned.x", "aligned.y", "aligned.z"})
    gui._reduction_fill_combo(
        gui.rd_frame_origin, kind="point", include_aligned=False
    )
    origin_ids = {x for x in _combo_data_ids(gui.rd_frame_origin) if x}
    assert "aligned.origin" not in origin_ids
    gui._reduction_fill_combo(
        gui.rd_frame_yaw_ref, kind="plane", include_aligned=False
    )
    yaw_ids = {x for x in _combo_data_ids(gui.rd_frame_yaw_ref) if x}
    assert not yaw_ids.intersection({"aligned.xy", "aligned.yz", "aligned.zx"})


def test_update_mode_operand_allowlist(gui):
    gui._select_tree_entity("left_in")
    gui.rd_mode_update.setChecked(True)
    allowed = gui._reduction_operand_allowlist()
    assert allowed is not None
    assert "tracker_left" in allowed
    assert "left_in" not in allowed
    assert "beam_axis" not in allowed
    assert "beam_on_target" not in allowed


def test_gui_form_step_roundtrip(gui):
    step = gui._reduction.construct_step("left_in")
    assert step is not None
    gui._reduction_load_step_into_form(step)
    roundtrip = gui._reduction_step_from_form("left_in")
    assert roundtrip["op"] == step["op"]
    assert roundtrip["of"] == step["of"]
    assert float(roundtrip["distance_mm"]) == float(step["distance_mm"])


def test_update_mode_sync_apply_button(gui):
    gui._select_tree_entity("beam_axis")
    gui._reduction_sync_operation_from_selection()
    assert gui.rd_apply_btn.text() == "Update"
    assert gui.rd_mode_update.isChecked()


def test_export_aligned_without_view_frame(gui, tmp_path):
    gui._set_combo(gui.rd_frame_axis, "beam_axis")
    gui._set_combo(gui.rd_frame_origin, "beam_on_target")
    gui.rd_export_frame_cb.setChecked(True)
    assert gui._view_frame is None
    result = gui._build_export_result()
    assert result.aligned is not None
    assert result.frame is not None
    assert "beam_on_target" in result.aligned["points"]


def test_export_survey_only_when_aligned_checkbox_off(gui):
    gui._set_combo(gui.rd_frame_axis, "beam_axis")
    gui._set_combo(gui.rd_frame_origin, "beam_on_target")
    gui.rd_export_frame_cb.setChecked(False)
    result = gui._build_export_result()
    assert result.aligned is None


def test_capture_frame_spec_yaw_exclusive(gui):
    gui._set_combo(gui.rd_frame_axis, "beam_axis")
    gui._set_combo(gui.rd_frame_origin, "beam_on_target")
    gui._set_combo(gui.rd_frame_yaw_to, "x")
    gui.rd_frame_yaw_kind.setCurrentIndex(gui.rd_frame_yaw_kind.findData("plane"))
    gui._set_combo(gui.rd_frame_yaw_ref, "target")
    gui._reduction_capture_frame_spec()
    spec = gui._reduction.frame_spec
    assert spec is not None
    assert spec.get("yaw_plane") == "target"
    assert "yaw_line" not in spec
    gui.rd_frame_yaw_kind.setCurrentIndex(gui.rd_frame_yaw_kind.findData("line"))
    gui._set_combo(gui.rd_frame_yaw_ref, "beam_axis")
    gui._reduction_capture_frame_spec()
    spec = gui._reduction.frame_spec
    assert spec.get("yaw_line") == "beam_axis"
    assert "yaw_plane" not in spec


def test_replay_warnings_surface_in_status(gui):
    gui._reduction.frame_spec = {
        "axis": "beam_axis",
        "origin": "missing_origin",
        "flip_z": False,
    }
    wall = Plane(np.array([1.0, 0.0, 0.0]), 10.0)
    gui._reduction.bind_scanned("tracker_left", wall, group_name="G0", group_id=0)
    gui._reduction_status_with_replay("imported wall")
    assert gui.status_messages
    assert "dropped frame" in gui.status_messages[-1]


def test_measure_op_combo_uses_metadata(gui):
    gui._build_measure_dock()
    labels = [gui.rd_measure_op.itemText(i) for i in range(gui.rd_measure_op.count())]
    expected = [label for label, _key in MEASURE_MENU_ITEMS]
    assert labels == expected


def test_normal_plane_form_roundtrip(gui):
    far = Plane(np.array([0.0, 0.0, 1.0]), -200.0)
    gui._reduction.bind_scanned("far", far, group_name="G9", group_id=9)
    gui._reduction.intersect_normal_plane("hit", "target", "far")
    gui._reduction_refresh_operand_combos()
    step = gui._reduction.construct_step("hit")
    gui._reduction_load_step_into_form(step)
    assert gui._reduction_current_op() == "intersect_normal_plane"
    roundtrip = gui._reduction_step_from_form("hit")
    assert roundtrip["src"] == "target"
    assert roundtrip["dst"] == "far"
    assert gui._reduction_combo_id(gui.rd_np_src) == "target"
    assert gui._reduction_combo_id(gui.rd_np_dst) == "far"


def test_update_mode_locks_op_combo(gui):
    gui._select_tree_entity("left_in")
    gui._reduction_sync_operation_from_selection()
    assert not gui.rd_op_combo.isEnabled()
    gui.rd_mode_new.setChecked(True)
    gui._reduction_sync_apply_button()
    assert gui.rd_op_combo.isEnabled()


def _tree_ids(gui) -> list[str]:
    return [
        str(gui.rd_tree.topLevelItem(i).data(0, Qt.UserRole))
        for i in range(gui.rd_tree.topLevelItemCount())
    ]


def test_aligned_axes_absent_from_tree_without_frame(gui):
    gui._refresh_reduction_tree()
    ids = _tree_ids(gui)
    assert "aligned.x" not in ids
    assert "aligned.origin" not in ids
    assert gui._reduction_aligned_entity_ids() == []


def test_aligned_entities_appear_at_tree_bottom_when_frame_set(gui):
    gui._reduction.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    gui._refresh_reduction_tree()
    ids = _tree_ids(gui)
    assert ids[-7:] == [
        "aligned.origin",
        "aligned.x",
        "aligned.y",
        "aligned.z",
        "aligned.yz",
        "aligned.zx",
        "aligned.xy",
    ]
    origin_item = gui.rd_tree.topLevelItem(gui.rd_tree.topLevelItemCount() - 7)
    assert origin_item.text(0) == "aligned origin"
    assert origin_item.text(1) == "origin"
    item = gui.rd_tree.topLevelItem(gui.rd_tree.topLevelItemCount() - 6)
    assert item.text(0) == "aligned X axis"
    assert item.text(1) == "axis"
    assert item.text(2) == "—"
    assert item.checkState(0) == Qt.Checked
    assert not (item.flags() & Qt.ItemIsEditable)
    assert item.flags() & Qt.ItemIsUserCheckable
    assert item.foreground(1).color().name() == QColor(RD_ALIGNED["aligned.x"]).name()
    plane_item = gui.rd_tree.topLevelItem(gui.rd_tree.topLevelItemCount() - 1)
    assert plane_item.text(0) == "aligned XY plane"
    assert plane_item.text(1) == "plane"


def test_selecting_aligned_axis_does_not_enter_update_mode(gui):
    gui._reduction.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    gui._select_tree_entity("aligned.x")
    assert gui._reduction_editing_id() is None
    gui._reduction_sync_operation_from_selection()
    assert gui.rd_mode_new.isChecked()
    assert gui.rd_apply_btn.text() != "Update"


def test_aligned_entity_cannot_be_deleted(gui):
    gui._reduction.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    gui._select_tree_entity("aligned.origin")
    with pytest.raises(ValueError, match="cannot be renamed or deleted"):
        gui._reduction_delete_selected()
    assert "aligned.origin" in gui._reduction.available_aligned_ids()


def test_operation_pages_follow_reduction_ops(gui):
    assert gui.rd_stack.count() == 1 + len(REDUCTION_OPS)
    for op in REDUCTION_OPS:
        page = getattr(gui, op.operands[0].widget).parentWidget()
        assert gui.rd_stack.indexOf(page) == GUI_PAGE_INDEX[op.gui_key]
        for field in op.operands:
            widget = getattr(gui, field.widget)
            assert isinstance(widget, QComboBox)
        for field in op.scalars:
            widget = getattr(gui, field.widget)
            assert isinstance(widget, QDoubleSpinBox)
            assert widget.suffix() == field.suffix
            assert widget.value() == pytest.approx(field.default)
    assert hasattr(gui, "rd_offset_slider")
    gui.rd_op_combo.setCurrentIndex(gui.rd_op_combo.findData("rotate_plane_about_line"))
    assert gui.rd_stack.currentIndex() == GUI_PAGE_INDEX["rotate_plane_about_line"]
