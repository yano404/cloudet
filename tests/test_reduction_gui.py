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
from PySide6.QtWidgets import QApplication, QComboBox, QMainWindow

from cloudet.frame import ALIGNED_AXIS_IDS
from cloudet.plane import Plane
from cloudet.reduce import ReductionSession, export_reduction_result
from cloudet.reduction_ops import MEASURE_MENU_ITEMS
from cloudet.ui.app_common import AppCommonMixin
from cloudet.ui.frame_mixin import FrameMixin
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


def test_frame_axis_combo_omits_aligned_axes(gui):
    gui._reduction.frame_spec = {
        "axis": "beam_axis",
        "origin": "beam_on_target",
        "flip_z": False,
    }
    gui._reduction_fill_combo(
        gui.rd_frame_axis, kind="line", include_aligned_axes=False
    )
    ids = {x for x in _combo_data_ids(gui.rd_frame_axis) if x}
    assert not ids.intersection(set(ALIGNED_AXIS_IDS))


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
