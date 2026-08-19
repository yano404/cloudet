"""Display-only view frame (Align Z) helpers."""

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import QComboBox

from cloudet.frame import RigidFrame, transform_record, with_aligned_copy
from cloudet.reduce import build_frame_spec
from cloudet.ui.widgets import _reset_combo


class FrameMixin:
    """Survey ↔ view coordinate transforms and Align Z controls."""

    def _to_view_points(self, xyz: np.ndarray) -> np.ndarray:
        if self._view_frame is None or xyz is None:
            return xyz
        arr = np.asarray(xyz, dtype=np.float64)
        if arr.size == 0:
            return arr
        return self._view_frame.apply_points(arr)

    def _to_view_point(self, xyz) -> np.ndarray:
        p = np.asarray(xyz, dtype=np.float64).reshape(3)
        if self._view_frame is None:
            return p
        return self._view_frame.apply_points(p)

    def _to_survey_point(self, xyz) -> np.ndarray:
        p = np.asarray(xyz, dtype=np.float64).reshape(3)
        if self._view_frame is None:
            return p
        return self._view_frame.inverse_points(p)

    def _frame_yaw_kind(self) -> str:
        if not hasattr(self, "rd_frame_yaw_kind"):
            return "line"
        kind = self.rd_frame_yaw_kind.currentData()
        return str(kind or "line")

    def _reduction_fill_frame_yaw_ref(self, *, keep: str | None = None) -> None:
        if not hasattr(self, "rd_frame_yaw_ref"):
            return
        kind = self._frame_yaw_kind()
        self._reduction_fill_combo(
            self.rd_frame_yaw_ref,
            kind=kind,
            keep=keep,
            include_aligned=False,
        )

    def _on_frame_yaw_kind_changed(self, *_args) -> None:
        keep = self._reduction_combo_id(getattr(self, "rd_frame_yaw_ref", None))
        self._reduction_fill_frame_yaw_ref(keep=keep)
        self._on_frame_combo_changed()

    def _on_frame_combo_changed(self, *_args) -> None:
        self._sync_frame_align_enabled()
        self._reduction_capture_frame_spec()
        if hasattr(self, "rd_rot_line"):
            self._reduction_refresh_operand_combos(include_frame=False)
        if hasattr(self, "rd_tree"):
            self._refresh_reduction_tree()
            self._refresh_reduction_actors()

    def _reduction_capture_frame_spec(self) -> None:
        axis = self._reduction_combo_id(getattr(self, "rd_frame_axis", None))
        origin = self._reduction_combo_id(getattr(self, "rd_frame_origin", None))
        if not axis or not origin:
            self._reduction.frame_spec = None
            return
        flip = False
        if hasattr(self, "rd_frame_flip"):
            flip = bool(self.rd_frame_flip.currentData())
        yaw_to = None
        if hasattr(self, "rd_frame_yaw_to"):
            yaw_to = self.rd_frame_yaw_to.currentData()
        yaw_ref = self._reduction_combo_id(getattr(self, "rd_frame_yaw_ref", None))
        if yaw_to and yaw_ref:
            self._reduction.frame_spec = build_frame_spec(
                axis=axis,
                origin=origin,
                flip_z=flip,
                yaw_to=str(yaw_to),
                yaw_kind=self._frame_yaw_kind(),
                yaw_ref=yaw_ref,
            )
        else:
            self._reduction.frame_spec = build_frame_spec(
                axis=axis,
                origin=origin,
                flip_z=flip,
            )

    def _reduction_restore_frame_combos(self) -> None:
        if not hasattr(self, "rd_frame_axis"):
            return
        spec = self._reduction.frame_spec
        for combo, key in (
            (self.rd_frame_axis, "axis"),
            (self.rd_frame_origin, "origin"),
        ):
            combo.blockSignals(True)
            if spec:
                idx = combo.findData(spec.get(key))
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            elif combo.count():
                combo.setCurrentIndex(0)
            combo.blockSignals(False)
        if hasattr(self, "rd_frame_flip"):
            self.rd_frame_flip.blockSignals(True)
            if spec:
                idx = self.rd_frame_flip.findData(bool(spec.get("flip_z", False)))
                self.rd_frame_flip.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                self.rd_frame_flip.setCurrentIndex(0)
            self.rd_frame_flip.blockSignals(False)
        if hasattr(self, "rd_frame_yaw_to"):
            self.rd_frame_yaw_to.blockSignals(True)
            if spec and spec.get("yaw_to"):
                idx = self.rd_frame_yaw_to.findData(spec.get("yaw_to"))
                self.rd_frame_yaw_to.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                self.rd_frame_yaw_to.setCurrentIndex(0)
            self.rd_frame_yaw_to.blockSignals(False)
        yaw_kind = "line"
        yaw_keep = None
        if spec:
            if spec.get("yaw_plane"):
                yaw_kind = "plane"
                yaw_keep = spec.get("yaw_plane")
            elif spec.get("yaw_line"):
                yaw_kind = "line"
                yaw_keep = spec.get("yaw_line")
        if hasattr(self, "rd_frame_yaw_kind"):
            self.rd_frame_yaw_kind.blockSignals(True)
            idx = self.rd_frame_yaw_kind.findData(yaw_kind)
            self.rd_frame_yaw_kind.setCurrentIndex(idx if idx >= 0 else 0)
            self.rd_frame_yaw_kind.blockSignals(False)
        if hasattr(self, "rd_frame_yaw_ref"):
            self._reduction_fill_frame_yaw_ref(keep=yaw_keep)
        self._sync_frame_align_enabled()

    def _sync_frame_align_enabled(self) -> None:
        if not hasattr(self, "rd_frame_align_btn"):
            return
        axis = self._reduction_combo_id(getattr(self, "rd_frame_axis", None))
        origin = self._reduction_combo_id(getattr(self, "rd_frame_origin", None))
        ok = bool(axis and origin)
        yaw_to = None
        if hasattr(self, "rd_frame_yaw_to"):
            yaw_to = self.rd_frame_yaw_to.currentData()
        if hasattr(self, "rd_frame_yaw_kind"):
            enabled = bool(yaw_to)
            self.rd_frame_yaw_kind.setEnabled(enabled)
        if hasattr(self, "rd_frame_yaw_ref"):
            self.rd_frame_yaw_ref.setEnabled(bool(yaw_to))
        if yaw_to:
            yaw_ref = self._reduction_combo_id(getattr(self, "rd_frame_yaw_ref", None))
            ok = ok and bool(yaw_ref)
        self.rd_frame_align_btn.setEnabled(ok)

    def _place_orientation_axes(self) -> None:
        """Lift the corner triad so the frame label fits underneath."""
        widget = getattr(self, "_axes_widget", None)
        if widget is None:
            return
        try:
            widget.SetViewport(0.0, 0.08, 0.2, 0.28)
        except Exception:
            pass

    def _refresh_frame_overlay(self) -> None:
        """Draw the current frame name under the orientation axes."""
        if not hasattr(self, "plotter"):
            return
        text = self._frame_status_text()
        try:
            self.plotter.add_text(
                text,
                name="frame_overlay",
                position=(0.02, 0.012),
                font_size=10,
                color="#333333",
                viewport=True,
                render=False,
            )
        except TypeError:
            self.plotter.add_text(
                text,
                name="frame_overlay",
                position="lower_left",
                font_size=10,
                color="#333333",
            )

    def _update_frame_controls(self) -> None:
        if hasattr(self, "rd_frame_status"):
            self.rd_frame_status.setText(self._frame_status_text())
        if hasattr(self, "rd_frame_survey_btn"):
            self.rd_frame_survey_btn.setEnabled(self._view_frame is not None)
        self._sync_frame_align_enabled()
        self._refresh_frame_overlay()

    def _refresh_aligned_view(self, *, reset_camera: bool = False) -> None:
        self._refresh_base_actor()
        self._refresh_group_actors()
        # Keep group plane n,d readable in the current view coordinate system.
        # DISPLAY-only transforms change how the plane equation should be shown.
        if hasattr(self, "tree"):
            self._refresh_tree()
        self._refresh_active_plane_bbox(render=False)
        if hasattr(self, "rd_tree"):
            self._refresh_reduction_tree()
            self._refresh_reduction_actors(render=False)
            self._reduction_update_live_preview()
        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()

    def _set_view_frame(self, frame: RigidFrame | None, *, reset_camera: bool = True) -> None:
        self._view_frame = frame
        self._update_frame_controls()
        self._rebuild_status_default()
        self._refresh_aligned_view(reset_camera=reset_camera)

    def _frame_align(self) -> None:
        self._reduction_capture_frame_spec()
        axis_id = self._reduction_combo_id(getattr(self, "rd_frame_axis", None))
        origin_id = self._reduction_combo_id(getattr(self, "rd_frame_origin", None))
        if not axis_id or not origin_id:
            raise ValueError("choose an axis and an origin")
        yaw_to = None
        if hasattr(self, "rd_frame_yaw_to"):
            yaw_to = self.rd_frame_yaw_to.currentData()
        if yaw_to and not self._reduction_combo_id(getattr(self, "rd_frame_yaw_ref", None)):
            raise ValueError("choose a line or plane for XY")
        frame = self._reduction.rigid_frame()
        if frame is None:
            raise ValueError("choose an axis and an origin")
        self._set_view_frame(frame, reset_camera=True)
        self._status(self._frame_status_text())

    def _frame_survey(self) -> None:
        if self._view_frame is None:
            return
        self._set_view_frame(None, reset_camera=True)
        self._status("frame: survey")
