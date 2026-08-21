"""Shared helpers for CloudetAppWindow (status, guard, settings help)."""

from __future__ import annotations

import traceback
from datetime import datetime

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFont, QHelpEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QToolTip, QWidget

from cloudet.core.array_backend import (
    cupy_unavailable_reason,
    device_name,
    resolve_compute_backend,
)
from cloudet.ui.constants import SETTINGS_HELP_DEFAULT


class AppCommonMixin:
    """Status bar, error guard, settings help event filter."""

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self, lambda: self._guard(self._save_all))
        QShortcut(QKeySequence("F"), self, lambda: self._guard(self._fit_active))
        QShortcut(QKeySequence("Backspace"), self, lambda: self._guard(self._delete_active))
        rd_del = QShortcut(QKeySequence("Delete"), self.rd_tree)
        rd_del.setContext(Qt.WidgetWithChildrenShortcut)
        rd_del.activated.connect(
            lambda: self._guard(self._reduction_delete_selected, busy=False)
        )
        if hasattr(self, "rd_measure_tree"):
            meas_del = QShortcut(QKeySequence("Delete"), self.rd_measure_tree)
            meas_del.setContext(Qt.WidgetWithChildrenShortcut)
            meas_del.activated.connect(
                lambda: self._guard(self._reduction_delete_measures, busy=False)
            )
        QShortcut(QKeySequence("M"), self, self.append_cb.toggle)
        QShortcut(QKeySequence("V"), self, self.solo_cb.toggle)

        # Depth cycling rides VTK's key path rather than Qt's: the render
        # widget keeps focus while picking, and there P (bound by
        # enable_point_picking) reaches us but Qt shortcuts do not.
        for keysym, delta in (("greater", +1), (">", +1),
                              ("less", -1), ("<", -1)):
            self.plotter.add_key_event(
                keysym,
                lambda d=delta: self._guard(lambda: self._cycle_pick_depth(d)),
            )
        # Escape clears an in-progress cylinder 3-point seed (VTK has focus).
        self.plotter.add_key_event(
            "Escape",
            lambda: self._guard(self._clear_cylinder_seeds, busy=False),
        )
        QShortcut(
            QKeySequence(Qt.Key_Escape),
            self,
            lambda: self._guard(self._clear_cylinder_seeds, busy=False),
        )

    def _register_settings_help(self, widget: QWidget, html: str):
        """Show setting help inside the window instead of a native tooltip."""
        self._settings_help_targets[widget] = html
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        tree = getattr(self, "tree", None)
        if tree is not None and watched is tree.viewport():
            if event.type() == QEvent.ToolTip and isinstance(event, QHelpEvent):
                item = tree.itemAt(event.pos())
                if item is not None:
                    data = item.data(0, Qt.UserRole)
                    if data and data[0] == "plane":
                        tip = item.text(1).strip()
                        if tip:
                            QToolTip.showText(event.globalPos(), tip, tree)
                            return True
                QToolTip.hideText()
                return True
            if event.type() == QEvent.Leave:
                QToolTip.hideText()
        help_text = self._settings_help_targets.get(watched)
        if help_text is not None and hasattr(self, "settings_help_label"):
            if event.type() in (QEvent.Enter, QEvent.FocusIn):
                self.settings_help_label.setText(help_text)
            elif event.type() in (QEvent.Leave, QEvent.FocusOut):
                self.settings_help_label.setText(SETTINGS_HELP_DEFAULT)
        return super().eventFilter(watched, event)

    def _compute_status_suffix(self) -> str:
        try:
            resolved = resolve_compute_backend(self.settings.detection.compute_backend)
        except ImportError as e:
            return f"compute: error ({e})"
        if resolved == "cupy":
            name = device_name() or "CUDA"
            return f"compute: cupy ({name})"
        reason = cupy_unavailable_reason()
        if reason and self.settings.detection.compute_backend in ("auto", "cupy"):
            short = reason.splitlines()[0]
            if len(short) > 80:
                short = short[:77] + "..."
            return f"compute: numpy ({short})"
        return "compute: numpy"

    def _frame_status_text(self) -> str:
        fr = self._view_frame
        if fr is None:
            return "frame: survey"
        axis = fr.axis_id or "?"
        origin = fr.origin_id or "?"
        extra = ", flip" if fr.flip_z else ""
        if fr.yaw_id and fr.yaw_to:
            tag = "n:" if fr.yaw_kind == "plane" else ""
            extra += f", {tag}{fr.yaw_id}→{fr.yaw_to.upper()}"
        return f"frame: aligned ({axis}, {origin}{extra})"

    def _rebuild_status_default(self) -> None:
        ready = "Ready"
        ready += f"  |  {self._compute_status_suffix()}"
        ready += f"  |  {self._frame_status_text()}"
        if getattr(self, "_vtk_log_path", None) is not None:
            ready += f"  |  VTK messages -> {self._vtk_log_path}"
        ready += f"  |  fit timing -> {self._fit_log_path}"
        self._status_default = ready

    def _status(self, msg):
        self.statusBar().showMessage(str(msg)) if hasattr(self, "statusBar") else print(msg)

    def _append_fit_log(self, message: str) -> None:
        """Append one line to project ``fit.log`` (fit timings and breakdown)."""
        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self._fit_log_path, "a", encoding="utf-8") as f:
                f.write(f"{stamp}  {message}\n")
        except OSError:
            pass

    def _format_fit_log_line(self, timing: dict, *, kind: str = "fit") -> str:
        plane_bits = " | ".join(
            f"p{p['plane_index']}:{p['status']} n={p['n_points']:,} "
            f"mad={p['mad_sigma_mm'] * 1e3:.0f}um"
            + (" BIMODAL" if p.get("bimodal") else "")
            for p in timing.get("planes") or []
        )
        mode = "multi" if timing.get("multi") else "single"
        parts = [
            f"{kind}  {timing['group']}  n_pts={timing['n_pts']:,}  "
            f"compute={timing['compute']}  ransac={timing.get('ransac_backend', '?')}  "
            f"mode={mode}",
        ]
        if timing.get("depth_s") is not None:
            parts.append(f"depth={timing['depth_s']:.3f}s")
        if timing.get("pick_s") is not None:
            parts.append(f"pick={timing['pick_s']:.3f}s")
        detail = timing.get("pick_detail") or {}
        if detail:
            parts.append(
                "pick_detail="
                + ",".join(
                    f"{k}={detail[k]:.3f}s"
                    for k in (
                        "neighbor_s",
                        "grid_build_s",
                        "neighbor_query_s",
                        "local_fit_s",
                        "progressive_s",
                        "accumulate_s",
                        "accumulate_dist_s",
                        "accumulate_lsq_s",
                        "accumulate_connect_s",
                    )
                    if k in detail
                )
            )
            if "n_candidates" in detail:
                parts.append(f"n_candidates={detail['n_candidates']:,}")
            if "n_neighbors" in detail:
                parts.append(f"n_neighbors={detail['n_neighbors']:,}")
        parts.append(f"fit={timing['fit_s']:.3f}s")
        parts.append(f"uv={timing['uv_s']:.3f}s")
        if timing.get("post_s") is not None:
            parts.append(f"post={timing['post_s']:.3f}s")
        if timing.get("wall_s") is not None:
            parts.append(f"wall={timing['wall_s']:.3f}s")
        else:
            parts.append(f"total={timing['total_s']:.3f}s")
        parts.append(plane_bits)
        return "  ".join(parts)

    def _fit_timing_status(self, timing: dict) -> str:
        if timing.get("wall_s") is not None:
            bits = []
            if timing.get("depth_s") is not None and timing["depth_s"] > 0.005:
                bits.append(f"depth {timing['depth_s']:.2f}s")
            if timing.get("pick_s") is not None:
                bits.append(f"pick {timing['pick_s']:.2f}s")
            bits.append(f"fit {timing['fit_s']:.2f}s")
            bits.append(f"uv {timing['uv_s']:.2f}s")
            if timing.get("post_s") is not None:
                bits.append(f"ui {timing['post_s']:.2f}s")
            return " + ".join(bits) + f" = {timing['wall_s']:.2f}s"
        return (
            f"fit {timing['fit_s']:.2f}s + uv {timing['uv_s']:.2f}s "
            f"= {timing['total_s']:.2f}s"
        )

    def _log_fit_timing(self, timing: dict, *, kind: str = "fit") -> None:
        self._append_fit_log(self._format_fit_log_line(timing, kind=kind))

    def _set_settings_dirty(self, dirty: bool):
        self._settings_dirty = bool(dirty)
        if not hasattr(self, "settings_dirty_label"):
            return
        if dirty:
            self.settings_dirty_label.setText("Unsaved changes")
            self.settings_dirty_label.setObjectName("badgeWarn")
            self.apply_btn.setEnabled(True)
        else:
            self.settings_dirty_label.setText("Saved")
            self.settings_dirty_label.setObjectName("muted")
        self.settings_dirty_label.style().unpolish(self.settings_dirty_label)
        self.settings_dirty_label.style().polish(self.settings_dirty_label)

    def _update_source_meta(self):
        if not hasattr(self, "source_meta_label"):
            return
        if len(self.full_points) == 0:
            self.source_meta_label.setText("0 pts loaded")
        else:
            self.source_meta_label.setText(
                f"{len(self.full_points):,} pts loaded  |  {self._n_displayed:,} displayed"
            )

    def _sync_action_states(self):
        if not hasattr(self, "fit_btn"):
            return
        has_groups = bool(self.groups)
        active = self._active_group() is not None
        selected = len(self._selected_group_ids()) if hasattr(self, "tree") else 0
        self.fit_btn.setEnabled(active)
        self.fit_all_btn.setEnabled(has_groups)
        self.merge_btn.setEnabled(selected >= 2)
        self.delete_btn.setEnabled(active)
        self.clear_btn.setEnabled(has_groups)
        self.save_all_btn.setEnabled(has_groups)
        self.group_count_label.setText(f"{len(self.groups)} groups")

    def _guard(self, fn, *, busy: bool = True):
        try:
            if busy:
                QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                fn()
            finally:
                if busy:
                    QApplication.restoreOverrideCursor()
        except Exception as e:
            traceback.print_exc()
            self._status(f"error: {e}")

    def _get_group(self, gid):
        for g in self.groups:
            if g["id"] == gid:
                return g
        return None

    def _active_group(self):
        return self._get_group(self.active_group_id) if self.active_group_id is not None else None
