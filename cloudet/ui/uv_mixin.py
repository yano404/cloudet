"""Residual u–v map dock and histogram panel."""

from __future__ import annotations

import time

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cloudet.array_backend import resolve_compute_backend
from cloudet.mainplane import MainPlaneParams, extract_main_plane
from cloudet.multiplane import _bimodality_flag
from cloudet.pipeline import residual_uv_map
from cloudet.plane import Plane, mad_sigma
from cloudet.ui.constants import FIT_MAX_THRESHOLD_MM
from cloudet.ui.plane_labels import _plane_id_token, _plane_label
from cloudet.ui.uv_plot import _UVSelectViewBox, _rdbu_r_colormap
from cloudet.ui.widgets import UI_STYLE, _make_collapsible_card


class UvMixin:
    """Residual map, histogram, and selection refit UI."""

    def _build_uv_dock(self):
        dock = QDockWidget("Residuals", self)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        dock.setStyleSheet(UI_STYLE)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)

        title = QLabel("RESIDUALS")
        title.setObjectName("sectionTitle")
        lay.addWidget(title)

        self.uv_meta_label = QLabel(
            "Fit a plane to see the u–v map and residual histogram."
        )
        self.uv_meta_label.setObjectName("muted")
        self.uv_meta_label.setWordWrap(True)
        lay.addWidget(self.uv_meta_label)

        def _mini_card(heading: str, *, expanded: bool = True) -> tuple[QFrame, QVBoxLayout]:
            return _make_collapsible_card(heading, expanded=expanded)

        # ---- Display + Map view (side by side) --------------------------
        disp_card, disp_lay = _mini_card("DISPLAY")
        disp_form = QFormLayout()
        disp_form.setContentsMargins(0, 0, 0, 0)
        disp_form.setSpacing(4)
        disp_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.uv_bins_spin = QSpinBox()
        self.uv_bins_spin.setRange(40, 512)
        self.uv_bins_spin.setSingleStep(20)
        self.uv_bins_spin.setValue(200)
        self.uv_bins_spin.setToolTip(
            "Number of bins along each u–v axis (N → N×N grid)."
        )
        disp_form.addRow("u–v bins", self.uv_bins_spin)
        self.uv_range_spin = QDoubleSpinBox()
        self.uv_range_spin.setRange(50.0, 2000.0)
        self.uv_range_spin.setDecimals(0)
        self.uv_range_spin.setSingleStep(50.0)
        self.uv_range_spin.setValue(FIT_MAX_THRESHOLD_MM * 1e3)
        self.uv_range_spin.setSuffix(" µm")
        self.uv_range_spin.setToolTip(
            "Colorbar and histogram half-range (±). Does not change the fitted plane."
        )
        disp_form.addRow("range ±", self.uv_range_spin)
        disp_lay.addLayout(disp_form)
        disp_btn_row = QHBoxLayout()
        self.uv_apply_display_btn = QPushButton("Apply display")
        self.uv_apply_display_btn.setObjectName("secondaryBtn")
        self.uv_apply_display_btn.setToolTip(
            "Rebuild the u–v map and histogram with the bins / range above."
        )
        self.uv_apply_display_btn.clicked.connect(
            lambda: self._guard(self._apply_uv_display_settings)
        )
        disp_btn_row.addWidget(self.uv_apply_display_btn)
        disp_btn_row.addStretch(1)
        disp_lay.addLayout(disp_btn_row)

        map_card, map_lay = _mini_card("MAP VIEW")
        map_hint = QLabel("Wheel zoom / drag pan.")
        map_hint.setObjectName("muted")
        map_hint.setWordWrap(True)
        map_lay.addWidget(map_hint)
        map_row = QHBoxLayout()
        self.uv_map_base_btn = QPushButton("Base fit")
        self.uv_map_base_btn.setObjectName("secondaryBtn")
        self.uv_map_base_btn.setCheckable(True)
        self.uv_map_base_btn.setChecked(True)
        self.uv_map_base_btn.setToolTip("Show residual u–v map for the base fit.")
        self.uv_map_refit_btn = QPushButton("Selection refit")
        self.uv_map_refit_btn.setObjectName("secondaryBtn")
        self.uv_map_refit_btn.setCheckable(True)
        self.uv_map_refit_btn.setEnabled(False)
        self.uv_map_refit_btn.setToolTip(
            "Show residual u–v map for the selection refit (same u–v frame; "
            "outside the rectangle is empty)."
        )
        self.uv_map_mode_group = QButtonGroup(self)
        self.uv_map_mode_group.setExclusive(True)
        self.uv_map_mode_group.addButton(self.uv_map_base_btn)
        self.uv_map_mode_group.addButton(self.uv_map_refit_btn)
        self.uv_map_base_btn.clicked.connect(lambda: self._set_uv_map_mode("base"))
        self.uv_map_refit_btn.clicked.connect(lambda: self._set_uv_map_mode("refit"))
        map_row.addWidget(self.uv_map_base_btn)
        map_row.addWidget(self.uv_map_refit_btn)
        map_lay.addLayout(map_row)
        map_lay.addStretch(1)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        top_row.addWidget(disp_card, 1)
        top_row.addWidget(map_card, 1)
        lay.addLayout(top_row)

        # ---- Selection controls -----------------------------------------
        sel_card, sel_lay = _mini_card("SELECTION")
        sel_hint = QLabel(
            "Cmd/Ctrl+drag on the map to select; handles resize the rectangle."
        )
        sel_hint.setObjectName("muted")
        sel_hint.setWordWrap(True)
        sel_lay.addWidget(sel_hint)
        sel_btn_row = QHBoxLayout()
        self.uv_refit_btn = QPushButton("Refit selection")
        self.uv_refit_btn.setObjectName("primaryBtn")
        self.uv_refit_btn.setEnabled(False)
        self.uv_refit_btn.setToolTip(
            "Fit a plane on the selected u–v rectangle and add it as the next "
            "plane (p1, p2, …) on this group. The original plane is kept. "
            "Import the new plane into Reduction as G6_p1, etc."
        )
        self.uv_refit_btn.clicked.connect(
            lambda: self._guard(self._refit_uv_selection)
        )
        sel_btn_row.addWidget(self.uv_refit_btn)
        self.uv_clear_refit_btn = QPushButton("Clear refit")
        self.uv_clear_refit_btn.setObjectName("secondaryBtn")
        self.uv_clear_refit_btn.setEnabled(False)
        self.uv_clear_refit_btn.setToolTip(
            "Remove the last plane added by Refit selection (p1, p2, …). "
            "The original plane and the u–v rectangle stay."
        )
        self.uv_clear_refit_btn.clicked.connect(self._clear_uv_refit)
        sel_btn_row.addWidget(self.uv_clear_refit_btn)
        self.uv_clear_rect_btn = QPushButton("Clear selection")
        self.uv_clear_rect_btn.setObjectName("secondaryBtn")
        self.uv_clear_rect_btn.setEnabled(False)
        self.uv_clear_rect_btn.setToolTip(
            "Clear the u–v rectangle (and any selection refit)."
        )
        self.uv_clear_rect_btn.clicked.connect(self._clear_uv_rect)
        sel_btn_row.addWidget(self.uv_clear_rect_btn)
        sel_lay.addLayout(sel_btn_row)
        lay.addWidget(sel_card)

        try:
            import pyqtgraph as pg

            pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
            self._uv_cmap = _rdbu_r_colormap()
            self._uv_glw = pg.GraphicsLayoutWidget()
            self._uv_glw.setBackground("#f7f7f5")
            self._uv_glw.setMinimumHeight(420)

            vb = _UVSelectViewBox.create(self._apply_uv_rect)
            self._uv_plot = self._uv_glw.addPlot(
                row=0, col=0, viewBox=vb, title="u–v map"
            )
            self._uv_plot.setLabel("bottom", "u (mm)")
            self._uv_plot.setLabel("left", "v (mm)")
            self._uv_plot.showGrid(x=True, y=True, alpha=0.15)
            self._uv_img = pg.ImageItem(axisOrder="row-major")
            self._uv_plot.addItem(self._uv_img)
            vlim0 = float(self._uv_vlim_mm()) * 1e3
            self._uv_cbar = pg.ColorBarItem(
                values=(-vlim0, vlim0),
                colorMap=self._uv_cmap,
                label="µm",
                interactive=False,
                width=14,
            )
            self._uv_glw.addItem(self._uv_cbar, row=0, col=1)
            self._uv_cbar.setImageItem(self._uv_img)

            self._hist_plot = self._uv_glw.addPlot(row=1, col=0, colspan=2)
            self._hist_plot.setLabel("bottom", "residual", units="µm")
            self._hist_plot.setLabel("left", "count")
            self._hist_plot.showGrid(x=True, y=True, alpha=0.15)
            self._uv_glw.ci.layout.setRowStretchFactor(0, 3)
            self._uv_glw.ci.layout.setRowStretchFactor(1, 2)
            self._uv_glw.ci.layout.setColumnStretchFactor(0, 10)
            self._uv_glw.ci.layout.setColumnStretchFactor(1, 1)

            lay.addWidget(self._uv_glw, stretch=1)
            self._clear_uv_plot()
        except Exception as e:
            self._uv_glw = None
            self._uv_plot = None
            self._hist_plot = None
            self._uv_img = None
            err = QLabel(f"pyqtgraph unavailable:\n{e}")
            err.setObjectName("muted")
            err.setWordWrap(True)
            lay.addWidget(err)
            lay.addStretch(1)

        dock.setWidget(w)
        dock.setMinimumWidth(360)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.uv_dock = dock
    def _uv_bins_value(self) -> int:
        if hasattr(self, "uv_bins_spin"):
            return int(self.uv_bins_spin.value())
        return 200

    def _uv_vlim_mm(self) -> float:
        if hasattr(self, "uv_range_spin"):
            return max(float(self.uv_range_spin.value()) * 1e-3, 1e-4)
        return float(FIT_MAX_THRESHOLD_MM)

    def _apply_uv_display_settings(self):
        """Rebuild residual plots using the current bins / range controls."""
        g = self._active_group()
        if g is None or g.get("fit") is None:
            raise ValueError("no fitted active group")
        p = self._active_plane_entry()
        if p is None:
            raise ValueError("no active plane")
        for key in ("uv", "uv_samples", "uv_local_idx", "residual_hist"):
            p.pop(key, None)
        rf = p.get("selection_refit")
        if rf is not None:
            rf.pop("uv", None)
            rf.pop("residual_hist", None)
        bins = self._uv_bins_value()
        thr = self._uv_vlim_mm()
        self._status(
            f"rebuilding residual display ({bins}² bins, ±{thr*1e3:.0f} µm) ..."
        )
        QApplication.processEvents()
        # Rebuild base u–v cache first so selection-refit can reuse the grid.
        self._show_uv_for_selection()
        p = self._active_plane_entry()
        if p is not None and p.get("selection_refit") is not None and self._uv_rect:
            samples = self._ensure_uv_samples(g, p)
            rf = p["selection_refit"]
            if samples is not None and self.full_points.size and len(g.get("indices", [])):
                u0, u1, v0, v1 = self._uv_rect
                sel = (
                    (samples["u"] >= u0)
                    & (samples["u"] <= u1)
                    & (samples["v"] >= v0)
                    & (samples["v"] <= v1)
                )
                local = np.asarray(samples["local_idx"][sel], dtype=np.int64)
                pts = self.full_points[g["indices"]][local]
                plane = Plane.from_array(rf["abcd"])
                r_sel = plane.signed_distances(pts)
                rf["residual_hist"] = self._residual_hist_from_r(
                    r_sel, threshold_mm=thr, mad_mm=float(rf["mad_sigma_mm"])
                )
                self._rebuild_selection_refit_uv(p)
                self._show_uv_for_selection()
        self._status(f"display updated: {bins}² bins, ±{thr*1e3:.0f} µm")

    def _cache_uv_for_plane(
        self,
        pts: np.ndarray,
        plane_entry: dict,
        mask: np.ndarray | None,
        *,
        lock_basis: dict | None = None,
    ):
        plane = Plane.from_array(plane_entry["abcd"])
        bins = self._uv_bins_value()
        basis = lock_basis if lock_basis is not None else plane_entry.get("uv_basis")
        kw = {}
        if (
            basis is not None
            and basis.get("u") is not None
            and basis.get("v") is not None
        ):
            kw["u_axis"] = basis["u"]
            kw["v_axis"] = basis["v"]
            if basis.get("center") is not None:
                kw["center"] = basis["center"]
        # Map + hist only; per-point u/v samples are built lazily on selection.
        uv = residual_uv_map(
            pts,
            plane,
            mask=mask,
            bins=bins,
            return_points=False,
            compute_backend=self.settings.detection.compute_backend,
            **kw,
        )
        mad = float(plane_entry["mad_sigma_mm"])
        threshold = float(self._uv_vlim_mm())
        plane_entry["threshold_mm"] = float(
            plane_entry.get("threshold_mm", FIT_MAX_THRESHOLD_MM)
        )
        if mask is None:
            local_idx = np.arange(len(pts), dtype=np.int32)
        else:
            local_idx = np.flatnonzero(mask).astype(np.int32)
        lo, hi = uv["extents_uvn"]
        plane_entry["uv"] = {
            "mean": uv["mean"],
            "counts": uv["counts"],
            "u_edges": uv["u_edges"],
            "v_edges": uv["v_edges"],
            "vlim_mm": threshold,
            "n_used": int(uv["n_used"]),
            "bins": bins,
        }
        plane_entry["uv_basis"] = {
            "u": np.asarray(uv["u_axis"], dtype=np.float64),
            "v": np.asarray(uv["v_axis"], dtype=np.float64),
            "center": np.asarray(uv["center"], dtype=np.float64),
            "lo": np.asarray(lo, dtype=np.float64),
            "hi": np.asarray(hi, dtype=np.float64),
            "basis": "minrect",
        }
        plane_entry["uv_local_idx"] = local_idx
        plane_entry.pop("uv_samples", None)
        plane_entry["residual_hist"] = self._residual_hist_from_r(
            uv["r"], threshold_mm=threshold, mad_mm=mad
        )

    def _ensure_uv_samples(self, g: dict, plane_entry: dict) -> dict | None:
        """Build per-point u/v/r samples on demand (selection / refit)."""
        samples = plane_entry.get("uv_samples")
        if (
            samples is not None
            and "local_idx" in samples
            and samples.get("basis") == "minrect"
        ):
            return samples
        local = plane_entry.get("uv_local_idx")
        basis = plane_entry.get("uv_basis")
        if local is None or len(local) == 0:
            if self._ensure_plane_uv(g, plane_entry) is None:
                return None
            local = plane_entry.get("uv_local_idx")
            basis = plane_entry.get("uv_basis")
        if local is None or len(local) == 0 or self.full_points.size == 0:
            return None
        pts = np.asarray(self.full_points[g["indices"]][local], dtype=np.float64)
        plane = Plane.from_array(plane_entry["abcd"])
        if basis is not None and basis.get("basis") == "minrect":
            u = np.asarray(basis["u"], dtype=np.float64)
            v = np.asarray(basis["v"], dtype=np.float64)
            center = np.asarray(basis["center"], dtype=np.float64)
            uu = (pts - center) @ u
            vv = (pts - center) @ v
            r = plane.signed_distances(pts)
        else:
            uv = residual_uv_map(
                pts, plane, mask=None, bins=self._uv_bins_value(), return_points=True,
                compute_backend=self.settings.detection.compute_backend,
            )
            uu, vv, r = uv["u"], uv["v"], uv["r"]
        plane_entry["uv_samples"] = {
            "u": np.asarray(uu, dtype=np.float32),
            "v": np.asarray(vv, dtype=np.float32),
            "r": np.asarray(r, dtype=np.float32),
            "local_idx": np.asarray(local, dtype=np.int32),
            "basis": "minrect",
        }
        return plane_entry["uv_samples"]

    @staticmethod
    def _residual_hist_from_r(
        r: np.ndarray, *, threshold_mm: float, mad_mm: float | None = None
    ) -> dict | None:
        r = np.asarray(r, dtype=np.float64)
        if r.size == 0:
            return None
        thr = max(float(threshold_mm), 1e-4)
        counts, edges = np.histogram(r, bins=81, range=(-thr, thr))
        mad = float(mad_sigma(r)) if mad_mm is None else float(mad_mm)
        return {
            "counts": counts,
            "edges_mm": edges,
            "mad_mm": mad,
            "threshold_mm": thr,
            "n_used": int(r.size),
            "n_outside": int(np.count_nonzero(np.abs(r) > thr)),
        }

    def _hist_for_current_uv_rect(self, plane_entry: dict) -> dict | None:
        thr = float(self._uv_vlim_mm())
        refit = plane_entry.get("selection_refit")
        if refit is not None and self._uv_rect is not None:
            hist = refit.get("residual_hist")
            if hist is not None:
                return hist
        if self._uv_rect is None:
            hist = plane_entry.get("residual_hist")
            if hist is not None:
                return hist
        g = self._active_group()
        if g is None:
            return plane_entry.get("residual_hist")
        samples = self._ensure_uv_samples(g, plane_entry)
        if samples is None:
            return plane_entry.get("residual_hist")
        u = samples["u"]
        v = samples["v"]
        r = samples["r"]
        if self._uv_rect is None:
            return self._residual_hist_from_r(r, threshold_mm=thr)
        u0, u1, v0, v1 = self._uv_rect
        sel = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
        return self._residual_hist_from_r(r[sel], threshold_mm=thr)

    def _uv_hist_scope(self, plane_entry: dict | None) -> str:
        if plane_entry is not None and plane_entry.get("selection_refit") is not None:
            return "selection refit"
        if self._uv_rect is not None:
            return "selection"
        return "full face"

    def _find_plane(self, g: dict | None, plane_index: int) -> dict | None:
        if g is None or g.get("fit") is None:
            return None
        planes = g["fit"].get("planes") or []
        return next(
            (x for x in planes if int(x.get("plane_index", 0)) == int(plane_index)),
            None,
        )

    def _active_plane_entry(self) -> dict | None:
        g = self._active_group()
        if g is None or g.get("fit") is None:
            return None
        planes = g["fit"].get("planes") or []
        if not planes:
            return None
        p = self._find_plane(g, self._active_plane_index)
        return p if p is not None else planes[0]

    def _uv_rect_local_indices(self, plane_entry: dict) -> np.ndarray:
        """Group-local indices of points inside the current u–v rectangle."""
        if self._uv_rect is None:
            raise ValueError("no u–v rectangle selected")
        g = self._active_group()
        if g is None:
            raise ValueError("no active group")
        samples = self._ensure_uv_samples(g, plane_entry)
        if samples is None or "local_idx" not in samples:
            raise ValueError("u–v samples missing; Fit the plane again first")
        u0, u1, v0, v1 = self._uv_rect
        sel = (
            (samples["u"] >= u0)
            & (samples["u"] <= u1)
            & (samples["v"] >= v0)
            & (samples["v"] <= v1)
        )
        return np.asarray(samples["local_idx"][sel], dtype=np.int64)

    def _sync_uv_action_buttons(self):
        has_rect = self._uv_rect is not None
        p = self._active_plane_entry()
        g = self._active_group()
        has_promoted = bool(
            g is not None
            and g.get("fit")
            and any(
                "selection_refit" in (x.get("reasons") or [])
                for x in (g["fit"].get("planes") or [])
            )
        )
        has_refit = bool(p is not None and p.get("selection_refit") is not None)
        if hasattr(self, "uv_refit_btn"):
            self.uv_refit_btn.setEnabled(has_rect)
        if hasattr(self, "uv_clear_rect_btn"):
            self.uv_clear_rect_btn.setEnabled(has_rect)
        if hasattr(self, "uv_clear_refit_btn"):
            self.uv_clear_refit_btn.setEnabled(has_promoted or has_refit)
        if hasattr(self, "uv_map_base_btn"):
            self.uv_map_base_btn.setEnabled(True)
            self.uv_map_base_btn.setChecked(self._uv_map_mode == "base" or not has_refit)
        if hasattr(self, "uv_map_refit_btn"):
            self.uv_map_refit_btn.setEnabled(has_refit)
            self.uv_map_refit_btn.setChecked(has_refit and self._uv_map_mode == "refit")
        if not has_refit and self._uv_map_mode == "refit":
            self._uv_map_mode = "base"

    @staticmethod
    def _binned_uv_mean(
        u: np.ndarray,
        v: np.ndarray,
        r: np.ndarray,
        u_edges: np.ndarray,
        v_edges: np.ndarray,
        *,
        vlim_mm: float,
    ) -> dict:
        """Mean signed residual on an existing (u, v) grid."""
        counts, _, _ = np.histogram2d(u, v, bins=[u_edges, v_edges])
        sums, _, _ = np.histogram2d(u, v, bins=[u_edges, v_edges], weights=r)
        with np.errstate(invalid="ignore"):
            mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        return {
            "mean": mean,
            "counts": counts,
            "u_edges": np.asarray(u_edges, dtype=np.float64),
            "v_edges": np.asarray(v_edges, dtype=np.float64),
            "vlim_mm": float(vlim_mm),
            "n_used": int(len(r)),
            "bins": int(len(u_edges) - 1),
        }

    def _active_uv_map(self, plane_entry: dict) -> dict | None:
        thr = float(self._uv_vlim_mm())
        rf = plane_entry.get("selection_refit")
        if self._uv_map_mode == "refit" and rf is not None:
            if rf.get("uv") is None:
                self._rebuild_selection_refit_uv(plane_entry)
            uv = rf.get("uv")
            if uv is not None:
                uv["vlim_mm"] = thr
                return uv
        uv = plane_entry.get("uv")
        if uv is not None:
            uv["vlim_mm"] = thr
        return uv

    def _rebuild_selection_refit_uv(self, plane_entry: dict) -> None:
        """Rebuild selection-refit u–v map on the current base grid."""
        rf = plane_entry.get("selection_refit")
        base_uv = plane_entry.get("uv")
        rect = plane_entry.get("uv_rect") or self._uv_rect
        g = self._active_group()
        if rf is None or base_uv is None or rect is None or g is None:
            return
        samples = self._ensure_uv_samples(g, plane_entry)
        if samples is None:
            return
        u0, u1, v0, v1 = rect
        sel = (
            (samples["u"] >= u0)
            & (samples["u"] <= u1)
            & (samples["v"] >= v0)
            & (samples["v"] <= v1)
        )
        if not np.any(sel):
            return
        local = np.asarray(samples["local_idx"][sel], dtype=np.int64)
        if len(g.get("indices", [])) == 0:
            return
        pts = self.full_points[g["indices"]][local]
        plane = Plane.from_array(rf["abcd"])
        r = plane.signed_distances(pts)
        rf["uv"] = self._binned_uv_mean(
            samples["u"][sel],
            samples["v"][sel],
            r,
            base_uv["u_edges"],
            base_uv["v_edges"],
            vlim_mm=float(self._uv_vlim_mm()),
        )

    def _set_uv_map_mode(self, mode: str):
        if mode not in ("base", "refit"):
            return
        p = self._active_plane_entry()
        if mode == "refit" and (p is None or p.get("selection_refit") is None):
            mode = "base"
        self._uv_map_mode = mode
        self._sync_uv_action_buttons()
        if self._uv_view is None or p is None:
            return
        uv = self._active_uv_map(p)
        if uv is None:
            return
        self._uv_view["uv"] = uv
        if mode == "refit":
            rf = p["selection_refit"]
            self._uv_view["mad_um"] = float(rf["mad_sigma_mm"]) * 1e3
            self._uv_view["status"] = str(rf.get("status", ""))
        else:
            self._uv_view["mad_um"] = float(p["mad_sigma_mm"]) * 1e3
            self._uv_view["status"] = str(p.get("status", ""))
        self._uv_view["hist"] = self._hist_for_current_uv_rect(p)
        self._uv_view["plane_entry"] = p
        self._redraw_uv_view()

    def _set_uv_rect_buttons(self, enabled: bool):
        # Backward-compatible helper used by clear-plot paths.
        if not enabled:
            if hasattr(self, "uv_refit_btn"):
                self.uv_refit_btn.setEnabled(False)
            if hasattr(self, "uv_clear_rect_btn"):
                self.uv_clear_rect_btn.setEnabled(False)
            if hasattr(self, "uv_clear_refit_btn"):
                self.uv_clear_refit_btn.setEnabled(False)
            return
        self._sync_uv_action_buttons()

    def _persist_uv_rect_to_plane(self):
        p = self._active_plane_entry()
        if p is None:
            return
        p["uv_rect"] = None if self._uv_rect is None else tuple(self._uv_rect)

    def _restore_uv_rect_from_plane(self, plane_entry: dict):
        stored = plane_entry.get("uv_rect")
        self._uv_rect = None if stored is None else tuple(stored)

    def _uv_meta_text(
        self,
        *,
        title: str,
        uv: dict,
        mad_um: float,
        status: str,
        hist: dict | None,
        plane_entry: dict | None = None,
    ) -> str:
        n_hist = int(hist["n_used"]) if hist else 0
        rect_note = ""
        if self._uv_rect is not None:
            u0, u1, v0, v1 = self._uv_rect
            rect_note = (
                f"  |  rect u[{u0:.1f},{u1:.1f}] v[{v0:.1f},{v1:.1f}] mm"
                f"  |  {n_hist:,} sel pts"
            )
        refit_note = ""
        if plane_entry is not None and plane_entry.get("selection_refit") is not None:
            rf = plane_entry["selection_refit"]
            refit_note = (
                f"  |  refit mad {float(rf['mad_sigma_mm'])*1e3:.0f} µm"
                f"  {rf.get('status', '')}"
            )
        return (
            f"{title}  |  {uv['n_used']:,} pts  |  "
            f"base mad {mad_um:.0f} µm  |  {status}{rect_note}{refit_note}"
        )

    def _remove_uv_roi(self):
        if self._uv_roi is not None and self._uv_plot is not None:
            try:
                self._uv_roi.sigRegionChangeFinished.disconnect(self._on_uv_roi_finished)
            except Exception:
                pass
            try:
                self._uv_plot.removeItem(self._uv_roi)
            except Exception:
                pass
        self._uv_roi = None

    def _sync_uv_roi_item(self):
        """Keep an editable RectROI in sync with ``self._uv_rect``."""
        import pyqtgraph as pg

        if self._uv_plot is None:
            return
        if self._uv_rect is None:
            self._remove_uv_roi()
            return
        u0, u1, v0, v1 = self._uv_rect
        self._uv_roi_block = True
        try:
            if self._uv_roi is None:
                self._uv_roi = pg.RectROI(
                    [u0, v0],
                    [max(u1 - u0, 1e-6), max(v1 - v0, 1e-6)],
                    pen=pg.mkPen("#1a4a9a", width=1.5),
                    hoverPen=pg.mkPen("#2a6fdb", width=2.0),
                    handlePen=pg.mkPen("#1a4a9a"),
                    invertible=True,
                )
                self._uv_roi.addScaleHandle([1, 1], [0, 0])
                self._uv_roi.addScaleHandle([0, 0], [1, 1])
                self._uv_roi.addScaleHandle([1, 0], [0, 1])
                self._uv_roi.addScaleHandle([0, 1], [1, 0])
                self._uv_plot.addItem(self._uv_roi)
                self._uv_roi.sigRegionChangeFinished.connect(self._on_uv_roi_finished)
            else:
                self._uv_roi.setPos([u0, v0], update=False)
                self._uv_roi.setSize([max(u1 - u0, 1e-6), max(v1 - v0, 1e-6)])
        finally:
            self._uv_roi_block = False

    def _on_uv_roi_finished(self):
        if self._uv_roi_block or self._uv_roi is None:
            return
        pos = self._uv_roi.pos()
        size = self._uv_roi.size()
        u0, u1 = sorted((float(pos.x()), float(pos.x() + size.x())))
        v0, v1 = sorted((float(pos.y()), float(pos.y() + size.y())))
        if (u1 - u0) <= 0 or (v1 - v0) <= 0:
            return
        self._apply_uv_rect(u0, u1, v0, v1, from_roi=True)

    def _apply_uv_rect(self, u0, u1, v0, v1, *, from_roi: bool = False):
        self._uv_rect = (float(u0), float(u1), float(v0), float(v1))
        p = self._active_plane_entry()
        if p is not None:
            p["uv_rect"] = self._uv_rect
        if not from_roi:
            self._sync_uv_roi_item()
        self._sync_uv_action_buttons()
        self._update_uv_view_hist()
        self._refresh_uv_hist_panel()

    def _on_uv_rect_selected(self, eclick, erelease):
        # Kept for compatibility; Cmd/Ctrl+drag / ROI handles call _apply_uv_rect.
        if eclick.xdata is None or erelease.xdata is None:
            return
        if eclick.ydata is None or erelease.ydata is None:
            return
        u0, u1 = sorted((float(eclick.xdata), float(erelease.xdata)))
        v0, v1 = sorted((float(eclick.ydata), float(erelease.ydata)))
        if (u1 - u0) <= 0 or (v1 - v0) <= 0:
            return
        self._apply_uv_rect(u0, u1, v0, v1)

    def _clear_uv_rect(self):
        p = self._active_plane_entry()
        if p is not None:
            p["uv_rect"] = None
        self._uv_rect = None
        self._remove_uv_roi()
        self._sync_uv_action_buttons()
        self._update_uv_view_hist()
        self._refresh_uv_hist_panel()
        self._refresh_tree()

    def _clear_uv_refit(self):
        """Remove the last plane added by Refit selection on this group."""
        g = self._active_group()
        if g is None or g.get("fit") is None:
            return
        planes = g["fit"].get("planes") or []
        for i in range(len(planes) - 1, -1, -1):
            reasons = planes[i].get("reasons") or []
            if "selection_refit" in reasons:
                removed = planes.pop(i)
                if planes:
                    stay = planes[min(i, len(planes) - 1)]
                    self._active_plane_index = int(stay.get("plane_index", 0))
                    self._tree_focus = "plane"
                else:
                    g["fit"] = None
                    self._active_plane_index = 0
                    self._tree_focus = "group"
                self._uv_map_mode = "base"
                self._sync_uv_action_buttons()
                self._refresh_tree()
                self._reduction_fill_bind_combo()
                self._show_uv_for_selection()
                self._status(
                    f"removed {g['name']}/{_plane_label(removed)} "
                    "(selection refit); earlier planes kept"
                )
                return
        p = self._active_plane_entry()
        if p is not None and p.get("selection_refit") is not None:
            p.pop("selection_refit", None)
            self._uv_map_mode = "base"
            self._sync_uv_action_buttons()
            self._refresh_tree()
            self._show_uv_for_selection()
            self._status("cleared leftover selection-refit sidecar")
            return
        self._status("no selection-refit plane to remove")

    def _refit_uv_selection(self):
        """Fit the u–v selection and append it as the next plane (p1, p2, …)."""
        if self._uv_rect is None:
            raise ValueError("select a u–v rectangle first")
        g = self._active_group()
        if g is None or g.get("fit") is None:
            raise ValueError("no fitted active group")
        planes = g["fit"].get("planes") or []
        if not planes:
            raise ValueError("no planes to refit")
        p = self._active_plane_entry()
        if p is None:
            raise ValueError("no planes to refit")
        if p.get("uv") is None and self._ensure_plane_uv(g, p) is None:
            raise ValueError("u–v map missing; Fit the plane again first")
        samples = self._ensure_uv_samples(g, p)
        if samples is None:
            raise ValueError("u–v samples missing; Fit the plane again first")
        local = self._uv_rect_local_indices(p)
        n_sel = int(len(local))
        if n_sel < 50:
            raise ValueError(f"too few points in selection ({n_sel}; need ≥ 50)")

        pts = self.full_points[g["indices"]]
        subset = pts[local]
        backend = self.settings.detection.ransac_backend
        min_pts = min(1000, max(50, n_sel // 5))
        compute = resolve_compute_backend(
            self.settings.detection.compute_backend, n_points=n_sel
        )
        self._status(
            f"refitting {g['name']}/p{p['plane_index']} on {n_sel:,} selected pts "
            f"({compute}) ..."
        )
        QApplication.processEvents()

        t0 = time.perf_counter()
        res = extract_main_plane(
            subset,
            MainPlaneParams(
                ransac_backend=backend,
                max_threshold_mm=FIT_MAX_THRESHOLD_MM,
                min_points=min_pts,
                compute_backend=self.settings.detection.compute_backend,
            ),
            clicked=None,
            coarse_plane=np.asarray(p["abcd"], dtype=np.float64),
        )
        t_fit = time.perf_counter()
        if res.n_main < 50:
            raise ValueError("refit produced too few main-component points")

        mad = float(res.fit.stats_inliers["mad_sigma"])
        bimodal = bool(
            _bimodality_flag(
                res.plane.signed_distances(subset[res.main_mask]),
                mad,
            )
        )
        inlier_local = np.asarray(local, dtype=np.int64)[res.main_mask]
        next_index = max(int(x.get("plane_index", 0)) for x in planes) + 1
        entry = {
            "plane_index": next_index,
            "abcd": res.plane.as_array().tolist(),
            "n_points": int(res.n_main),
            "status": res.status,
            "reasons": list(res.reasons) + ["selection_refit"],
            "bimodal": bimodal,
            "mad_sigma_mm": mad,
            "threshold_mm": float(res.fit.threshold),
            "source_plane_index": int(p["plane_index"]),
            "n_selected": n_sel,
            "inlier_local": inlier_local,
        }
        p["uv_rect"] = tuple(self._uv_rect)
        mask = np.zeros(len(pts), dtype=bool)
        mask[inlier_local] = True
        self._cache_uv_for_plane(pts, entry, mask, lock_basis=p.get("uv_basis"))
        planes.append(entry)
        t_end = time.perf_counter()
        timing = {
            "group": f"{g['name']}/p{next_index}",
            "n_pts": n_sel,
            "compute": compute,
            "ransac_backend": backend,
            "multi": False,
            "fit_s": t_fit - t0,
            "uv_s": t_end - t_fit,
            "total_s": t_end - t0,
            "wall_s": t_end - t0,
            "planes": [{
                "plane_index": next_index,
                "status": res.status,
                "n_points": int(res.n_main),
                "mad_sigma_mm": mad,
                "bimodal": bimodal,
            }],
        }
        self._log_fit_timing(timing, kind="selection_refit")
        self._active_plane_index = next_index
        self._tree_focus = "plane"
        self._uv_map_mode = "base"
        self._refresh_tree()
        self._reduction_fill_bind_combo()
        self._sync_uv_action_buttons()
        self._show_uv_for_selection()
        self._status(
            f"{g['name']}/p{next_index} from selection on {g['name']}/p{p['plane_index']} "
            f"({n_sel:,} pts) → mad {mad*1e3:.0f} µm  |  {res.status}"
            + (" BIMODAL" if bimodal else "")
            + f"  |  {self._fit_timing_status(timing)}"
            + f"  |  import as {g['name']}_{_plane_id_token(entry)}  |  fit.log"
        )
        self._sync_action_states()

    def _update_uv_view_hist(self):
        if self._uv_view is None:
            return
        p = self._active_plane_entry()
        if p is None:
            return
        self._uv_view["hist"] = self._hist_for_current_uv_rect(p)
        self._uv_view["plane_entry"] = p

    def _refresh_uv_hist_panel(self):
        """Update only the histogram + meta after a u–v rectangle change."""
        if self._uv_view is None or self._hist_plot is None:
            return
        view = self._uv_view
        uv = view["uv"]
        hist = view["hist"]
        p = view.get("plane_entry") or self._active_plane_entry()
        vlim_um = float(uv["vlim_mm"]) * 1e3
        scope = self._uv_hist_scope(p)
        self._draw_hist(
            hist, vlim_um=vlim_um, mad_um=view["mad_um"], scope=scope
        )
        if hasattr(self, "uv_meta_label"):
            self.uv_meta_label.setText(
                self._uv_meta_text(
                    title=view["title"],
                    uv=uv,
                    mad_um=view["mad_um"],
                    status=view["status"],
                    hist=hist,
                    plane_entry=p,
                )
            )

    def _clear_uv_plot(self, msg: str = "Fit a plane to see residuals."):
        if hasattr(self, "uv_meta_label"):
            self.uv_meta_label.setText(msg)
        self._uv_view = None
        self._uv_rect = None
        self._uv_map_mode = "base"
        self._remove_uv_roi()
        self._set_uv_rect_buttons(False)
        if self._uv_img is not None:
            self._uv_img.clear()
        if self._uv_plot is not None:
            self._uv_plot.setTitle("u–v map")
        self._clear_hist_items()
        if self._hist_plot is not None:
            self._hist_plot.setTitle("no data")
        self._refresh_active_plane_bbox()

    def _clear_hist_items(self):
        import pyqtgraph as pg

        if self._hist_plot is None:
            return
        if self._hist_bar is not None:
            try:
                self._hist_plot.removeItem(self._hist_bar)
            except Exception:
                pass
            self._hist_bar = None
        for line in self._hist_lines:
            try:
                self._hist_plot.removeItem(line)
            except Exception:
                pass
        self._hist_lines = []

    def _draw_hist(self, hist: dict | None, *, vlim_um: float, mad_um: float, scope: str):
        import pyqtgraph as pg

        if self._hist_plot is None:
            return
        self._clear_hist_items()
        if hist is not None and hist.get("counts") is not None and hist["n_used"] > 0:
            edges_um = np.asarray(hist["edges_mm"], dtype=np.float64) * 1e3
            counts = np.asarray(hist["counts"], dtype=np.float64)
            centers = 0.5 * (edges_um[:-1] + edges_um[1:])
            width = float(edges_um[1] - edges_um[0]) if len(edges_um) > 1 else 1.0
            cmap = self._uv_cmap or _rdbu_r_colormap()
            brushes = []
            pens = []
            for c in centers:
                t = 0.5 + 0.5 * float(np.clip(c / max(vlim_um, 1e-9), -1.0, 1.0))
                qcol = cmap.map(t, mode="qcolor")
                brushes.append(pg.mkBrush(qcol))
                pens.append(pg.mkPen("#4a4a4a", width=0.5))
            self._hist_bar = pg.BarGraphItem(
                x=centers,
                height=counts,
                width=width * 0.92,
                brushes=brushes,
                pens=pens,
            )
            self._hist_plot.addItem(self._hist_bar)
            mad = float(hist.get("mad_mm", mad_um / 1e3)) * 1e3
            for x, style in (
                (0.0, Qt.SolidLine),
                (+mad, Qt.DashLine),
                (-mad, Qt.DashLine),
            ):
                line = pg.InfiniteLine(
                    pos=x,
                    angle=90,
                    pen=pg.mkPen("#333333" if x == 0.0 else "#666666", width=1, style=style),
                )
                self._hist_plot.addItem(line)
                self._hist_lines.append(line)
            n_out = int(hist.get("n_outside", 0))
            out_note = f"  ·  {n_out} outside" if n_out else ""
            threshold_um = float(hist.get("threshold_mm", vlim_um / 1e3)) * 1e3
            self._hist_plot.setTitle(
                f"{scope}  ·  ±{threshold_um:.0f} µm"
                f"  ·  mad {mad:.0f} µm{out_note}",
                size="9pt",
            )
            self._hist_plot.setXRange(-vlim_um, vlim_um, padding=0.02)
            ymax = float(counts.max()) if counts.size else 1.0
            self._hist_plot.setYRange(0.0, ymax * 1.05, padding=0.0)
        else:
            self._hist_plot.setTitle("no points in selection", size="9pt")

    def _redraw_uv_view(self, *, keep_selector: bool = False):
        import pyqtgraph as pg
        from PySide6.QtCore import QRectF

        if self._uv_view is None or self._uv_img is None or self._uv_plot is None:
            return
        view = self._uv_view
        uv = view["uv"]
        hist = view["hist"]
        title = view["title"]
        mad_um = view["mad_um"]
        status = view["status"]
        p = view.get("plane_entry") or self._active_plane_entry()
        vlim_um = float(uv["vlim_mm"]) * 1e3

        mean = np.asarray(uv["mean"], dtype=np.float64) * 1e3
        # ImageItem: row = v, col = u (matches former mean.T pcolormesh).
        img = np.array(mean.T, copy=True)
        img[~np.isfinite(img)] = np.nan
        ue = np.asarray(uv["u_edges"], dtype=np.float64)
        ve = np.asarray(uv["v_edges"], dtype=np.float64)
        self._uv_img.setImage(img, autoLevels=False)
        self._uv_img.setLevels([-vlim_um, vlim_um])
        if self._uv_cmap is not None:
            self._uv_img.setLookupTable(self._uv_cmap.getLookupTable(nPts=256))
        self._uv_img.setRect(QRectF(ue[0], ve[0], ue[-1] - ue[0], ve[-1] - ve[0]))
        if self._uv_cbar is not None:
            self._uv_cbar.setLevels((-vlim_um, vlim_um))

        map_note = "base fit"
        if p is not None and p.get("selection_refit") is not None:
            if self._uv_map_mode == "refit":
                map_note = "selection refit"
            else:
                map_note = "base fit (refit available)"
        self._uv_plot.setTitle(f"{title}  ·  u–v map ({map_note})", size="9pt")
        # Keep view range stable when toggling base/refit on the same axes.
        if keep_selector or self._uv_rect is not None:
            pass
        else:
            self._uv_plot.getViewBox().autoRange()

        scope = self._uv_hist_scope(p)
        self._draw_hist(hist, vlim_um=vlim_um, mad_um=mad_um, scope=scope)
        self._sync_uv_roi_item()
        self._sync_uv_action_buttons()

        if hasattr(self, "uv_meta_label"):
            self.uv_meta_label.setText(
                self._uv_meta_text(
                    title=title,
                    uv=uv,
                    mad_um=mad_um,
                    status=status,
                    hist=hist,
                    plane_entry=p,
                )
            )

    def _draw_uv(
        self,
        uv: dict,
        hist: dict | None,
        *,
        title: str,
        mad_um: float,
        status: str,
        plane_entry: dict | None = None,
    ):
        self._uv_view = {
            "uv": uv,
            "hist": hist,
            "title": title,
            "mad_um": mad_um,
            "status": status,
            "plane_entry": plane_entry,
        }
        self._redraw_uv_view()

    def _ensure_plane_uv(self, g: dict, plane_entry: dict) -> dict | None:
        uv = plane_entry.get("uv")
        n_idx = len(g.get("indices", []))
        want_bins = self._uv_bins_value()
        want_vlim = float(self._uv_vlim_mm())
        basis = plane_entry.get("uv_basis") or {}
        if (
            uv is not None
            and plane_entry.get("uv_local_idx") is not None
            and basis.get("basis") == "minrect"
            and plane_entry.get("residual_hist") is not None
            and int(uv.get("bins", 0)) == want_bins
        ):
            uv["vlim_mm"] = want_vlim
            return uv
        if self.full_points.size == 0 or n_idx == 0:
            return None
        pts = self.full_points[g["indices"]]
        plane = Plane.from_array(plane_entry["abcd"])
        mad = float(plane_entry["mad_sigma_mm"])
        thr = max(3.0 * mad, 0.05)
        inlier = plane_entry.get("inlier_local")
        if inlier is not None and len(inlier):
            mask = np.zeros(len(pts), dtype=bool)
            loc = np.asarray(inlier, dtype=np.int64)
            loc = loc[(loc >= 0) & (loc < len(pts))]
            mask[loc] = True
        else:
            mask = np.abs(plane.signed_distances(pts)) <= thr
        if not np.any(mask):
            return None
        # Drop stale selection-refit UV that was binned on the old grid.
        if plane_entry.get("selection_refit") is not None:
            plane_entry["selection_refit"].pop("uv", None)
        self._cache_uv_for_plane(pts, plane_entry, mask)
        return plane_entry.get("uv")

    def _show_uv_for_selection(self):
        g = self._active_group()
        if g is None or g.get("fit") is None:
            self._clear_uv_plot("Fit a plane to see residuals.")
            return
        p = self._active_plane_entry()
        if p is None:
            self._clear_uv_plot("No planes in fit.")
            return
        title = f"{g['name']} / {_plane_label(p)}"
        # Restore this plane's rectangle (and keep any selection refit).
        self._restore_uv_rect_from_plane(p)
        if p.get("selection_refit") is None and self._uv_map_mode == "refit":
            self._uv_map_mode = "base"
        base_uv = self._ensure_plane_uv(g, p)
        if base_uv is None:
            self._clear_uv_plot("Not enough inliers for residual plots.")
            return
        uv = self._active_uv_map(p) or base_uv
        hist = self._hist_for_current_uv_rect(p)
        if self._uv_map_mode == "refit" and p.get("selection_refit") is not None:
            rf = p["selection_refit"]
            mad_um = float(rf["mad_sigma_mm"]) * 1e3
            status = str(rf.get("status", ""))
        else:
            mad_um = float(p["mad_sigma_mm"]) * 1e3
            status = str(p.get("status", ""))
        self._draw_uv(
            uv,
            hist,
            title=title,
            mad_um=mad_um,
            status=status,
            plane_entry=p,
        )
        self._refresh_active_plane_bbox()
