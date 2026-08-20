"""Groups dock, settings, pick/fit, project I/O, and tree UI."""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cloudet.core.array_backend import (
    cupy_unavailable_reason,
    device_name,
    resolve_compute_backend,
    set_default_backend,
)
from cloudet.project.groups import load_groups
from cloudet.fit.mainplane import MainPlaneParams, extract_main_plane
from cloudet.fit.multiplane import MultiPlaneParams, bimodality_flag, extract_planes
from cloudet.core.neighbors import (
    VoxelHashGrid,
    depth_layers_along_ray,
    display_xyz,
    resolve_display_backend,
)
from cloudet.fit.picking import PickParams, pick_plane_region
from cloudet.core.plane import Plane, mad_sigma
from cloudet.core.plyio import read_ply_xyz
from cloudet.project import (
    SourceInfo,
    ViewSettings,
    load_group_doc,
    load_group_indices,
    load_plane_inlier_indices,
    load_settings,
    read_manifest,
    save_group,
    save_settings,
    write_manifest,
)
from cloudet.project.spatial_cache import load_display_xyz, load_voxel_grid, save_display_xyz, save_voxel_grid
from cloudet.project.settings_apply import classify_settings_apply
from cloudet.ui.constants import (
    DEPTH_TIP,
    FIT_MAX_THRESHOLD_MM,
    SETTINGS_HELP_DEFAULT,
)
from cloudet.ui.plane_labels import plane_id_token, plane_label
from cloudet.ui.qt_helpers import route_vtk_messages_to_file
from cloudet.ui.widgets import UI_STYLE, _make_collapsible_card, _reset_tree_widget, group_color


class GroupsMixin:
    """Left dock: project, source, pick, groups tree, settings, fit/save."""

    def _build_dock(self):
        dock = QDockWidget("cloudet", self)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        dock.setStyleSheet(UI_STYLE)
        tabs = QTabWidget()

        # ---- Groups tab ----
        gw = QWidget()
        gl = QVBoxLayout(gw)
        gl.setSpacing(8)

        project_card, pr_lay = _make_collapsible_card("PROJECT")
        self.project_label = QLabel(self.project_dir.name)
        self.project_label.setWordWrap(True)
        self.project_label.setToolTip(str(self.project_dir.resolve()))
        pr_lay.addWidget(self.project_label)
        self.project_meta_label = QLabel(str(self.project_dir.resolve()))
        self.project_meta_label.setObjectName("muted")
        self.project_meta_label.setWordWrap(True)
        pr_lay.addWidget(self.project_meta_label)
        project_row = QHBoxLayout()
        self.browse_project_btn = QPushButton("Browse...")
        self.browse_project_btn.setObjectName("secondaryBtn")
        self.browse_project_btn.setToolTip(
            "Choose the output folder for groups/, settings.json, and manifest.json."
        )
        self.browse_project_btn.clicked.connect(
            lambda: self._guard(self._browse_project, busy=False)
        )
        project_row.addWidget(self.browse_project_btn)
        pr_lay.addLayout(project_row)
        gl.addWidget(project_card)

        source_card, s_lay = _make_collapsible_card("SOURCE")
        self.cloud_label = QLabel("(no cloud loaded)")
        self.cloud_label.setWordWrap(True)
        s_lay.addWidget(self.cloud_label)
        self.source_meta_label = QLabel("0 pts displayed")
        self.source_meta_label.setObjectName("muted")
        s_lay.addWidget(self.source_meta_label)
        source_row = QHBoxLayout()
        self.open_cloud_btn = QPushButton("Open...")
        self.open_cloud_btn.setObjectName("secondaryBtn")
        self.open_cloud_btn.clicked.connect(self._browse_cloud)
        source_row.addWidget(self.open_cloud_btn)
        self.load_cloud_btn = QPushButton("Load")
        self.load_cloud_btn.setObjectName("primaryBtn")
        self.load_cloud_btn.clicked.connect(lambda: self._guard(self._load_cloud))
        source_row.addWidget(self.load_cloud_btn)
        s_lay.addLayout(source_row)
        gl.addWidget(source_card)

        pick_card, p_lay = _make_collapsible_card("PICK")
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.mode_new_rb = QCheckBox("New")
        self.mode_append_rb = QCheckBox("Append")
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.mode_new_rb)
        self.mode_group.addButton(self.mode_append_rb)
        self.mode_new_rb.setChecked(True)
        self.mode_new_rb.toggled.connect(
            lambda checked: self.append_cb.setChecked(not checked)
        )
        self.mode_append_rb.toggled.connect(
            lambda checked: self.append_cb.setChecked(checked)
        )
        mode_row.addWidget(self.mode_new_rb)
        mode_row.addWidget(self.mode_append_rb)
        mode_row.addStretch()
        p_lay.addLayout(mode_row)

        self.append_cb = QCheckBox("Append picks to active group")
        self.append_cb.hide()
        self.append_cb.toggled.connect(
            lambda checked: (
                self.mode_append_rb.setChecked(checked),
                self.mode_new_rb.setChecked(not checked),
            )
        )
        p_lay.addWidget(self.append_cb)
        self.solo_cb = QCheckBox("Show only active group")
        self.solo_cb.toggled.connect(lambda _: self._refresh_group_actors())
        p_lay.addWidget(self.solo_cb)
        self.autofit_cb = QCheckBox("Auto-fit after pick")
        self.autofit_cb.setChecked(True)
        self.autofit_cb.setToolTip(
            "Run fit + QC right after each pick. Large faces can take a while; "
            "turn off for faster picking, then press Fit when ready."
        )
        p_lay.addWidget(self.autofit_cb)
        self.multiplane_cb = QCheckBox("Extract multiple planes (p0, p1, …)")
        self.multiplane_cb.setChecked(False)
        self.multiplane_cb.setToolTip(
            "Split one picked group into several planes (near-parallel faces). "
            "Each plane appears as p0, p1, … under the group and can be imported "
            "into Reduction as G6_p1, etc. Keep off for one-click one-face."
        )
        p_lay.addWidget(self.multiplane_cb)

        self.snap_front_cb = QCheckBox("Pick nearer surface")
        self.snap_front_cb.setChecked(True)
        self.snap_front_cb.setToolTip(
            "Point clouds have visual gaps, so a pick can land on a farther "
            "surface. When on, re-snap to the nearer surface on the view ray. "
            "Use > / < after P to step farther / nearer."
        )
        p_lay.addWidget(self.snap_front_cb)
        gl.addWidget(pick_card)

        pick_badge = QLabel(" P  PICK ")
        pick_badge.setStyleSheet(
            "border: 1px solid palette(mid); border-radius: 4px; "
            "font-size: 10px; font-weight: 600;"
        )
        pick_badge.setToolTip("Hover the 3D view and press P")
        depth_card, depth_box = _make_collapsible_card(
            "DEPTH", header_extra=pick_badge
        )
        depth_card.setToolTip(DEPTH_TIP)
        depth_box.setSpacing(2)

        depth_row = QHBoxLayout()
        self.depth_prev_btn = QPushButton("<")
        self.depth_prev_btn.setObjectName("depthArrow")
        self.depth_prev_btn.setToolTip("Nearer surface on the view ray (key <)")
        self.depth_prev_btn.clicked.connect(
            lambda: self._guard(lambda: self._cycle_pick_depth(-1))
        )
        depth_row.addWidget(self.depth_prev_btn)
        self.depth_label = QLabel("— / —")
        self.depth_label.setObjectName("depthCount")
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setToolTip(DEPTH_TIP)
        depth_row.addWidget(self.depth_label, stretch=1)
        self.depth_next_btn = QPushButton(">")
        self.depth_next_btn.setObjectName("depthArrow")
        self.depth_next_btn.setToolTip("Farther surface on the view ray (key >)")
        self.depth_next_btn.clicked.connect(
            lambda: self._guard(lambda: self._cycle_pick_depth(+1))
        )
        depth_row.addWidget(self.depth_next_btn)
        depth_box.addLayout(depth_row)
        self.depth_dots = QLabel("○")
        self.depth_dots.setObjectName("depthDots")
        self.depth_dots.setAlignment(Qt.AlignCenter)
        depth_box.addWidget(self.depth_dots)
        self.depth_meta = QLabel("P")
        self.depth_meta.setObjectName("depthMeta")
        self.depth_meta.setAlignment(Qt.AlignCenter)
        depth_box.addWidget(self.depth_meta)
        gl.addWidget(depth_card)
        self._update_depth_controls()

        groups_hdr = QHBoxLayout()
        groups_title = QLabel("GROUPS")
        groups_title.setObjectName("sectionTitle")
        groups_hdr.addWidget(groups_title)
        groups_hdr.addStretch()
        self.group_count_label = QLabel("0")
        self.group_count_label.setObjectName("muted")
        groups_hdr.addWidget(self.group_count_label)
        gl.addLayout(groups_hdr)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["group / plane", "n, d", "points", "quality"])
        self.tree.setColumnWidth(0, 120)
        self.tree.setColumnWidth(1, 280)
        self.tree.setAlternatingRowColors(True)
        self.tree.setMouseTracking(True)
        self.tree.viewport().setMouseTracking(True)
        self.tree.viewport().installEventFilter(self)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(self._on_item_selected)
        gl.addWidget(self.tree, stretch=1)
        self.tree.itemSelectionChanged.connect(self._sync_action_states)

        toolbar = QGridLayout()
        toolbar.setHorizontalSpacing(6)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setObjectName("primaryBtn")
        self.fit_btn.clicked.connect(lambda: self._guard(self._fit_active))
        toolbar.addWidget(self.fit_btn, 0, 0)
        self.fit_all_btn = QPushButton("Fit All")
        self.fit_all_btn.setObjectName("secondaryBtn")
        self.fit_all_btn.clicked.connect(lambda: self._guard(self._fit_all))
        toolbar.addWidget(self.fit_all_btn, 0, 1)
        self.merge_btn = QPushButton("Merge")
        self.merge_btn.setObjectName("secondaryBtn")
        self.merge_btn.clicked.connect(lambda: self._guard(self._merge_selected))
        toolbar.addWidget(self.merge_btn, 0, 2)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setObjectName("dangerBtn")
        self.delete_btn.setToolTip(
            "Delete the selected plane, or the whole group if a group row "
            "is selected. Backspace does the same. Double-click a name to rename."
        )
        self.delete_btn.clicked.connect(lambda: self._guard(self._delete_active))
        toolbar.addWidget(self.delete_btn, 0, 3)
        self.load_all_btn = QPushButton("Load All")
        self.load_all_btn.setObjectName("secondaryBtn")
        self.load_all_btn.clicked.connect(lambda: self._guard(self._load_all))
        toolbar.addWidget(self.load_all_btn, 1, 0)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("dangerBtn")
        self.clear_btn.clicked.connect(lambda: self._guard(self._clear_all))
        toolbar.addWidget(self.clear_btn, 1, 1)
        gl.addLayout(toolbar)
        self.save_all_btn = QPushButton("Save All")
        self.save_all_btn.setObjectName("primaryBtn")
        self.save_all_btn.clicked.connect(lambda: self._guard(self._save_all))
        gl.addWidget(self.save_all_btn)

        tabs.addTab(gw, "Groups")

        # ---- Settings tab ----
        sw = QWidget()
        swl = QVBoxLayout(sw)
        swl.setSpacing(8)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_body = QWidget()
        settings_body_layout = QVBoxLayout(settings_body)
        settings_body_layout.setContentsMargins(0, 0, 0, 0)
        settings_body_layout.setSpacing(8)
        settings_scroll.setWidget(settings_body)
        swl.addWidget(settings_scroll, stretch=1)
        d, v = self.settings.detection, self.settings.view

        def setting_tip(
            title: str,
            purpose: str,
            applies_to: str,
            key: str,
            *,
            larger: str = "",
            smaller: str = "",
        ) -> str:
            effect = ""
            if larger or smaller:
                effect = (
                    "<p><b>Effect</b><br>"
                    + (f"Larger: {larger}<br>" if larger else "")
                    + (f"Smaller: {smaller}" if smaller else "")
                    + "</p>"
                )
            return (
                "<div style='white-space: normal;'>"
                f"<b style='font-size: 13px;'>{title}</b>"
                f"<p><b>Purpose</b><br>{purpose}</p>"
                f"{effect}"
                f"<p><b>Applies to</b><br>{applies_to}</p>"
                f"<p style='color: #777;'>Internal name: <code>{key}</code></p>"
                "</div>"
            )

        def dspin(value, lo=0.0, hi=1e6, step=0.1, dec=3, tip_text=""):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setSingleStep(step)
            w.setValue(value)
            if tip_text:
                w.setToolTip(tip_text)
            return w

        def ispin(value, lo=0, hi=100_000_000, tip_text=""):
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(int(value))
            if tip_text:
                w.setToolTip(tip_text)
            return w

        def card(title: str, blurb: str = "") -> tuple[QFrame, QVBoxLayout, QFormLayout]:
            frame, vbox = _make_collapsible_card(title)
            if blurb:
                note = QLabel(blurb)
                note.setObjectName("muted")
                note.setWordWrap(True)
                vbox.addWidget(note)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignLeft)
            form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.DontWrapRows)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(4)
            vbox.addLayout(form)
            return frame, vbox, form

        def labeled(text: str, tip_text: str = "") -> QLabel:
            lbl = QLabel(text)
            # Every settings card uses one shared visual label column.
            lbl.setFixedWidth(205)
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            if tip_text:
                self._register_settings_help(lbl, tip_text)
            return lbl

        # --- Detection controls ---
        self.s_radius = dspin(
            d.local_radius_mm,
            tip_text=setting_tip(
                "Initial plane search radius",
                "Sets the radius around the clicked point used to collect points "
                "for the initial plane estimate.",
                "Initial plane near the click only",
                "local_radius_mm",
                larger="uses a wider neighborhood and may include nearby geometry",
                smaller="uses a more local neighborhood but may leave too few points",
            ),
        )
        self.s_radius.setSuffix(" mm")
        self.s_locthr = dspin(
            d.local_distance_threshold_mm,
            tip_text=setting_tip(
                "Local RANSAC inlier distance",
                "During the initial RANSAC fit, a point is an inlier only when "
                "its distance from a trial plane is within this value.",
                "Initial RANSAC near the click; not full-face inclusion",
                "local_distance_threshold_mm",
                larger="accepts more local noise into the initial plane",
                smaller="requires a cleaner initial plane",
            ),
        )
        self.s_locthr.setSuffix(" mm")
        self.s_lociter = ispin(
            d.local_ransac_iterations,
            tip_text=setting_tip(
                "RANSAC trials",
                "Number of random plane hypotheses tested during the initial fit.",
                "Initial RANSAC near the click",
                "local_ransac_iterations",
                larger="is slower but can be more robust",
                smaller="is faster but may miss the correct plane",
            ),
        )
        self.s_minnb = ispin(
            d.min_neighbor_points,
            tip_text=setting_tip(
                "Minimum nearby point count",
                "Rejects the pick when the initial search radius contains fewer "
                "points than this count.",
                "Initial neighborhood validation",
                "min_neighbor_points",
                larger="requires denser local data",
                smaller="allows picks in sparse areas",
            ),
        )
        self.s_minin = ispin(
            d.min_local_inliers,
            tip_text=setting_tip(
                "Minimum initial-plane inlier count",
                "Rejects the pick when initial RANSAC finds fewer plane inlier "
                "points than this count.",
                "Initial plane validation",
                "min_local_inliers",
                larger="requires stronger evidence for the initial plane",
                smaller="accepts initial planes supported by fewer points",
            ),
        )
        self.s_accthr = dspin(
            d.accumulate_threshold_mm,
            tip_text=setting_tip(
                "Face inclusion distance",
                "A point may join the selected face when its distance from the "
                "current fitted plane is within this value.",
                "Full-face expansion and repeated refitting; not initial RANSAC",
                "accumulate_threshold_mm",
                larger="includes more face points but may include nearby noise",
                smaller="produces a cleaner but possibly incomplete face",
            ),
        )
        self.s_accthr.setSuffix(" mm")
        self.s_connect = QCheckBox("On")
        self.s_connect.setChecked(d.connect)
        self.s_connect.setToolTip(
            setting_tip(
                "Connected surface only",
                "Keeps only the connected component containing the clicked face, "
                "preventing distant coplanar surfaces from joining the group.",
                "Full-face extraction",
                "connect",
            )
        )
        self.s_cell = dspin(
            d.cell_size_mm,
            tip_text=setting_tip(
                "Connectivity cell size",
                "Cell size of the in-plane grid used to determine whether face "
                "points are spatially connected.",
                "Connected-component filtering during full-face extraction",
                "cell_size_mm",
                larger="bridges wider gaps and uses coarser connectivity",
                smaller="separates smaller gaps but is more sensitive to sparse data",
            ),
        )
        self.s_cell.setSuffix(" mm")
        self.s_expand = dspin(
            getattr(d, "expand_step_mm", 25.0),
            tip_text=setting_tip(
                "Refinement radius step",
                "Amount added to the in-plane search radius before each plane "
                "refit. This lets a tilted initial plane correct gradually.",
                "Progressive full-face refinement",
                "expand_step_mm",
                larger="covers the face in fewer, coarser refinement steps",
                smaller="uses more gradual refinement steps",
            ),
        )
        self.s_expand.setSuffix(" mm")
        self.s_maxexp = ispin(
            getattr(d, "max_expand_rounds", 40),
            tip_text=setting_tip(
                "Maximum refinement rounds",
                "Maximum number of expand-and-refit cycles allowed for one pick.",
                "Progressive full-face refinement",
                "max_expand_rounds",
                larger="allows refinement to continue farther",
                smaller="stops refinement sooner",
            ),
        )
        max_in = getattr(d, "max_inplane_radius_mm", None)
        self.s_maxinplane = dspin(
            float(max_in) if max_in is not None else 0.0,
            tip_text=setting_tip(
                "Maximum face radius from click",
                "Optional hard limit on the in-plane distance from the clicked "
                "point to included face points. 0 means no radius limit.",
                "Full-face extraction",
                "max_inplane_radius_mm",
                larger="permits a larger face region",
                smaller="cuts the face off closer to the click",
            ),
        )
        self.s_maxinplane.setSuffix(" mm")
        self.s_backend = QComboBox()
        self.s_backend.addItem("built-in (GPU)", "seeded")
        self.s_backend.addItem("built-in (CPU)", "seeded_cpu")
        self.s_backend.addItem("open3d (CPU)", "open3d")
        ransac_raw = getattr(d, "ransac_backend", "seeded")
        if ransac_raw == "numpy":
            ransac_raw = "seeded"
        idx = self.s_backend.findData(ransac_raw)
        self.s_backend.setCurrentIndex(idx if idx >= 0 else 0)
        self.s_backend.setToolTip(
            setting_tip(
                "RANSAC backend",
                "Built-in (GPU) uses cloudet's reproducible RANSAC and scores on "
                "the GPU when CuPy is available (falls back to CPU otherwise). "
                "Built-in (CPU) forces NumPy. Open3D uses segment_plane on CPU only. "
                "Independent of Compute backend (robust Fit / Pick distances / UV).",
                "Initial RANSAC near the click (and Fit seed when RANSAC runs)",
                "ransac_backend",
            )
        )
        self.s_compute_backend = QComboBox()
        self.s_compute_backend.addItems(["auto", "cupy", "numpy"])
        self.s_compute_backend.setCurrentText(
            getattr(d, "compute_backend", "auto")
        )
        self.s_compute_backend.setToolTip(
            setting_tip(
                "Compute backend",
                "Chooses CPU (NumPy) or GPU (CuPy) for Fit, Pick distances, and "
                "residual u–v maps. auto uses CuPy when CUDA is available; cupy "
                "forces GPU even on small groups. RANSAC device is chosen "
                "separately under RANSAC backend.",
                "Fit, Pick, and residual QC (not RANSAC)",
                "compute_backend",
            )
        )

        det_card, det_box, det_form = card(
            "DETECTION",
            "P first estimates a plane near the click, then extracts the full "
            "connected face.",
        )
        initial_stage = QLabel("<b>1  INITIAL PLANE</b>  <small>NEAR CLICK</small>")
        initial_stage.setObjectName("sectionTitle")
        det_form.addRow(initial_stage)
        det_form.addRow(
            labeled(
                "Initial plane search radius",
                self.s_radius.toolTip(),
            ),
            self.s_radius,
        )
        det_form.addRow(
            labeled("Local RANSAC inlier distance", self.s_locthr.toolTip()),
            self.s_locthr,
        )
        det_form.addRow(
            labeled("Minimum nearby point count", self.s_minnb.toolTip()),
            self.s_minnb,
        )
        det_form.addRow(
            labeled(
                "Minimum initial-plane inlier count",
                self.s_minin.toolTip(),
            ),
            self.s_minin,
        )
        det_form.addRow(
            labeled("RANSAC trials", self.s_lociter.toolTip()),
            self.s_lociter,
        )
        det_form.addRow(
            labeled("RANSAC backend", self.s_backend.toolTip()),
            self.s_backend,
        )
        det_form.addRow(
            labeled("Compute backend", self.s_compute_backend.toolTip()),
            self.s_compute_backend,
        )

        face_stage = QLabel(
            "<b>2  FACE EXTRACTION</b>  <small>FULL CONNECTED FACE</small>"
        )
        face_stage.setObjectName("sectionTitle")
        det_form.addRow(face_stage)
        det_form.addRow(
            labeled("Face inclusion distance", self.s_accthr.toolTip()),
            self.s_accthr,
        )
        det_form.addRow(
            labeled("Connected surface only", self.s_connect.toolTip()),
            self.s_connect,
        )
        det_form.addRow(
            labeled("Connectivity cell size", self.s_cell.toolTip()),
            self.s_cell,
        )
        det_form.addRow(
            labeled("Refinement radius step", self.s_expand.toolTip()),
            self.s_expand,
        )
        det_form.addRow(
            labeled(
                "Maximum face radius from click",
                self.s_maxinplane.toolTip(),
            ),
            self.s_maxinplane,
        )
        det_form.addRow(
            labeled("Maximum refinement rounds", self.s_maxexp.toolTip()),
            self.s_maxexp,
        )
        settings_body_layout.addWidget(det_card)

        # --- Display controls ---
        self.s_voxel = dspin(
            v.display_voxel_size_mm,
            tip_text=setting_tip(
                "Display voxel size",
                "Sets display-only voxel spacing. Picking, fitting, and saving "
                "continue to use the full-resolution cloud. 0 disables the filter.",
                "3D display only",
                "display_voxel_size_mm",
                larger="draws fewer points and refreshes faster",
                smaller="draws a denser cloud and refreshes slower",
            ),
        )
        self.s_voxel.setSuffix(" mm")
        self.s_maxdisp = ispin(
            v.display_max_points,
            lo=100_000,
            tip_text=setting_tip(
                "Maximum displayed points",
                "Hard upper limit on the number of points uploaded for display.",
                "3D display only",
                "display_max_points",
                larger="shows a denser cloud but can slow interaction",
                smaller="improves interaction speed",
            ),
        )
        self.s_ds_backend = QComboBox()
        self.s_ds_backend.addItems(["auto", "cupy", "open3d", "numpy"])
        self.s_ds_backend.setCurrentText(
            getattr(v, "display_downsample_backend", "auto")
        )
        self.s_ds_backend.setToolTip(
            setting_tip(
                "Display downsampling method",
                "Chooses how display points are thinned. auto prefers CuPy, "
                "then Open3D when available; rendering remains in Qt / PyVista.",
                "3D display preparation only",
                "display_downsample_backend",
            )
        )
        self.s_ptsize = dspin(
            v.base_point_size,
            lo=0.5,
            hi=20,
            step=0.5,
            dec=1,
            tip_text=setting_tip(
                "Source cloud point size",
                "Controls the rendered point size of the original source cloud.",
                "3D display only",
                "base_point_size",
            ),
        )
        self.s_ptsize.setSuffix(" px")
        self.s_active_pt = dspin(
            v.active_point_size,
            lo=0.5,
            hi=20,
            step=0.5,
            dec=1,
            tip_text=setting_tip(
                "Active group point size",
                "Controls the rendered point size of the currently selected group.",
                "3D display only",
                "active_point_size",
            ),
        )
        self.s_active_pt.setSuffix(" px")
        self.s_inactive_pt = dspin(
            v.inactive_point_size,
            lo=0.5,
            hi=20,
            step=0.5,
            dec=1,
            tip_text=setting_tip(
                "Other groups point size",
                "Controls the rendered point size of groups that are not active.",
                "3D display only",
                "inactive_point_size",
            ),
        )
        self.s_inactive_pt.setSuffix(" px")

        view_card, _view_box, view_form = card(
            "DISPLAY",
            "View only. Does not change pick geometry or fit results.",
        )
        view_form.addRow(
            labeled("Display voxel size", self.s_voxel.toolTip()),
            self.s_voxel,
        )
        view_form.addRow(
            labeled("Maximum displayed points", self.s_maxdisp.toolTip()),
            self.s_maxdisp,
        )
        view_form.addRow(
            labeled("Display downsampling method", self.s_ds_backend.toolTip()),
            self.s_ds_backend,
        )
        view_form.addRow(
            labeled("Source cloud point size", self.s_ptsize.toolTip()),
            self.s_ptsize,
        )
        view_form.addRow(
            labeled("Active group point size", self.s_active_pt.toolTip()),
            self.s_active_pt,
        )
        view_form.addRow(
            labeled("Other groups point size", self.s_inactive_pt.toolTip()),
            self.s_inactive_pt,
        )
        settings_body_layout.addWidget(view_card)
        settings_body_layout.addStretch()

        help_card = QFrame()
        help_card.setObjectName("card")
        help_layout = QVBoxLayout(help_card)
        help_layout.setContentsMargins(10, 8, 10, 8)
        help_title = QLabel("HELP")
        help_title.setObjectName("sectionTitle")
        help_layout.addWidget(help_title)
        self.settings_help_label = QLabel(SETTINGS_HELP_DEFAULT)
        self.settings_help_label.setWordWrap(True)
        # Rich-text content has a different size hint for every setting.
        # Ignore that horizontal hint so hovering never resizes the dock.
        self.settings_help_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self.settings_help_label.setMinimumWidth(0)
        self.settings_help_label.setMinimumHeight(125)
        self.settings_help_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        help_layout.addWidget(self.settings_help_label)
        swl.addWidget(help_card)

        row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("primaryBtn")
        # Detection-only Apply must stay lightweight; WaitCursor is set
        # inside _apply_settings only when display rebuild is needed.
        self.apply_btn.clicked.connect(lambda: self._guard(self._apply_settings, busy=False))
        row.addWidget(self.apply_btn)
        self.save_settings_btn = QPushButton("Save as Default")
        self.save_settings_btn.setObjectName("secondaryBtn")
        self.save_settings_btn.clicked.connect(lambda: self._guard(self._save_settings, busy=False))
        row.addWidget(self.save_settings_btn)
        self.settings_dirty_label = QLabel("Saved")
        self.settings_dirty_label.setObjectName("muted")
        row.addWidget(self.settings_dirty_label)
        row.addStretch()
        swl.addLayout(row)

        self._settings_controls = [
            self.s_radius, self.s_locthr, self.s_lociter, self.s_minnb, self.s_minin,
            self.s_accthr, self.s_connect, self.s_cell, self.s_expand, self.s_maxexp,
            self.s_maxinplane, self.s_backend, self.s_compute_backend,
            self.s_voxel, self.s_maxdisp,
            self.s_ds_backend, self.s_ptsize, self.s_active_pt, self.s_inactive_pt,
        ]
        for w in self._settings_controls:
            help_text = w.toolTip()
            if help_text:
                self._register_settings_help(w, help_text)
                # Native Qt tooltips can appear on the wrong macOS display.
                w.setToolTip("")
            if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                w.valueChanged.connect(lambda *_: self._set_settings_dirty(True))
            elif isinstance(w, QCheckBox):
                w.toggled.connect(lambda *_: self._set_settings_dirty(True))
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(lambda *_: self._set_settings_dirty(True))
        self._set_settings_dirty(False)

        tabs.addTab(sw, "Settings")
        dock.setWidget(tabs)
        dock.setMinimumWidth(380)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._sync_action_states()

    def _read_form_detection(self) -> PickParams:
        max_in = self.s_maxinplane.value()
        return PickParams(
            local_radius_mm=self.s_radius.value(),
            local_distance_threshold_mm=self.s_locthr.value(),
            local_ransac_iterations=self.s_lociter.value(),
            min_neighbor_points=self.s_minnb.value(),
            min_local_inliers=self.s_minin.value(),
            accumulate_threshold_mm=self.s_accthr.value(),
            connect=self.s_connect.isChecked(),
            cell_size_mm=self.s_cell.value(),
            ransac_backend=self.s_backend.currentData() or "seeded",
            compute_backend=self.s_compute_backend.currentText(),
            seed=self.settings.detection.seed,
            min_points_per_cell=self.settings.detection.min_points_per_cell,
            expand_step_mm=self.s_expand.value(),
            max_expand_rounds=self.s_maxexp.value(),
            max_inplane_radius_mm=None if max_in <= 0 else max_in,
            refine_max_points=self.settings.detection.refine_max_points,
        )

    def _read_form_view(self) -> ViewSettings:
        # Preserve fields not exposed in the form (axis_*).
        old = self.settings.view
        return ViewSettings(
            base_point_size=self.s_ptsize.value(),
            active_point_size=self.s_active_pt.value(),
            inactive_point_size=self.s_inactive_pt.value(),
            display_voxel_size_mm=self.s_voxel.value(),
            display_max_points=self.s_maxdisp.value(),
            display_downsample_backend=self.s_ds_backend.currentText(),
            axis_size_mm=old.axis_size_mm,
            axis_margin_mm=old.axis_margin_mm,
        )

    # ------------------------------------------------------------------
    # settings
    # ------------------------------------------------------------------

    def _apply_settings(self) -> str:
        """Apply form values. Returns a short summary of what ran (for status)."""
        old_det = self.settings.detection
        old_view = self.settings.view
        new_det = self._read_form_detection()
        new_view = self._read_form_view()
        effects = classify_settings_apply(old_det, new_det, old_view, new_view)

        self.settings.detection = new_det
        self.settings.view = new_view
        set_default_backend(new_det.compute_backend)

        if effects.invalidate_grid:
            self.grid = None

        # Fast path: Detection (and/or axis-only view) — never touch the plotter.
        if not effects.refresh_display and not effects.update_point_sizes:
            if effects.detection_changed:
                parts = ["detection"]
                if effects.invalidate_grid:
                    parts.append("grid rebuild on next pick")
                summary = ", ".join(parts)
            elif effects.view_changed:
                summary = "view (no redraw)"
            else:
                summary = "unchanged"
            self._status(f"settings applied ({summary})")
            self._set_settings_dirty(False)
            return summary

        if effects.refresh_display:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._base_display_xyz = None
                backend = resolve_display_backend(
                    self.settings.view.display_downsample_backend
                )
                QApplication.processEvents()
                self._refresh_base_actor()
                self._refresh_group_actors()
                self._update_source_meta()
                summary = (
                    f"display refreshed via {backend}; "
                    f"showing {self._n_displayed:,} pts"
                )
                self._status(f"settings applied ({summary})")
            finally:
                QApplication.restoreOverrideCursor()
            self._set_settings_dirty(False)
            return summary

        # Point sizes only: update VTK properties; do not re-upload points.
        self._set_actor_point_sizes_only()
        summary = "point sizes updated"
        self._status(f"settings applied ({summary})")
        self._set_settings_dirty(False)
        return summary

    def _save_settings(self):
        summary = self._apply_settings()
        path = save_settings(self.project_dir, self.settings)
        self._status(f"saved settings to {path}  [{summary}]")
        self._set_settings_dirty(False)

    # ------------------------------------------------------------------
    # cloud
    # ------------------------------------------------------------------

    def _update_project_labels(self):
        if not hasattr(self, "project_label"):
            return
        resolved = str(self.project_dir.resolve())
        self.project_label.setText(self.project_dir.name)
        self.project_label.setToolTip(resolved)
        self.project_meta_label.setText(resolved)
        self.setWindowTitle(f"cloudet - {self.project_dir.name}")

    def _write_form_from_settings(self):
        """Push loaded settings into the Settings form without marking dirty."""
        d, v = self.settings.detection, self.settings.view
        controls = getattr(self, "_settings_controls", [])
        for w in controls:
            w.blockSignals(True)
        try:
            self.s_radius.setValue(d.local_radius_mm)
            self.s_locthr.setValue(d.local_distance_threshold_mm)
            self.s_lociter.setValue(d.local_ransac_iterations)
            self.s_minnb.setValue(d.min_neighbor_points)
            self.s_minin.setValue(d.min_local_inliers)
            self.s_accthr.setValue(d.accumulate_threshold_mm)
            self.s_connect.setChecked(d.connect)
            self.s_cell.setValue(d.cell_size_mm)
            self.s_expand.setValue(getattr(d, "expand_step_mm", 25.0))
            self.s_maxexp.setValue(getattr(d, "max_expand_rounds", 40))
            max_in = getattr(d, "max_inplane_radius_mm", None)
            self.s_maxinplane.setValue(float(max_in) if max_in is not None else 0.0)
            ransac_raw = getattr(d, "ransac_backend", "seeded")
            if ransac_raw == "numpy":
                ransac_raw = "seeded"
            idx = self.s_backend.findData(ransac_raw)
            self.s_backend.setCurrentIndex(idx if idx >= 0 else 0)
            self.s_voxel.setValue(v.display_voxel_size_mm)
            self.s_maxdisp.setValue(v.display_max_points)
            self.s_ds_backend.setCurrentText(
                getattr(v, "display_downsample_backend", "auto")
            )
            self.s_compute_backend.setCurrentText(
                getattr(d, "compute_backend", "auto")
            )
            self.s_ptsize.setValue(v.base_point_size)
            self.s_active_pt.setValue(v.active_point_size)
            self.s_inactive_pt.setValue(v.inactive_point_size)
        finally:
            for w in controls:
                w.blockSignals(False)
        self._set_settings_dirty(False)

    def _browse_project(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select project folder",
            str(self.project_dir),
        )
        if not path:
            return
        self._set_project_dir(Path(path))

    def _set_project_dir(self, path: Path):
        path = Path(path).expanduser()
        if path.resolve() == self.project_dir.resolve():
            self._status(f"project already set to {path}")
            return
        if self.groups:
            answer = QMessageBox.question(
                self,
                "Switch project?",
                "Unsaved groups in the current session will be cleared.\n"
                "Save All first if you need them.\n\n"
                f"Switch to:\n{path}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            self._clear_all()

        path.mkdir(parents=True, exist_ok=True)
        self.project_dir = path
        self._vtk_log_path = route_vtk_messages_to_file(self.project_dir / "vtk.log")
        self._fit_log_path = self.project_dir / "fit.log"
        self.settings = load_settings(self.project_dir, warn=self._status)
        set_default_backend(self.settings.detection.compute_backend)
        self._write_form_from_settings()
        self.grid = None

        # Prefer the cloud recorded in the new project's manifest when present.
        manifest = read_manifest(self.project_dir)
        src = (manifest or {}).get("source", {}).get("path", "")
        if src:
            self.pcd_edit_path = src
            self.cloud_label.setText(Path(src).name)
            self.cloud_label.setToolTip(src)

        self._update_project_labels()
        self._rebuild_status_default()
        self._status(f"project set to {self.project_dir.resolve()}")

    def _browse_cloud(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select point cloud", "",
            "Point clouds (*.ply);;All files (*)",
        )
        if path:
            self.pcd_edit_path = path
            self.cloud_label.setText(Path(path).name)
            self.cloud_label.setToolTip(path)

    def _load_cloud(self):
        path = getattr(self, "pcd_edit_path", "").strip()
        if not path:
            raise ValueError("no cloud file selected (Browse...)")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._status(f"loading {path} ...")
        QApplication.processEvents()
        self.full_points = read_ply_xyz(path)
        self.pcd_path = path
        self.cloud_label.setText(Path(path).name)
        self.cloud_label.setToolTip(path)
        self.grid = None
        self._base_display_xyz = None
        self._clear_all()
        backend = resolve_display_backend(
            self.settings.view.display_downsample_backend
        )
        QApplication.processEvents()
        self._refresh_base_actor()
        self.plotter.reset_camera()
        self._status(
            f"loaded {len(self.full_points):,} points "
            f"(displaying {self._n_displayed:,} via {backend})"
        )
        self._update_source_meta()
        self._sync_action_states()
        # Build (or mmap-cache) the pick index here with the chunked path.
        # Deferring to first pick only postponed the same RAM peak.
        self._ensure_grid()

    def _source_cloud_path(self) -> str:
        return str(
            getattr(self, "pcd_path", "") or getattr(self, "pcd_edit_path", "") or ""
        ).strip()

    def _ensure_grid(self) -> VoxelHashGrid:
        if len(self.full_points) == 0:
            raise ValueError("no cloud loaded")
        cell = VoxelHashGrid.cell_size_for_radius(
            self.settings.detection.local_radius_mm
        )
        if self.grid is not None and self.grid.cell_size == cell:
            return self.grid
        source = self._source_cloud_path()
        if source:
            cached = load_voxel_grid(
                self.project_dir, self.full_points, source, cell
            )
            if cached is not None:
                self.grid = cached
                self._status("spatial index ready (cached)")
                return self.grid
        self._status(
            f"building spatial index ({len(self.full_points):,} pts, "
            f"cell={cell:.1f} mm) ..."
        )
        QApplication.processEvents()
        self.grid = VoxelHashGrid(self.full_points, cell_size=cell)
        if source:
            save_voxel_grid(self.project_dir, self.grid, source)
        self._status("spatial index ready")
        return self.grid

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _ensure_base_display_xyz(self) -> np.ndarray:
        if self._base_display_xyz is not None:
            return self._base_display_xyz
        v = self.settings.view
        voxel = float(v.display_voxel_size_mm)
        max_points = int(v.display_max_points)
        backend = v.display_downsample_backend
        source = self._source_cloud_path()
        if source:
            cached = load_display_xyz(
                self.project_dir,
                len(self.full_points),
                source,
                voxel_size=voxel,
                max_points=max_points,
                backend=backend,
            )
            if cached is not None:
                self._base_display_xyz = cached
                self._status("display ready (cached)")
                return self._base_display_xyz
        resolved = resolve_display_backend(backend)
        self._status(
            f"decimating {len(self.full_points):,} points for display "
            f"({resolved}) ..."
        )
        QApplication.processEvents()
        xyz = display_xyz(
            self.full_points,
            voxel,
            max_points,
            backend=backend,
        )
        self._base_display_xyz = xyz
        if source:
            save_display_xyz(
                self.project_dir,
                xyz,
                source,
                len(self.full_points),
                voxel_size=voxel,
                max_points=max_points,
                backend=backend,
            )
        return self._base_display_xyz

    def _camera_ray_through(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (camera_origin, unit direction toward ``world``)."""
        cam = np.asarray(self.plotter.camera_position[0], dtype=np.float64)
        world = np.asarray(world, dtype=np.float64)
        direction = world - cam
        n = float(np.linalg.norm(direction))
        if n == 0.0:
            # Fallback: camera look direction.
            focus = np.asarray(self.plotter.camera_position[1], dtype=np.float64)
            direction = focus - cam
            n = float(np.linalg.norm(direction)) or 1.0
        return cam, direction / n

    def _build_pick_layers(self, world: np.ndarray) -> list[dict]:
        """Depth layers along the camera ray, ordered front to back.

        The raw VTK hit is unreliable for point clouds: dots are rendered
        one pixel wide, so a click between them reports a surface further
        back. The hit still lies on the view ray, so re-clustering the
        cloud along that ray recovers the surface the user actually sees.
        """
        origin, direction = self._camera_ray_through(world)
        # The displayed (decimated) cloud is cheap enough to scan on every
        # pick, but decimation thins each surface, so the narrowest cylinder
        # may hold too few points to seed from. Widen only until the front
        # layer is solid: a wider cylinder also sweeps in neighbouring
        # geometry and invents surfaces that are not under the cursor.
        xyz = self._to_view_points(self._ensure_base_display_xyz())
        base_r = max(self.settings.detection.local_radius_mm * 0.3, 2.0)
        best: list[dict] = []
        for scale in (1.0, 2.0, 4.0, 8.0):
            layers = depth_layers_along_ray(
                xyz,
                origin,
                direction,
                cylinder_radius_mm=base_r * scale,
                min_points=6,
            )
            if len(layers) > len(best):
                best = layers
            if layers and layers[0]["n_points"] >= 15:
                best = layers
                break
        if best:
            return best
        # Fall back to the raw VTK hit so picking still works.
        return [
            {
                "seed": np.asarray(world, dtype=np.float64),
                "depth_mm": 0.0,
                "n_points": 1,
            }
        ]

    def _on_pick(self, picked_point, *_):
        world = np.asarray(picked_point, dtype=np.float64)
        self._guard(lambda: self._start_pick_at(world))

    def _start_pick_at(self, world: np.ndarray):
        """Fresh screen pick: snap to the frontmost surface on the view ray."""
        if len(self.full_points) == 0:
            raise ValueError("load a cloud first")
        wall_t0 = time.perf_counter()
        if self.snap_front_cb.isChecked():
            self._pick_layers = self._build_pick_layers(world)
        else:
            self._pick_layers = [
                {
                    "seed": np.asarray(world, dtype=np.float64),
                    "depth_mm": 0.0,
                    "n_points": 1,
                }
            ]
        self._pick_layer_i = 0  # frontmost = what the user sees
        self._pick_replace_gid = None
        self._pick_raw_hit = np.asarray(world, dtype=np.float64)
        self._extract_at_current_layer(replace=False, wall_t0=wall_t0)

    def _cycle_pick_depth(self, delta: int):
        if not self._pick_layers:
            raise ValueError("no depth candidates — press P to pick first")
        n = len(self._pick_layers)
        if n == 1:
            raise ValueError(
                "only one surface on the view ray — nothing to step with < >"
            )
        new_i = int(np.clip(self._pick_layer_i + delta, 0, n - 1))
        if new_i == self._pick_layer_i:
            raise ValueError(
                f"already at the {'nearest' if delta < 0 else 'farthest'} "
                f"surface ({n} candidates)"
            )
        self._pick_layer_i = new_i
        self._extract_at_current_layer(replace=True)

    def _update_depth_controls(self):
        layers = self._pick_layers
        n = len(layers)
        i = self._pick_layer_i
        self._update_pick_hint(n)
        if n == 0:
            self.depth_label.setText("— / —")
            self.depth_dots.setText("○")
            self.depth_meta.setText("P")
            self.depth_label.setToolTip(DEPTH_TIP)
        elif n == 1:
            self.depth_label.setText("1 / 1")
            self.depth_dots.setText("●")
            self.depth_meta.setText("")
            self.depth_label.setToolTip(
                DEPTH_TIP
                + "\n\n"
                + f"1: {layers[0]['depth_mm']:.0f} mm from camera, "
                + f"{layers[0]['n_points']:,} pts"
            )
        else:
            gap = layers[i + 1]["depth_mm"] - layers[i]["depth_mm"] if i + 1 < n else None
            self.depth_label.setText(f"{i + 1} / {n}")
            self.depth_dots.setText("  ".join("●" if k == i else "○" for k in range(n)))
            self.depth_meta.setText("" if gap is None else f"+{gap:.0f} mm")
            self.depth_label.setToolTip(
                DEPTH_TIP
                + "\n\n"
                + "\n".join(
                    f"{k + 1}: {L['depth_mm']:.0f} mm from camera, "
                    f"{L['n_points']:,} pts"
                    + (" <- current" if k == i else "")
                    for k, L in enumerate(layers)
                )
            )
        self.depth_prev_btn.setEnabled(n > 1 and i > 0)
        self.depth_next_btn.setEnabled(n > 1 and i < n - 1)

    def _update_pick_hint(self, n_layers: int):
        """Keep the 3D overlay minimal until depth navigation is available."""
        text = "P : Pick"
        if n_layers > 1:
            text += "\n< : Nearer\n> : Farther"
        actor = (getattr(self.plotter, "actors", None) or {}).get(
            "_point_picking_message"
        )
        if actor is not None:
            # add_text uses vtkCornerAnnotation; slot 2 is upper-left.
            actor.SetText(2, text)

    def _extract_at_current_layer(self, *, replace: bool, wall_t0: float | None = None):
        flow_t0 = wall_t0 if wall_t0 is not None else time.perf_counter()
        self._update_depth_controls()
        layer = self._pick_layers[self._pick_layer_i]
        world_view = np.asarray(layer["seed"], dtype=np.float64)
        n_layers = len(self._pick_layers)
        depth_tag = f"surface {self._pick_layer_i + 1}/{n_layers}"
        raw = getattr(self, "_pick_raw_hit", None)
        if raw is not None:
            moved = float(np.linalg.norm(world_view - raw))
            if moved > 1.0:
                depth_tag += f", snapped {moved:.0f} mm nearer"
        if n_layers > 1:
            depth_tag += " (> farther / < nearer)"
        world = self._to_survey_point(world_view)

        if replace and self._pick_replace_gid is not None:
            # Reuse the same group id so cycling does not burn ids.
            old = self._get_group(self._pick_replace_gid)
            if old is not None:
                self.plotter.remove_actor(
                    f"group_{old['id']:03d}", render=False
                )
                self.groups = [x for x in self.groups if x["id"] != old["id"]]
            reuse_gid = self._pick_replace_gid
        else:
            reuse_gid = None

        t_pick0 = time.perf_counter()
        depth_s = (t_pick0 - wall_t0) if wall_t0 is not None else None
        want_cell = VoxelHashGrid.cell_size_for_radius(
            self.settings.detection.local_radius_mm
        )
        had_grid = (
            self.grid is not None and self.grid.cell_size == want_cell
        )
        t_grid0 = time.perf_counter()
        grid = self._ensure_grid()
        grid_build_s = 0.0 if had_grid else time.perf_counter() - t_grid0
        self._status(f"picking ({depth_tag}): local plane + refine ...")
        QApplication.processEvents()
        t_query0 = time.perf_counter()
        nb = grid.radius_indices(world, self.settings.detection.local_radius_mm)
        neighbor_query_s = time.perf_counter() - t_query0
        neighbor_s = grid_build_s + neighbor_query_s
        pick_detail: dict = {
            "grid_build_s": grid_build_s,
            "neighbor_query_s": neighbor_query_s,
            "neighbor_s": neighbor_s,
            "n_neighbors": len(nb),
        }
        compute = resolve_compute_backend(
            self.settings.detection.compute_backend, n_points=len(self.full_points)
        )
        indices, plane = pick_plane_region(
            self.full_points,
            world,
            nb,
            self.settings.detection,
            compute_backend=compute,
            timings=pick_detail,
        )
        pick_s = time.perf_counter() - t_pick0

        if (
            not replace
            and self.append_cb.isChecked()
            and self.active_group_id is not None
        ):
            g = self._active_group()
            before = len(g["indices"])
            g["indices"] = np.union1d(g["indices"], indices)
            g["clicked"] = world
            g["coarse_plane"] = plane.as_array()
            g["fit"] = None
            added = len(g["indices"]) - before
            action = (
                f"appended {added:,} pts to {g['name']} "
                f"(total {len(g['indices']):,})"
            )
            self._pick_replace_gid = None  # append is not depth-replaceable
        else:
            gid = reuse_gid if reuse_gid is not None else self.next_group_id
            if reuse_gid is None:
                self.next_group_id += 1
            g = {
                "id": gid,
                "name": f"G{gid}",
                "visible": True,
                "color": group_color(gid),
                "clicked": world,
                "coarse_plane": plane.as_array(),
                "indices": np.asarray(indices, dtype=np.int64),
                "fit": None,
            }
            self.groups.append(g)
            self.active_group_id = gid
            self._pick_replace_gid = gid
            action = f"added {g['name']} with {len(indices):,} points"

        pts = self.full_points[g["indices"]]
        coarse_mad = float(mad_sigma(plane.signed_distances(pts)))
        coarse_msg = f"{action} | {depth_tag} | coarse mad≈{coarse_mad * 1e3:.0f} µm"
        self._status(coarse_msg)
        QApplication.processEvents()

        self._refresh_group_actors()
        self._refresh_tree()

        if self.autofit_cb.isChecked():
            n_g = len(g["indices"])
            compute = resolve_compute_backend(
                self.settings.detection.compute_backend, n_points=n_g
            )
            self._status(f"{coarse_msg} → fitting {n_g:,} pts ({compute}) ...")
            QApplication.processEvents()
            timing = self._fit_group(g, log=False)
            t_post0 = time.perf_counter()
            self._active_plane_index = 0
            self._refresh_tree()
            self._show_uv_for_selection()
            timing["pick_s"] = pick_s
            if depth_s is not None:
                timing["depth_s"] = depth_s
            timing["post_s"] = time.perf_counter() - t_post0
            timing["wall_s"] = time.perf_counter() - flow_t0
            timing["pick_detail"] = pick_detail
            self._log_fit_timing(timing, kind="pick+fit")
            planes = g["fit"]["planes"]
            self._status(
                f"{g['name']}: {len(planes)} plane(s) [{compute}, "
                f"{self._fit_timing_status(timing)}]: "
                + " | ".join(
                    f"p{p['plane_index']} {p['mad_sigma_mm']*1e3:.0f}um {p['status']}"
                    + (" BIMODAL" if p["bimodal"] else "")
                    for p in planes
                )
                + f"  | {depth_tag}  |  fit.log"
            )
        else:
            timing = {
                "group": g["name"],
                "n_pts": len(g["indices"]),
                "compute": resolve_compute_backend(
                    self.settings.detection.compute_backend, n_points=len(g["indices"])
                ),
                "ransac_backend": self.settings.detection.ransac_backend,
                "multi": False,
                "fit_s": 0.0,
                "uv_s": 0.0,
                "total_s": 0.0,
                "pick_s": pick_s,
                "pick_detail": pick_detail,
                "planes": [],
            }
            if depth_s is not None:
                timing["depth_s"] = depth_s
            timing["wall_s"] = time.perf_counter() - flow_t0
            self._log_fit_timing(timing, kind="pick")
            self._status(
                f"{coarse_msg}  |  pick {pick_s:.2f}s"
                + (f" + depth {depth_s:.2f}s" if depth_s is not None else "")
                + f" = {timing['wall_s']:.2f}s  |  fit.log"
            )

    def _handle_pick(self, world: np.ndarray):
        # Kept for compatibility with any external callers / tests.
        self._start_pick_at(world)

    def _fit_group(self, g, *, log: bool = True) -> dict:
        from dataclasses import replace

        t0 = time.perf_counter()
        pts = self.full_points[g["indices"]]
        n_pts = len(pts)
        backend = self.settings.detection.ransac_backend
        compute = resolve_compute_backend(
            self.settings.detection.compute_backend, n_points=n_pts
        )
        multi = self.multiplane_cb.isChecked()
        if multi:
            mp = MultiPlaneParams()
            mp = MultiPlaneParams(
                plane=replace(
                    mp.plane,
                    ransac_backend=backend,
                    max_threshold_mm=FIT_MAX_THRESHOLD_MM,
                    compute_backend=compute,
                )
            )
            extracted = extract_planes(
                pts, mp, clicked=g["clicked"], coarse_plane=g["coarse_plane"]
            )
        else:
            res = extract_main_plane(
                pts,
                MainPlaneParams(
                    ransac_backend=backend,
                    max_threshold_mm=FIT_MAX_THRESHOLD_MM,
                    compute_backend=compute,
                ),
                clicked=g["clicked"],
                coarse_plane=g["coarse_plane"],
            )
            extracted = [{
                "plane_index": 0,
                "result": res,
                "mask": res.main_mask,
                "n_points": res.n_main,
                "bimodal": bool(
                    bimodality_flag(
                        res.plane.signed_distances(pts[res.main_mask]),
                        res.fit.stats_inliers["mad_sigma"],
                    )
                ),
            }]
        t_fit = time.perf_counter()
        g["fit"] = {
            "planes": [
                {
                    "plane_index": p["plane_index"],
                    "abcd": p["result"].plane.as_array().tolist(),
                    "n_points": p["n_points"],
                    "status": p["result"].status,
                    "reasons": p["result"].reasons,
                    "bimodal": p["bimodal"],
                    "mad_sigma_mm": p["result"].fit.stats_inliers["mad_sigma"],
                    "threshold_mm": p["result"].fit.threshold,
                    "inlier_local": np.flatnonzero(p["mask"]).astype(np.int64),
                }
                for p in extracted
            ]
        }
        for entry, p in zip(g["fit"]["planes"], extracted):
            self._cache_uv_for_plane(pts, entry, p["mask"])
        t_end = time.perf_counter()
        timing = {
            "group": g["name"],
            "n_pts": n_pts,
            "compute": compute,
            "ransac_backend": backend,
            "multi": multi,
            "fit_s": t_fit - t0,
            "uv_s": t_end - t_fit,
            "total_s": t_end - t0,
            "planes": g["fit"]["planes"],
        }
        if log:
            self._log_fit_timing(timing)
        return timing

    def _fit_active(self):
        g = self._active_group()
        if g is None:
            raise ValueError("no active group")
        n_pts = len(g["indices"])
        compute = resolve_compute_backend(
            self.settings.detection.compute_backend, n_points=n_pts
        )
        wall_t0 = time.perf_counter()
        self._status(f"fitting {g['name']} ({n_pts:,} pts, {compute}) ...")
        QApplication.processEvents()
        timing = self._fit_group(g, log=False)
        t_post0 = time.perf_counter()
        self._active_plane_index = 0
        self._refresh_tree()
        self._show_uv_for_selection()
        timing["post_s"] = time.perf_counter() - t_post0
        timing["wall_s"] = time.perf_counter() - wall_t0
        self._log_fit_timing(timing)
        planes = g["fit"]["planes"]
        self._status(
            f"{g['name']}: {len(planes)} plane(s) [{compute}, "
            f"{self._fit_timing_status(timing)}]: "
            + " | ".join(
                f"p{p['plane_index']} {p['mad_sigma_mm']*1e3:.0f}um {p['status']}"
                + (" BIMODAL" if p["bimodal"] else "")
                for p in planes
            )
            + "  |  fit.log"
        )
        self._sync_action_states()

    def _fit_all(self):
        totals: list[dict] = []
        all_t0 = time.perf_counter()
        for g in self.groups:
            n_pts = len(g["indices"])
            compute = resolve_compute_backend(
                self.settings.detection.compute_backend, n_points=n_pts
            )
            self._status(f"fitting {g['name']} ({n_pts:,} pts, {compute}) ...")
            QApplication.processEvents()
            wall_t0 = time.perf_counter()
            timing = self._fit_group(g, log=False)
            timing["wall_s"] = time.perf_counter() - wall_t0
            self._log_fit_timing(timing)
            totals.append(timing)
        self._refresh_tree()
        self._show_uv_for_selection()
        total_s = sum(t["wall_s"] for t in totals)
        self._append_fit_log(
            f"fit_all  groups={len(totals)}  wall={total_s:.3f}s  "
            f"session={time.perf_counter() - all_t0:.3f}s"
        )
        self._status(
            f"fit all done ({len(totals)} groups, {total_s:.1f}s) — see fit.log"
        )
        self._sync_action_states()

    def _save_all(self):
        if not self.groups:
            raise ValueError("no groups to save")
        for g in self.groups:
            if g["fit"] is None:
                self._status(f"fitting {g['name']} ...")
                QApplication.processEvents()
                self._fit_group(g)
            save_group(
                self.project_dir, g["id"], g["name"],
                points=self.full_points[g["indices"]],
                indices=g["indices"],
                coarse_plane=g["coarse_plane"],
                clicked=g["clicked"],
                color=g["color"],
                detection=self.settings.detection,
                fit_summary=g["fit"],
            )
        write_manifest(
            self.project_dir,
            SourceInfo(
                path=self.pcd_path,
                n_points=len(self.full_points),
                size_bytes=os.path.getsize(self.pcd_path) if self.pcd_path else None,
            ),
            self.settings.detection,
            n_groups=len(self.groups),
        )
        self._refresh_tree()
        self._status(f"saved {len(self.groups)} groups to {self.project_dir / 'groups'}")
        self._sync_action_states()

    def _load_group_fit(
        self, group_id: int, group_indices: np.ndarray | None
    ) -> dict | None:
        """Restore fit.planes from group JSON, attaching inlier arrays."""
        doc = load_group_doc(self.project_dir, group_id)
        if not doc:
            return None
        fit = doc.get("fit")
        if not isinstance(fit, dict) or not isinstance(fit.get("planes"), list):
            return None
        gidx = None if group_indices is None else np.asarray(group_indices, dtype=np.int64)
        lookup = None if gidx is None else {int(v): i for i, v in enumerate(gidx)}
        planes = []
        for p in fit["planes"]:
            if not isinstance(p, dict) or "abcd" not in p:
                continue
            entry = dict(p)
            pi = int(entry.get("plane_index", 0))
            src = load_plane_inlier_indices(self.project_dir, group_id, pi)
            if src is not None:
                src = np.asarray(src, dtype=np.int64)
                entry["inlier_source"] = src
                entry["inlier_n"] = int(len(src))
                if lookup is not None:
                    local = np.array(
                        [lookup[int(s)] for s in src if int(s) in lookup],
                        dtype=np.int64,
                    )
                    if len(local) == len(src):
                        entry["inlier_local"] = local
            planes.append(entry)
        return {"planes": planes} if planes else None

    def _load_all(self):
        if len(self.full_points) == 0:
            raise ValueError("load the source cloud first")
        manifest = read_manifest(self.project_dir)
        if manifest is not None:
            src_n = manifest.get("source", {}).get("n_points")
            if src_n is not None and src_n != len(self.full_points):
                raise ValueError(
                    f"source cloud mismatch: saved={src_n:,}, "
                    f"current={len(self.full_points):,}"
                )
        infos = load_groups(self.project_dir)
        self._clear_all()
        for info in infos:
            indices = load_group_indices(self.project_dir, info.group_id)
            if indices is None:
                self._status(f"{info.name}: no indices file, skipped")
                continue
            if len(indices) and indices.max() >= len(self.full_points):
                raise ValueError(f"{info.name}: indices exceed current cloud")
            self.groups.append({
                "id": info.group_id,
                "name": info.name,
                "visible": True,
                "color": group_color(info.group_id),
                "clicked": info.clicked,
                "coarse_plane": info.coarse_plane,
                "indices": indices,
                "fit": self._load_group_fit(info.group_id, indices),
            })
        if self.groups:
            ids = sorted(g["id"] for g in self.groups)
            self.active_group_id = ids[0]
            self.next_group_id = ids[-1] + 1
        self._refresh_group_actors()
        self._refresh_tree()
        recipe_note = ""
        default_recipe = self.project_dir / "recipe.json"
        if default_recipe.is_file():
            try:
                n_ent = self._reduction_load_recipe_path(default_recipe, confirm=False)
                recipe_note = f", recipe.json ({n_ent} entities)"
            except Exception as e:
                traceback.print_exc()
                recipe_note = f", recipe.json skipped ({e})"
                self._reduction_refresh_operand_combos()
        else:
            self._reduction_refresh_operand_combos()
        self._status(f"loaded {len(self.groups)} groups{recipe_note}")
        self._sync_action_states()

    def _selected_group_ids(self) -> list[int]:
        ids: list[int] = []
        for item in self.tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if data and data[0] in ("group", "plane") and data[1] not in ids:
                ids.append(data[1])
        return sorted(ids)

    def _merge_selected(self):
        """Fuse groups that cover one physical face into a single group."""
        ids = self._selected_group_ids()
        if len(ids) < 2:
            raise ValueError(
                "select two or more groups in the tree (Ctrl/Shift+click) to merge"
            )
        target = self._get_group(ids[0])
        for gid in ids[1:]:
            other = self._get_group(gid)
            if other is None:
                continue
            target["indices"] = np.union1d(target["indices"], other["indices"])
            self.plotter.remove_actor(f"group_{gid:03d}", render=False)
            self.groups = [x for x in self.groups if x["id"] != gid]
        target["fit"] = None
        self.active_group_id = target["id"]
        self._active_plane_index = 0
        self._refresh_group_actors()
        self._refresh_tree()
        self._show_uv_for_selection()
        self._status(
            f"merged {len(ids)} groups into {target['name']} "
            f"({len(target['indices']):,} pts) — press Fit to refit"
        )
        self._sync_action_states()

    def _delete_active(self):
        if hasattr(self, "rd_tree") and self.rd_tree.hasFocus():
            self._reduction_delete_selected()
            return
        plane_sel: list[tuple[int, int]] = []
        group_sel: list[int] = []
        if hasattr(self, "tree"):
            for item in self.tree.selectedItems():
                data = item.data(0, Qt.UserRole)
                if not data:
                    continue
                if data[0] == "plane":
                    plane_sel.append((int(data[1]), int(data[2])))
                elif data[0] == "group":
                    group_sel.append(int(data[1]))
        if plane_sel:
            self._delete_planes(plane_sel)
            return
        gids = group_sel or (
            [self.active_group_id] if self.active_group_id is not None else []
        )
        if not gids:
            raise ValueError("no active group")
        self._delete_groups(gids)

    def _delete_planes(self, targets: list[tuple[int, int]]):
        by_group: dict[int, set[int]] = {}
        for gid, pi in targets:
            by_group.setdefault(gid, set()).add(int(pi))
        labels = []
        last_gid = None
        for gid, pis in by_group.items():
            g = self._get_group(gid)
            if g is None or not g.get("fit"):
                continue
            planes = g["fit"].get("planes") or []
            gone = [p for p in planes if int(p.get("plane_index", 0)) in pis]
            keep = [p for p in planes if int(p.get("plane_index", 0)) not in pis]
            if not gone:
                continue
            labels.extend(f"{g['name']}/{plane_label(p)}" for p in gone)
            last_gid = gid
            if keep:
                g["fit"]["planes"] = keep
                if gid == self.active_group_id:
                    cur = int(self._active_plane_index)
                    if cur in pis:
                        self._active_plane_index = int(keep[0].get("plane_index", 0))
                    self._tree_focus = "plane"
            else:
                g["fit"] = None
                if gid == self.active_group_id:
                    self._active_plane_index = 0
                    self._tree_focus = "group"
        if last_gid is not None:
            self.active_group_id = last_gid
        self._uv_map_mode = "base"
        self._refresh_group_actors()
        self._refresh_tree()
        self._show_uv_for_selection()
        self._sync_action_states()
        if labels:
            self._status("deleted " + ", ".join(labels))

    def _delete_groups(self, gids: list[int]):
        gids = sorted(set(gids))
        names = []
        for gid in gids:
            g = self._get_group(gid)
            if g is None:
                continue
            names.append(g["name"])
            self.plotter.remove_actor(f"group_{gid:03d}", render=False)
            self.groups = [x for x in self.groups if x["id"] != gid]
        if not names:
            raise ValueError("no active group")
        if self.active_group_id in gids or self._get_group(self.active_group_id) is None:
            self.active_group_id = (
                sorted(x["id"] for x in self.groups)[0] if self.groups else None
            )
            self._active_plane_index = 0
            self._tree_focus = "group"
        self._refresh_group_actors()
        self._refresh_tree()
        self._show_uv_for_selection()
        self._sync_action_states()
        self._status("deleted " + ", ".join(names))

    def _clear_all(self):
        was_aligned = self._view_frame is not None
        self._view_frame = None
        self._update_frame_controls()
        self._rebuild_status_default()
        for g in self.groups:
            self.plotter.remove_actor(f"group_{g['id']:03d}", render=False)
        self.groups = []
        self.active_group_id = None
        self.next_group_id = 0
        if hasattr(self, "_reduction"):
            self._reduction.clear()
            self._clear_reduction_preview()
            self._clear_reduction_actors()
            self._clear_measure_overlays()
            if hasattr(self, "rd_tree"):
                self._refresh_reduction_tree()
                self._refresh_measure_tree()
                self._reduction_refresh_operand_combos()
        self._refresh_group_actors()
        self._refresh_tree()
        self._show_uv_for_selection()
        self._sync_action_states()
        if was_aligned:
            self._refresh_base_actor()

    # ------------------------------------------------------------------
    # tree
    # ------------------------------------------------------------------

    def _refresh_tree(self):
        self.tree.blockSignals(True)
        _reset_tree_widget(self.tree)
        for g in sorted(self.groups, key=lambda x: x["id"]):
            item = QTreeWidgetItem([g["name"], "", f"{len(g['indices']):,}", ""])
            item.setData(0, Qt.UserRole, ("group", g["id"]))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(0, Qt.Checked if g["visible"] else Qt.Unchecked)
            c = g["color"]
            item.setBackground(
                0, QColor.fromRgbF(float(c[0]), float(c[1]), float(c[2]))
            )
            # Keep name readable on saturated group colors.
            luminance = 0.299 * float(c[0]) + 0.587 * float(c[1]) + 0.114 * float(c[2])
            item.setForeground(
                0, QColor(0, 0, 0) if luminance > 0.55 else QColor(255, 255, 255)
            )
            if g["id"] == self.active_group_id:
                f = item.font(0)
                f.setBold(True)
                item.setFont(0, f)
            if g["fit"] is not None:
                for p in g["fit"]["planes"]:
                    abcd = p["abcd"]
                    # When Align Z is active, rewrite the plane equation into the
                    # current view frame so n,d match what the user sees.
                    if self._view_frame is not None:
                        plane = Plane.from_array(abcd)
                        plane = self._view_frame.apply_plane(plane)
                        abcd = plane.as_array()
                    label = plane_label(p)
                    nxyz = (
                        f"n=({abcd[0]:+.4f}, {abcd[1]:+.4f}, {abcd[2]:+.4f})  "
                        f"d={abcd[3]:+.3f}"
                    )
                    qflag = "PASS" if p["status"] == "ok" else "WARN"
                    quality = (
                        f"{qflag}  {p['mad_sigma_mm']*1e3:.0f}um"
                        + (" BIMODAL" if p["bimodal"] else "")
                    )
                    if "selection_refit" in (p.get("reasons") or []):
                        src = p.get("source_plane_index")
                        src_p = self._find_plane(g, src) if src is not None else None
                        src_label = plane_label(src_p) if src_p is not None else (
                            f"p{src}" if src is not None else "selection"
                        )
                        quality += f"  ·  from {src_label}"
                    child = QTreeWidgetItem(
                        [label, nxyz, f"{p['n_points']:,}", quality]
                    )
                    child.setData(0, Qt.UserRole, ("plane", g["id"], p["plane_index"]))
                    child.setFlags(
                        (child.flags() | Qt.ItemIsEditable) & ~Qt.ItemIsUserCheckable
                    )
                    for col in range(4):
                        child.setToolTip(col, nxyz)
                    if qflag == "PASS":
                        child.setForeground(3, QColor(31, 122, 31))
                    else:
                        child.setForeground(3, QColor(160, 90, 0))
                    item.addChild(child)
            self.tree.addTopLevelItem(item)
            item.setExpanded(True)
        self.tree.blockSignals(False)
        self._restore_tree_selection()
        self._sync_action_states()
        self._reduction_fill_bind_combo()

    def _restore_tree_selection(self):
        gid = self.active_group_id
        if gid is None:
            return
        pi = self._active_plane_index
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            if not data or data[1] != gid:
                continue
            chosen = item
            if self._tree_focus == "plane":
                for j in range(item.childCount()):
                    ch = item.child(j)
                    cd = ch.data(0, Qt.UserRole)
                    if cd and cd[0] == "plane" and int(cd[2]) == int(pi):
                        chosen = ch
                        break
            self.tree.setCurrentItem(chosen)
            return

    def _on_item_selected(self, current, _prev):
        if current is None:
            self._sync_action_states()
            self._show_uv_for_selection()
            self._reduction_fill_bind_combo()
            return
        data = current.data(0, Qt.UserRole)
        if data and data[0] in ("group", "plane"):
            gid = data[1]
            if data[0] == "plane":
                self._active_plane_index = int(data[2])
                self._tree_focus = "plane"
            else:
                self._active_plane_index = 0
                self._tree_focus = "group"
            if gid != self.active_group_id:
                self.active_group_id = gid
                self._refresh_group_actors()
                self._refresh_tree()
            self._show_uv_for_selection()
        self._sync_action_states()
        self._reduction_fill_bind_combo()
        self._reduction_fill_bind_combo()

    def _on_item_changed(self, item, column):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        if data[0] == "plane":
            if column != 0:
                if column == 1:
                    g = self._get_group(data[1])
                    p = self._find_plane(g, data[2])
                    if p is not None:
                        abcd = p["abcd"]
                        if self._view_frame is not None:
                            plane = Plane.from_array(abcd)
                            plane = self._view_frame.apply_plane(plane)
                            abcd = plane.as_array()
                        self.tree.blockSignals(True)
                        item.setText(
                            1,
                            f"n=({abcd[0]:+.4f}, {abcd[1]:+.4f}, {abcd[2]:+.4f})  "
                            f"d={abcd[3]:+.3f}",
                        )
                        self.tree.blockSignals(False)
                return
            g = self._get_group(data[1])
            p = self._find_plane(g, data[2])
            if g is None or p is None:
                return
            raw = item.text(0).strip().replace("/", "-")
            default = f"p{int(p.get('plane_index', 0))}"
            new_name = raw or default
            for other in (g.get("fit") or {}).get("planes") or []:
                if other is p:
                    continue
                if plane_label(other) == new_name:
                    self.tree.blockSignals(True)
                    item.setText(0, plane_label(p))
                    self.tree.blockSignals(False)
                    self._status(f"name {new_name!r} already used on {g['name']}")
                    return
            if new_name == default:
                p.pop("name", None)
            else:
                p["name"] = new_name
            if item.text(0) != new_name:
                self.tree.blockSignals(True)
                item.setText(0, new_name)
                self.tree.blockSignals(False)
            self._reduction_fill_bind_combo()
            if (
                g["id"] == self.active_group_id
                and int(p.get("plane_index", 0)) == int(self._active_plane_index)
            ):
                self._show_uv_for_selection()
            return
        if data[0] != "group":
            return
        g = self._get_group(data[1])
        if g is None:
            return
        g["visible"] = item.checkState(0) == Qt.Checked
        new_name = item.text(0).strip()
        if new_name and new_name != g["name"]:
            g["name"] = new_name
        self._refresh_group_actor(g)
        self.plotter.render()
