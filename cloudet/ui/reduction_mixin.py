"""Reduction dock, measure dock, and interactive geometry reduction logic."""

from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QMessageBox,
    QAbstractItemView,
)

from cloudet.frame import (
    ALIGNED_AXIS_IDS,
    ALIGNED_AXIS_LABELS,
    RigidFrame,
    is_aligned_axis_id,
    transform_record,
)
from cloudet.geometry import (
    axis_arrow_points,
    line_segment_points,
    plane_patch_corners,
    project_point_to_line,
    project_point_to_plane,
)
from cloudet.plane import Plane
from cloudet.reduce import (
    export_reduction_result,
    load_recipe,
    preview_construct_step,
    scanned_plane_record,
    write_geometry_json,
    write_recipe_json,
)
from cloudet.reduction_ops import (
    GUI_APPLY_LABELS,
    GUI_ID_PREFIX,
    GUI_MENU_ITEMS,
    GUI_PAGE_INDEX,
    MEASURE_MENU_ITEMS,
    MEASURE_OP_BY_KEY,
    REDUCTION_OP_BY_GUI,
    build_construct_step,
    form_values_from_step,
)
from cloudet.ui.constants import (
    RD_ALIGNED_AXIS,
    RD_AXIS,
    RD_GUI_TO_RECIPE_OP,
    RD_KIND_LABEL,
    RD_MEASURE,
    RD_NORMAL,
    RD_PLANE_OFFSET,
    RD_PLANE_SCANNED,
    RD_POINT,
    RD_RECIPE_TO_GUI_OP,
    RD_SELECTED_RING,
)
from cloudet.ui.plane_labels import _plane_id_token, _plane_label
from cloudet.ui.widgets import UI_STYLE, _line_tube_mesh, _make_collapsible_card, _reset_combo, _reset_tree_widget


class ReductionMixin:
    """Reduction / measure docks and recipe-driven geometry construction."""

    def _build_reduction_dock(self):
        dock = QDockWidget("Reduction", self)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        dock.setStyleSheet(UI_STYLE)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)

        title = QLabel("REDUCTION")
        title.setObjectName("sectionTitle")
        lay.addWidget(title)

        hint = QLabel(
            "Choose an operation first. Only the controls for that step are shown."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        def _mini_card(heading: str, *, expanded: bool = True) -> tuple[QFrame, QVBoxLayout]:
            return _make_collapsible_card(heading, expanded=expanded)

        # ---- Operation + its inputs (one card; stack follows the combo) --
        op_card, op_lay = _mini_card("OPERATION")
        self.rd_op_combo = QComboBox()
        for label, key in GUI_MENU_ITEMS:
            self.rd_op_combo.addItem(label, key)
        self.rd_op_combo.currentIndexChanged.connect(self._reduction_on_op_changed)
        op_lay.addWidget(self.rd_op_combo)

        self.rd_stack = QStackedWidget()

        # Shared id field lives above the stack so every op can name the result.
        id_form = QFormLayout()
        id_form.setContentsMargins(0, 0, 0, 0)
        id_form.setSpacing(4)
        self.rd_id_edit = QLineEdit()
        self.rd_id_edit.setPlaceholderText("result id (optional, auto if empty)")
        id_form.addRow("New id", self.rd_id_edit)
        self.rd_id_form = id_form
        op_lay.addLayout(id_form)

        def _entity_combo() -> QComboBox:
            c = QComboBox()
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(12)
            c.currentIndexChanged.connect(self._reduction_on_operand_combo)
            return c

        def _form_page() -> tuple[QWidget, QFormLayout]:
            page = QWidget()
            outer = QVBoxLayout(page)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(6)
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(4)
            outer.addLayout(form)
            outer.addStretch(1)
            return page, form

        # page: bind
        bind_page, bind_form = _form_page()
        self.rd_bind_combo = _entity_combo()
        bind_form.addRow("Groups plane", self.rd_bind_combo)
        self.rd_bind_hint = QLabel(
            "Choose a fitted Groups plane (G6/p0, G6/front, …). "
            "Imported as G6_p0 or G6_front unless you set New id."
        )
        self.rd_bind_hint.setObjectName("muted")
        self.rd_bind_hint.setWordWrap(True)
        bind_page.layout().insertWidget(1, self.rd_bind_hint)
        self.rd_stack.addWidget(bind_page)  # index 0

        # page: offset
        off_page, off_form = _form_page()
        self.rd_offset_plane = _entity_combo()
        off_form.addRow("Plane", self.rd_offset_plane)
        self.rd_offset_spin = QDoubleSpinBox()
        self.rd_offset_spin.setRange(-1.0e6, 1.0e6)
        self.rd_offset_spin.setDecimals(3)
        self.rd_offset_spin.setSingleStep(0.1)
        self.rd_offset_spin.setSuffix(" mm")
        self.rd_offset_spin.setValue(12.0)
        self.rd_offset_spin.valueChanged.connect(self._reduction_on_offset_spin)
        off_form.addRow("Distance", self.rd_offset_spin)
        self.rd_offset_slider = QSlider(Qt.Horizontal)
        self.rd_offset_slider.setRange(-5000, 5000)
        self.rd_offset_slider.setValue(120)
        self.rd_offset_slider.setTickPosition(QSlider.TicksBelow)
        self.rd_offset_slider.setTickInterval(1000)
        self.rd_offset_slider.valueChanged.connect(self._reduction_on_offset_slider)
        off_form.addRow("", self.rd_offset_slider)
        self.rd_offset_range_label = QLabel(
            "slider ±500 mm (0.1 mm steps). Preview updates live."
        )
        self.rd_offset_range_label.setObjectName("muted")
        off_form.addRow("", self.rd_offset_range_label)
        self.rd_stack.addWidget(off_page)  # index 1

        # page: intersect 2 planes
        p2, p2_form = _form_page()
        self.rd_p2_a = _entity_combo()
        self.rd_p2_b = _entity_combo()
        p2_form.addRow("Plane A", self.rd_p2_a)
        p2_form.addRow("Plane B", self.rd_p2_b)
        self.rd_stack.addWidget(p2)  # 2

        # page: line ∩ plane
        lp, lp_form = _form_page()
        self.rd_lp_line = _entity_combo()
        self.rd_lp_plane = _entity_combo()
        lp_form.addRow("Axis", self.rd_lp_line)
        lp_form.addRow("Plane", self.rd_lp_plane)
        self.rd_stack.addWidget(lp)  # 3

        # page: 3 planes
        p3, p3_form = _form_page()
        self.rd_p3_a = _entity_combo()
        self.rd_p3_b = _entity_combo()
        self.rd_p3_c = _entity_combo()
        p3_form.addRow("Plane A", self.rd_p3_a)
        p3_form.addRow("Plane B", self.rd_p3_b)
        p3_form.addRow("Plane C", self.rd_p3_c)
        self.rd_stack.addWidget(p3)  # 4

        # page: point + normal → axis
        npage, n_form = _form_page()
        self.rd_pn_point = _entity_combo()
        self.rd_pn_plane = _entity_combo()
        n_form.addRow("Point", self.rd_pn_point)
        n_form.addRow("Normal from", self.rd_pn_plane)
        self.rd_pn_hint = QLabel(
            "Axis through the point, direction = that plane's normal. "
            "The point does not have to lie on the plane."
        )
        self.rd_pn_hint.setObjectName("muted")
        self.rd_pn_hint.setWordWrap(True)
        npage.layout().insertWidget(1, self.rd_pn_hint)
        self.rd_stack.addWidget(npage)  # 5

        # page: 2 points → axis
        ppage, pp_form = _form_page()
        self.rd_pp_a = _entity_combo()
        self.rd_pp_b = _entity_combo()
        pp_form.addRow("Point A", self.rd_pp_a)
        pp_form.addRow("Point B", self.rd_pp_b)
        self.rd_pp_hint = QLabel(
            "Axis through both points. Direction is B − A "
            "(sign is fixed by the largest component)."
        )
        self.rd_pp_hint.setObjectName("muted")
        self.rd_pp_hint.setWordWrap(True)
        ppage.layout().insertWidget(1, self.rd_pp_hint)
        self.rd_stack.addWidget(ppage)  # 6

        # page: line ∩ 2 planes → midpoint
        mpage, mp_form = _form_page()
        self.rd_mp_line = _entity_combo()
        self.rd_mp_a = _entity_combo()
        self.rd_mp_b = _entity_combo()
        mp_form.addRow("Axis", self.rd_mp_line)
        mp_form.addRow("Plane A", self.rd_mp_a)
        mp_form.addRow("Plane B", self.rd_mp_b)
        self.rd_mp_hint = QLabel(
            "Hits of the axis on the two planes form a segment. "
            "The result is that segment's midpoint."
        )
        self.rd_mp_hint.setObjectName("muted")
        self.rd_mp_hint.setWordWrap(True)
        mpage.layout().insertWidget(1, self.rd_mp_hint)
        self.rd_stack.addWidget(mpage)  # 7

        # page: plane + point → parallel plane
        pp_page, pp_form = _form_page()
        self.rd_pp_plane = _entity_combo()
        self.rd_pp_point = _entity_combo()
        pp_form.addRow("Plane", self.rd_pp_plane)
        pp_form.addRow("Point", self.rd_pp_point)
        self.rd_pp_hint = QLabel(
            "Plane parallel to the source, passing through the point."
        )
        self.rd_pp_hint.setObjectName("muted")
        self.rd_pp_hint.setWordWrap(True)
        pp_page.layout().insertWidget(1, self.rd_pp_hint)
        self.rd_stack.addWidget(pp_page)  # 8

        # page: line + point → plane
        lpp_page, lpp_form = _form_page()
        self.rd_lpp_line = _entity_combo()
        self.rd_lpp_point = _entity_combo()
        lpp_form.addRow("Axis", self.rd_lpp_line)
        lpp_form.addRow("Point", self.rd_lpp_point)
        self.rd_lpp_hint = QLabel(
            "Plane through the point with normal = the axis direction."
        )
        self.rd_lpp_hint.setObjectName("muted")
        self.rd_lpp_hint.setWordWrap(True)
        lpp_page.layout().insertWidget(1, self.rd_lpp_hint)
        self.rd_stack.addWidget(lpp_page)  # 9

        # page: 2 lines → plane
        l2p_page, l2p_form = _form_page()
        self.rd_l2p_a = _entity_combo()
        self.rd_l2p_b = _entity_combo()
        l2p_form.addRow("Axis A", self.rd_l2p_a)
        l2p_form.addRow("Axis B", self.rd_l2p_b)
        self.rd_l2p_hint = QLabel(
            "Plane containing both axes. They must be coplanar (intersect or "
            "parallel in the same plane); skew lines are rejected."
        )
        self.rd_l2p_hint.setObjectName("muted")
        self.rd_l2p_hint.setWordWrap(True)
        l2p_page.layout().insertWidget(1, self.rd_l2p_hint)
        self.rd_stack.addWidget(l2p_page)  # 10

        # page: rotate plane about axis
        rot_page, rot_form = _form_page()
        self.rd_rot_plane = _entity_combo()
        self.rd_rot_line = _entity_combo()
        self.rd_rot_angle = QDoubleSpinBox()
        self.rd_rot_angle.setRange(-360.0, 360.0)
        self.rd_rot_angle.setDecimals(3)
        self.rd_rot_angle.setSingleStep(1.0)
        self.rd_rot_angle.setSuffix(" °")
        self.rd_rot_angle.setValue(0.0)
        self.rd_rot_angle.valueChanged.connect(self._reduction_on_operand_combo)
        rot_form.addRow("Plane", self.rd_rot_plane)
        rot_form.addRow("Axis", self.rd_rot_line)
        rot_form.addRow("Angle", self.rd_rot_angle)
        self.rd_rot_hint = QLabel(
            "Rotate the plane rigidly about the axis. The axis does not have "
            "to lie in the plane. When FRAME axis and origin are set, aligned "
            "X/Y/Z appear in the Axis list. Positive angle follows the "
            "right-hand rule. Rotation about a normal-direction axis leaves "
            "an infinite plane unchanged."
        )
        self.rd_rot_hint.setObjectName("muted")
        self.rd_rot_hint.setWordWrap(True)
        rot_page.layout().insertWidget(1, self.rd_rot_hint)
        self.rd_stack.addWidget(rot_page)  # 11

        # page: source-plane normal ∩ destination plane → point
        np_page, np_form = _form_page()
        self.rd_np_src = _entity_combo()
        self.rd_np_dst = _entity_combo()
        np_form.addRow("Normal from", self.rd_np_src)
        np_form.addRow("Hit plane", self.rd_np_dst)
        self.rd_np_hint = QLabel(
            "Ray along the source plane's normal from the source overlay "
            "(the patch you see) intersecting the destination plane. "
            "Nearly perpendicular planes send the hit far away."
        )
        self.rd_np_hint.setObjectName("muted")
        self.rd_np_hint.setWordWrap(True)
        np_page.layout().insertWidget(1, self.rd_np_hint)
        self.rd_stack.addWidget(np_page)  # 12

        op_lay.addWidget(self.rd_stack)
        self._reduction_lock_operation_stack_height()

        mode_row = QHBoxLayout()
        self.rd_mode_group = QButtonGroup(self)
        self.rd_mode_update = QRadioButton("Update selected")
        self.rd_mode_new = QRadioButton("Create new")
        self.rd_mode_group.addButton(self.rd_mode_update, 0)
        self.rd_mode_group.addButton(self.rd_mode_new, 1)
        self.rd_mode_new.setChecked(True)
        self.rd_mode_group.idToggled.connect(self._reduction_on_mode_toggled)
        mode_row.addWidget(self.rd_mode_update)
        mode_row.addWidget(self.rd_mode_new)
        op_lay.addLayout(mode_row)

        self.rd_apply_btn = QPushButton("Apply")
        self.rd_apply_btn.setObjectName("primaryBtn")
        self.rd_apply_btn.clicked.connect(
            lambda: self._guard(self._reduction_apply, busy=False)
        )
        op_lay.addWidget(self.rd_apply_btn)
        lay.addWidget(op_card)

        # ---- Display (always available, secondary) ----------------------
        disp_card, disp_lay = _mini_card("DISPLAY", expanded=False)
        disp_form = QFormLayout()
        disp_form.setContentsMargins(0, 0, 0, 0)
        disp_form.setSpacing(4)
        self._rd_size_sync = False

        def _mm_spin(lo: float, hi: float, value: float, *, decimals: int = 1) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setSingleStep(10.0 if decimals == 1 and hi >= 100 else 0.5)
            s.setSuffix(" mm")
            s.setValue(value)
            s.setMaximumWidth(110)
            return s

        def _mm_slider(lo: int, hi: int, value: int) -> QSlider:
            sl = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(value)
            sl.setTickPosition(QSlider.TicksBelow)
            sl.setTickInterval(max(1, (hi - lo) // 4))
            return sl

        def _bind_size(
            spin: QDoubleSpinBox,
            slider: QSlider,
            *,
            kind: str,
            ticks_per_mm: float = 1.0,
        ):
            def on_spin(val: float):
                if not self._rd_size_sync:
                    self._rd_size_sync = True
                    try:
                        ticks = int(round(float(val) * ticks_per_mm))
                        slider.setValue(max(slider.minimum(), min(slider.maximum(), ticks)))
                    finally:
                        self._rd_size_sync = False
                if self._rd_size_loading:
                    return
                eids = self._reduction_apply_overlay_size(kind, float(val))
                self._refresh_reduction_entity_overlays(eids, render=False)
                self._reduction_update_live_preview()

            def on_slider(ticks: int):
                if self._rd_size_sync:
                    return
                val = float(ticks) / ticks_per_mm
                self._rd_size_sync = True
                try:
                    spin.blockSignals(True)
                    spin.setValue(val)
                    spin.blockSignals(False)
                finally:
                    self._rd_size_sync = False
                if self._rd_size_loading:
                    return
                eids = self._reduction_apply_overlay_size(kind, val)
                self._refresh_reduction_entity_overlays(eids, render=False)
                self._reduction_update_live_preview()

            spin.valueChanged.connect(on_spin)
            slider.valueChanged.connect(on_slider)

        def _size_row(spin: QDoubleSpinBox, slider: QSlider) -> QWidget:
            w = QWidget()
            row = QHBoxLayout(w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(spin)
            row.addWidget(slider, stretch=1)
            return w

        self._rd_size_loading = False
        self.rd_size_lbl = {
            "plane": QLabel("Plane"),
            "line": QLabel("Axis ±"),
            "line_diameter": QLabel("Axis ⌀"),
            "point": QLabel("Point"),
        }
        self.rd_patch_spin = _mm_spin(10.0, 1.0e5, 200.0)
        self.rd_patch_slider = _mm_slider(20, 2000, 200)
        _bind_size(self.rd_patch_spin, self.rd_patch_slider, kind="plane")
        disp_form.addRow(self.rd_size_lbl["plane"], _size_row(self.rd_patch_spin, self.rd_patch_slider))

        self.rd_axis_spin = _mm_spin(10.0, 1.0e5, 300.0)
        self.rd_axis_slider = _mm_slider(20, 2000, 300)
        _bind_size(self.rd_axis_spin, self.rd_axis_slider, kind="line")
        disp_form.addRow(self.rd_size_lbl["line"], _size_row(self.rd_axis_spin, self.rd_axis_slider))

        self.rd_axis_diam_spin = _mm_spin(0.2, 50.0, 1.0, decimals=2)
        self.rd_axis_diam_spin.setSingleStep(0.1)
        self.rd_axis_diam_slider = _mm_slider(2, 200, 10)  # 0.1 mm ticks → 1.0 mm
        _bind_size(
            self.rd_axis_diam_spin,
            self.rd_axis_diam_slider,
            kind="line_diameter",
            ticks_per_mm=10.0,
        )
        disp_form.addRow(
            self.rd_size_lbl["line_diameter"],
            _size_row(self.rd_axis_diam_spin, self.rd_axis_diam_slider),
        )

        self.rd_point_spin = _mm_spin(0.5, 1.0e4, 8.0, decimals=1)
        self.rd_point_spin.setSingleStep(0.5)
        self.rd_point_slider = _mm_slider(5, 2000, 80)  # 0.1 mm ticks → 8.0 mm, max 200 mm
        _bind_size(self.rd_point_spin, self.rd_point_slider, kind="point", ticks_per_mm=10.0)
        disp_form.addRow(self.rd_size_lbl["point"], _size_row(self.rd_point_spin, self.rd_point_slider))

        disp_lay.addLayout(disp_form)
        size_btn_row = QHBoxLayout()
        self.rd_size_hint = QLabel("No entity selected — sliders set the default size.")
        self.rd_size_hint.setObjectName("muted")
        self.rd_size_hint.setWordWrap(True)
        size_btn_row.addWidget(self.rd_size_hint, stretch=1)
        self.rd_reset_size_btn = QPushButton("Reset selected")
        self.rd_reset_size_btn.setObjectName("secondaryBtn")
        self.rd_reset_size_btn.setEnabled(False)
        self.rd_reset_size_btn.setToolTip(
            "Clear per-entity size so the selected items use the default again."
        )
        self.rd_reset_size_btn.clicked.connect(self._reduction_reset_selected_overlay)
        size_btn_row.addWidget(self.rd_reset_size_btn)
        disp_lay.addLayout(size_btn_row)
        rd_legend = QLabel(
            "Select an entity to size it alone. Empty selection edits defaults. "
            "Axis ± is half-length, Axis ⌀ is tube diameter. "
            "Blue = scanned plane, orange = offset, red = line, yellow = point."
        )
        rd_legend.setObjectName("muted")
        rd_legend.setWordWrap(True)
        disp_lay.addWidget(rd_legend)
        lay.addWidget(disp_card)

        # ---- Entities ---------------------------------------------------
        tree_card, tree_lay = _mini_card("ENTITIES")
        self.rd_selection_label = QLabel("Operands: (none)")
        self.rd_selection_label.setObjectName("muted")
        self.rd_selection_label.setWordWrap(True)
        tree_lay.addWidget(self.rd_selection_label)
        self.rd_tree = QTreeWidget()
        self.rd_tree.setHeaderLabels(["id", "type", "size", "detail"])
        self.rd_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.rd_tree.setRootIsDecorated(False)
        self.rd_tree.setAlternatingRowColors(True)
        self.rd_tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.rd_tree.itemChanged.connect(self._on_reduction_item_changed)
        self.rd_tree.itemSelectionChanged.connect(self._sync_reduction_entity_actions)
        _esc = QShortcut(QKeySequence(Qt.Key_Escape), self.rd_tree)
        _esc.setContext(Qt.WidgetShortcut)
        _esc.activated.connect(self.rd_tree.clearSelection)
        tree_lay.addWidget(self.rd_tree)
        ent_btn_row = QHBoxLayout()
        self.rd_delete_btn = QPushButton("Delete")
        self.rd_delete_btn.setObjectName("dangerBtn")
        self.rd_delete_btn.setEnabled(False)
        self.rd_delete_btn.setToolTip(
            "Delete the selected entities and anything built from them. "
            "Double-click an id to rename. Backspace/Delete when the list is focused."
        )
        self.rd_delete_btn.clicked.connect(
            lambda: self._guard(self._reduction_delete_selected, busy=False)
        )
        ent_btn_row.addWidget(self.rd_delete_btn)
        clear_btn = QPushButton("Clear session")
        clear_btn.setObjectName("dangerBtn")
        clear_btn.clicked.connect(
            lambda: self._guard(self._reduction_clear, busy=False)
        )
        ent_btn_row.addWidget(clear_btn)
        tree_lay.addLayout(ent_btn_row)
        lay.addWidget(tree_card, stretch=1)

        frame_card, frame_lay = _mini_card("FRAME")
        frame_form = QFormLayout()
        frame_form.setContentsMargins(0, 0, 0, 0)
        frame_form.setSpacing(4)
        self.rd_frame_axis = QComboBox()
        self.rd_frame_origin = QComboBox()
        for c in (self.rd_frame_axis, self.rd_frame_origin):
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(12)
            c.currentIndexChanged.connect(self._on_frame_combo_changed)
        self.rd_frame_flip = QComboBox()
        self.rd_frame_flip.addItem("along axis", False)
        self.rd_frame_flip.addItem("flipped", True)
        self.rd_frame_flip.currentIndexChanged.connect(self._on_frame_combo_changed)
        self.rd_frame_yaw_to = QComboBox()
        self.rd_frame_yaw_to.addItem("(none)", None)
        self.rd_frame_yaw_to.addItem("→ +X", "x")
        self.rd_frame_yaw_to.addItem("→ −X", "-x")
        self.rd_frame_yaw_to.addItem("→ +Y", "y")
        self.rd_frame_yaw_to.addItem("→ −Y", "-y")
        self.rd_frame_yaw_to.currentIndexChanged.connect(self._on_frame_combo_changed)
        self.rd_frame_yaw_kind = QComboBox()
        self.rd_frame_yaw_kind.addItem("Line", "line")
        self.rd_frame_yaw_kind.addItem("Plane normal", "plane")
        self.rd_frame_yaw_kind.currentIndexChanged.connect(self._on_frame_yaw_kind_changed)
        self.rd_frame_yaw_ref = QComboBox()
        self.rd_frame_yaw_ref.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.rd_frame_yaw_ref.setMinimumContentsLength(12)
        self.rd_frame_yaw_ref.currentIndexChanged.connect(self._on_frame_combo_changed)
        frame_form.addRow("Axis", self.rd_frame_axis)
        frame_form.addRow("Origin", self.rd_frame_origin)
        frame_form.addRow("+Z", self.rd_frame_flip)
        frame_form.addRow("XY", self.rd_frame_yaw_to)
        frame_form.addRow("XY from", self.rd_frame_yaw_kind)
        frame_form.addRow("", self.rd_frame_yaw_ref)
        frame_lay.addLayout(frame_form)
        frame_btn_row = QHBoxLayout()
        self.rd_frame_align_btn = QPushButton("Align Z")
        self.rd_frame_align_btn.setObjectName("primaryBtn")
        self.rd_frame_align_btn.setEnabled(False)
        self.rd_frame_align_btn.setToolTip(
            "Show the cloud and overlays with this axis as global +Z "
            "and this origin at (0, 0, 0). Optionally map a line direction or "
            "plane normal (XY projection) onto ±X or ±Y. Survey data is not rewritten."
        )
        self.rd_frame_align_btn.clicked.connect(
            lambda: self._guard(self._frame_align, busy=False)
        )
        frame_btn_row.addWidget(self.rd_frame_align_btn)
        self.rd_frame_survey_btn = QPushButton("Survey")
        self.rd_frame_survey_btn.setObjectName("secondaryBtn")
        self.rd_frame_survey_btn.setEnabled(False)
        self.rd_frame_survey_btn.setToolTip("Return the 3D view to survey coordinates.")
        self.rd_frame_survey_btn.clicked.connect(
            lambda: self._guard(self._frame_survey, busy=False)
        )
        frame_btn_row.addWidget(self.rd_frame_survey_btn)
        frame_lay.addLayout(frame_btn_row)
        self.rd_frame_status = QLabel("frame: survey")
        self.rd_frame_status.setObjectName("muted")
        self.rd_frame_status.setWordWrap(True)
        frame_lay.addWidget(self.rd_frame_status)
        frame_hint = QLabel(
            "Display only. Groups, recipe, and Fit stay in survey coordinates. "
            "Reduction overlays and the Entities table follow this view. "
            "Picking still extracts from the original cloud."
        )
        frame_hint.setObjectName("muted")
        frame_hint.setWordWrap(True)
        frame_lay.addWidget(frame_hint)
        lay.addWidget(frame_card)

        exp_card, exp_lay = _mini_card("EXPORT")
        self.rd_export_frame_cb = QCheckBox("Also write aligned-frame coordinates")
        self.rd_export_frame_cb.setChecked(True)
        self.rd_export_frame_cb.setToolTip(
            "When FRAME axis and origin are set, geometry.json keeps survey numbers "
            "and adds an aligned copy under \"aligned\" plus the frame pose. "
            "Align Z is not required for export (same rule as cloudet reduce). "
            "Off: survey coordinates only. The recipe is never transformed."
        )
        exp_lay.addWidget(self.rd_export_frame_cb)
        exp_row = QHBoxLayout()
        load_recipe_btn = QPushButton("Load recipe…")
        load_recipe_btn.setObjectName("secondaryBtn")
        load_recipe_btn.setToolTip(
            "Replace this session with a saved recipe.json. "
            "FRAME axis/origin are restored if saved; Align Z is not applied. "
            "Load All also restores project_dir/recipe.json when present."
        )
        load_recipe_btn.clicked.connect(
            lambda: self._guard(self._reduction_load_recipe, busy=False)
        )
        exp_row.addWidget(load_recipe_btn)
        save_recipe_btn = QPushButton("Save recipe…")
        save_recipe_btn.setObjectName("secondaryBtn")
        save_recipe_btn.clicked.connect(
            lambda: self._guard(self._reduction_save_recipe, busy=False)
        )
        exp_row.addWidget(save_recipe_btn)
        save_geom_btn = QPushButton("Export geometry…")
        save_geom_btn.setObjectName("primaryBtn")
        save_geom_btn.clicked.connect(
            lambda: self._guard(self._reduction_export_geometry, busy=False)
        )
        exp_row.addWidget(save_geom_btn)
        exp_lay.addLayout(exp_row)
        lay.addWidget(exp_card)

        scroll = QScrollArea()
        scroll.setWidget(w)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        dock.setWidget(scroll)
        dock.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.reduction_dock = dock
        if hasattr(self, "uv_dock"):
            self.tabifyDockWidget(self.uv_dock, dock)
            self.uv_dock.raise_()

        self._rd_offset_sync = False
        self._reduction_fill_frame_yaw_ref()
        self._reduction_on_op_changed()
        self._update_frame_controls()

    def _build_measure_dock(self):
        dock = QDockWidget("Measure", self)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        dock.setStyleSheet(UI_STYLE)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)

        title = QLabel("MEASURE")
        title.setObjectName("sectionTitle")
        lay.addWidget(title)
        hint = QLabel(
            "Read distances and angles from Reduction entities. "
            "Add measurement stores the pair in the recipe; values are recomputed."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self.rd_measure_op = QComboBox()
        for label, key in MEASURE_MENU_ITEMS:
            self.rd_measure_op.addItem(label, key)
        self.rd_measure_op.currentIndexChanged.connect(self._reduction_on_measure_op)
        lay.addWidget(self.rd_measure_op)

        meas_form = QFormLayout()
        meas_form.setContentsMargins(0, 0, 0, 0)
        meas_form.setSpacing(4)
        self.rd_measure_a = QComboBox()
        self.rd_measure_b = QComboBox()
        for c in (self.rd_measure_a, self.rd_measure_b):
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(12)
            c.currentIndexChanged.connect(self._reduction_update_live_measure)
        self.rd_measure_a_lbl = QLabel("Point A")
        self.rd_measure_b_lbl = QLabel("Point B")
        meas_form.addRow(self.rd_measure_a_lbl, self.rd_measure_a)
        meas_form.addRow(self.rd_measure_b_lbl, self.rd_measure_b)
        lay.addLayout(meas_form)

        self.rd_measure_live = QLabel("—")
        self.rd_measure_live.setObjectName("muted")
        self.rd_measure_live.setWordWrap(True)
        lay.addWidget(self.rd_measure_live)

        self.rd_measure_add_btn = QPushButton("Add measurement")
        self.rd_measure_add_btn.setObjectName("primaryBtn")
        self.rd_measure_add_btn.setEnabled(False)
        self.rd_measure_add_btn.clicked.connect(
            lambda: self._guard(self._reduction_add_measure, busy=False)
        )
        lay.addWidget(self.rd_measure_add_btn)

        self.rd_measure_tree = QTreeWidget()
        self.rd_measure_tree.setHeaderLabels(["id", "type", "value", "of"])
        self.rd_measure_tree.setRootIsDecorated(False)
        self.rd_measure_tree.setAlternatingRowColors(True)
        self.rd_measure_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.rd_measure_tree.itemSelectionChanged.connect(
            self._sync_measure_delete_enabled
        )
        lay.addWidget(self.rd_measure_tree, stretch=1)

        self.rd_measure_del_btn = QPushButton("Delete")
        self.rd_measure_del_btn.setObjectName("dangerBtn")
        self.rd_measure_del_btn.setEnabled(False)
        self.rd_measure_del_btn.clicked.connect(
            lambda: self._guard(self._reduction_delete_measures, busy=False)
        )
        lay.addWidget(self.rd_measure_del_btn)

        meas_hint = QLabel(
            "Distances in mm (always ≥ 0). Angles in degrees (0–90): "
            "line–plane is 0° if parallel to the plane, 90° if perpendicular."
        )
        meas_hint.setObjectName("muted")
        meas_hint.setWordWrap(True)
        lay.addWidget(meas_hint)

        dock.setWidget(w)
        dock.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.measure_dock = dock
        if hasattr(self, "uv_dock"):
            self.tabifyDockWidget(self.uv_dock, dock)
        elif hasattr(self, "reduction_dock"):
            self.tabifyDockWidget(self.reduction_dock, dock)

        self._reduction_on_measure_op()
    def _reduction_new_id(self, prefix: str) -> str:
        typed = self.rd_id_edit.text().strip() if hasattr(self, "rd_id_edit") else ""
        if typed:
            if typed in self._reduction.ids():
                raise ValueError(f"id {typed!r} already exists")
            return typed
        return self._reduction.unique_id(prefix)

    def _reduction_combo_id(self, combo: QComboBox | None) -> str | None:
        if combo is None or combo.currentIndex() < 0:
            return None
        data = combo.currentData()
        if data is None:
            return None
        return str(data)

    def _reduction_fill_combo(
        self,
        combo: QComboBox,
        *,
        kind: str | None = None,
        keep: str | None = None,
        placeholder: str = "(choose)",
        allowed: set[str] | None = None,
        include_aligned_axes: bool = True,
    ) -> None:
        ids = self._reduction.ids(kind=kind) if kind else self._reduction.ids()
        if allowed is not None:
            ids = [eid for eid in ids if eid in allowed]
        combo.blockSignals(True)
        _reset_combo(combo)
        combo.addItem(placeholder, None)
        if include_aligned_axes and kind == "line":
            extras = self._reduction.available_aligned_axis_ids(
                before=None if allowed is None else allowed
            )
            for eid in extras:
                label = ALIGNED_AXIS_LABELS.get(eid, eid)
                combo.addItem(label, eid)
        for eid in ids:
            tag = RD_KIND_LABEL.get(self._reduction.kind_of(eid), "")
            combo.addItem(f"{eid}  ({tag})" if tag else eid, eid)
        if keep:
            idx = combo.findData(keep)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _reduction_fill_bind_combo(self) -> None:
        if not hasattr(self, "rd_bind_combo"):
            return
        keep = self._reduction_combo_id(self.rd_bind_combo)
        combo = self.rd_bind_combo
        combo.blockSignals(True)
        _reset_combo(combo)
        combo.addItem("(choose a fitted Groups plane)", None)
        for g in self.groups:
            planes = (g.get("fit") or {}).get("planes") or []
            for p in planes:
                pi = int(p.get("plane_index", 0))
                key = f"{g['id']}:{pi}"
                label = f"{g['name']}/{_plane_label(p)}"
                combo.addItem(label, key)
        if keep:
            idx = combo.findData(keep)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _reduction_refresh_operand_combos(
        self,
        *,
        rename: tuple[str, str] | None = None,
        include_frame: bool = True,
    ) -> None:
        self._reduction_fill_bind_combo()
        keep = {
            "offset": self._reduction_combo_id(getattr(self, "rd_offset_plane", None)),
            "p2a": self._reduction_combo_id(getattr(self, "rd_p2_a", None)),
            "p2b": self._reduction_combo_id(getattr(self, "rd_p2_b", None)),
            "lp_line": self._reduction_combo_id(getattr(self, "rd_lp_line", None)),
            "lp_plane": self._reduction_combo_id(getattr(self, "rd_lp_plane", None)),
            "p3a": self._reduction_combo_id(getattr(self, "rd_p3_a", None)),
            "p3b": self._reduction_combo_id(getattr(self, "rd_p3_b", None)),
            "p3c": self._reduction_combo_id(getattr(self, "rd_p3_c", None)),
            "pn_point": self._reduction_combo_id(getattr(self, "rd_pn_point", None)),
            "pn_plane": self._reduction_combo_id(getattr(self, "rd_pn_plane", None)),
            "pp_a": self._reduction_combo_id(getattr(self, "rd_pp_a", None)),
            "pp_b": self._reduction_combo_id(getattr(self, "rd_pp_b", None)),
            "mp_line": self._reduction_combo_id(getattr(self, "rd_mp_line", None)),
            "mp_a": self._reduction_combo_id(getattr(self, "rd_mp_a", None)),
            "mp_b": self._reduction_combo_id(getattr(self, "rd_mp_b", None)),
            "pp_plane": self._reduction_combo_id(getattr(self, "rd_pp_plane", None)),
            "pp_point": self._reduction_combo_id(getattr(self, "rd_pp_point", None)),
            "lpp_line": self._reduction_combo_id(getattr(self, "rd_lpp_line", None)),
            "lpp_point": self._reduction_combo_id(getattr(self, "rd_lpp_point", None)),
            "l2p_a": self._reduction_combo_id(getattr(self, "rd_l2p_a", None)),
            "l2p_b": self._reduction_combo_id(getattr(self, "rd_l2p_b", None)),
            "rot_plane": self._reduction_combo_id(getattr(self, "rd_rot_plane", None)),
            "rot_line": self._reduction_combo_id(getattr(self, "rd_rot_line", None)),
            "np_src": self._reduction_combo_id(getattr(self, "rd_np_src", None)),
            "np_dst": self._reduction_combo_id(getattr(self, "rd_np_dst", None)),
            "frame_axis": self._reduction_combo_id(getattr(self, "rd_frame_axis", None)),
            "frame_origin": self._reduction_combo_id(
                getattr(self, "rd_frame_origin", None)
            ),
            "frame_yaw": self._reduction_combo_id(
                getattr(self, "rd_frame_yaw_ref", None)
            ),
            "measure_a": self._reduction_combo_id(getattr(self, "rd_measure_a", None)),
            "measure_b": self._reduction_combo_id(getattr(self, "rd_measure_b", None)),
        }
        if rename:
            old, new = rename
            keep = {k: (new if v == old else v) for k, v in keep.items()}
        allowed = self._reduction_operand_allowlist()
        fill_kw = {"allowed": allowed} if allowed is not None else {}
        if hasattr(self, "rd_offset_plane"):
            self._reduction_fill_combo(
                self.rd_offset_plane, kind="plane", keep=keep["offset"], **fill_kw
            )
            self._reduction_fill_combo(
                self.rd_p2_a, kind="plane", keep=keep["p2a"], **fill_kw
            )
            self._reduction_fill_combo(
                self.rd_p2_b, kind="plane", keep=keep["p2b"], **fill_kw
            )
            self._reduction_fill_combo(
                self.rd_lp_line, kind="line", keep=keep["lp_line"], **fill_kw
            )
            self._reduction_fill_combo(
                self.rd_lp_plane, kind="plane", keep=keep["lp_plane"], **fill_kw
            )
            self._reduction_fill_combo(
                self.rd_p3_a, kind="plane", keep=keep["p3a"], **fill_kw
            )
            self._reduction_fill_combo(
                self.rd_p3_b, kind="plane", keep=keep["p3b"], **fill_kw
            )
            self._reduction_fill_combo(
                self.rd_p3_c, kind="plane", keep=keep["p3c"], **fill_kw
            )
            self._reduction_fill_combo(
                self.rd_pn_point, kind="point", keep=keep["pn_point"], **fill_kw
            )
            self._reduction_fill_combo(
                self.rd_pn_plane, kind="plane", keep=keep["pn_plane"], **fill_kw
            )
            if hasattr(self, "rd_pp_a"):
                self._reduction_fill_combo(
                    self.rd_pp_a, kind="point", keep=keep["pp_a"], **fill_kw
                )
                self._reduction_fill_combo(
                    self.rd_pp_b, kind="point", keep=keep["pp_b"], **fill_kw
                )
            if hasattr(self, "rd_mp_line"):
                self._reduction_fill_combo(
                    self.rd_mp_line, kind="line", keep=keep["mp_line"], **fill_kw
                )
                self._reduction_fill_combo(
                    self.rd_mp_a, kind="plane", keep=keep["mp_a"], **fill_kw
                )
                self._reduction_fill_combo(
                    self.rd_mp_b, kind="plane", keep=keep["mp_b"], **fill_kw
                )
            if hasattr(self, "rd_pp_plane"):
                self._reduction_fill_combo(
                    self.rd_pp_plane, kind="plane", keep=keep["pp_plane"], **fill_kw
                )
            if hasattr(self, "rd_pp_point"):
                self._reduction_fill_combo(
                    self.rd_pp_point, kind="point", keep=keep["pp_point"], **fill_kw
                )
            if hasattr(self, "rd_lpp_line"):
                self._reduction_fill_combo(
                    self.rd_lpp_line, kind="line", keep=keep["lpp_line"], **fill_kw
                )
                self._reduction_fill_combo(
                    self.rd_lpp_point, kind="point", keep=keep["lpp_point"], **fill_kw
                )
            if hasattr(self, "rd_l2p_a"):
                self._reduction_fill_combo(
                    self.rd_l2p_a, kind="line", keep=keep["l2p_a"], **fill_kw
                )
                self._reduction_fill_combo(
                    self.rd_l2p_b, kind="line", keep=keep["l2p_b"], **fill_kw
                )
            if hasattr(self, "rd_rot_plane"):
                self._reduction_fill_combo(
                    self.rd_rot_plane, kind="plane", keep=keep["rot_plane"], **fill_kw
                )
                self._reduction_fill_combo(
                    self.rd_rot_line, kind="line", keep=keep["rot_line"], **fill_kw
                )
            if hasattr(self, "rd_np_src"):
                self._reduction_fill_combo(
                    self.rd_np_src, kind="plane", keep=keep["np_src"], **fill_kw
                )
                self._reduction_fill_combo(
                    self.rd_np_dst, kind="plane", keep=keep["np_dst"], **fill_kw
                )
            if include_frame and hasattr(self, "rd_frame_axis"):
                self._reduction_fill_combo(
                    self.rd_frame_axis,
                    kind="line",
                    keep=keep.get("frame_axis"),
                    include_aligned_axes=False,
                )
                self._reduction_fill_combo(
                    self.rd_frame_origin, kind="point", keep=keep.get("frame_origin")
                )
                if hasattr(self, "rd_frame_yaw_ref"):
                    spec = self._reduction.frame_spec or {}
                    yaw_kind = "plane" if spec.get("yaw_plane") else "line"
                    idx = self.rd_frame_yaw_kind.findData(yaw_kind)
                    if idx >= 0:
                        self.rd_frame_yaw_kind.blockSignals(True)
                        self.rd_frame_yaw_kind.setCurrentIndex(idx)
                        self.rd_frame_yaw_kind.blockSignals(False)
                    self._reduction_fill_frame_yaw_ref(keep=keep.get("frame_yaw"))
                if rename and self._view_frame is not None:
                    old, new = rename
                    self._view_frame = self._view_frame.relabel(old, new)
                    self._update_frame_controls()
                    self._rebuild_status_default()
                self._sync_frame_align_enabled()
            if hasattr(self, "rd_measure_a"):
                kinds, _labels = self._measure_operand_meta()
                self._reduction_fill_combo(
                    self.rd_measure_a, kind=kinds[0], keep=keep.get("measure_a")
                )
                self._reduction_fill_combo(
                    self.rd_measure_b, kind=kinds[1], keep=keep.get("measure_b")
                )
                self._reduction_update_live_measure()

    def _reduction_selected_ids(self) -> list[str]:
        """Operand ids for the current operation (from OPERATION combos)."""
        op = self._reduction_current_op()
        op_def = REDUCTION_OP_BY_GUI.get(op)
        if op_def is None:
            return []
        ids: list[str] = []
        for field in op_def.operands:
            eid = self._reduction_combo_id(getattr(self, field.widget, None))
            if eid:
                ids.append(eid)
        return ids

    def _reduction_current_op(self) -> str:
        if not hasattr(self, "rd_op_combo"):
            return "bind"
        return str(self.rd_op_combo.currentData())

    def _reduction_editing_id(self) -> str | None:
        """Construct entity id when ENTITIES has exactly one construct selected."""
        ids = self._reduction_tree_selected_ids()
        if len(ids) != 1:
            return None
        eid = ids[0]
        step = self._reduction.construct_step(eid)
        if step is None:
            return None
        if step.get("op") not in RD_RECIPE_TO_GUI_OP:
            return None
        return eid

    def _reduction_operand_allowlist(self) -> set[str] | None:
        if not self._reduction_is_update_mode():
            return None
        eid = self._reduction_editing_id()
        if eid is None:
            return None
        allowed = self._reduction.operand_ids_before(eid)
        allowed.update(self._reduction.available_aligned_axis_ids(before=allowed))
        return allowed

    def _reduction_set_combo(self, combo: QComboBox | None, eid: str | None) -> None:
        if combo is None:
            return
        combo.blockSignals(True)
        idx = combo.findData(eid) if eid else -1
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _reduction_sync_apply_button(self) -> None:
        update_mode = self._reduction_is_update_mode()
        editing = update_mode and self._reduction_editing_id() is not None
        op = self._reduction_current_op()
        show_id = True
        if hasattr(self, "rd_id_edit"):
            self.rd_id_edit.setVisible(show_id)
            self.rd_id_edit.setEnabled((not editing) or op == "bind")
            label = None
            if hasattr(self, "rd_id_form"):
                label = self.rd_id_form.labelForField(self.rd_id_edit)
            if label is not None:
                label.setVisible(show_id)
        if hasattr(self, "rd_mode_update"):
            self.rd_mode_update.setEnabled(
                self._reduction_editing_id() is not None and op != "bind"
            )
        if hasattr(self, "rd_op_combo"):
            self.rd_op_combo.setEnabled(not editing)
        if not hasattr(self, "rd_apply_btn"):
            return
        if editing and op != "bind":
            self.rd_apply_btn.setText("Update")
            return
        self.rd_apply_btn.setText(GUI_APPLY_LABELS.get(op, "Apply"))

    def _reduction_sync_operation_from_selection(self) -> None:
        eid = self._reduction_editing_id()
        if hasattr(self, "rd_mode_update"):
            if eid is not None:
                self.rd_mode_update.setChecked(True)
            else:
                self.rd_mode_new.setChecked(True)
        if eid == getattr(self, "_rd_form_entity_id", None):
            self._reduction_sync_apply_button()
            return
        self._rd_form_entity_id = eid
        if eid is None:
            self._reduction_sync_apply_button()
            return
        step = self._reduction.construct_step(eid)
        if step is None:
            self._reduction_sync_apply_button()
            return
        self._reduction_load_step_into_form(step)
        self._reduction_sync_apply_button()

    def _reduction_read_operand_values(self, op_def) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for field in op_def.operands:
            combo = getattr(self, field.widget, None)
            out[field.widget] = self._reduction_combo_id(combo)
        return out

    def _reduction_read_scalar_values(self, op_def) -> dict[str, float]:
        out: dict[str, float] = {}
        for field in op_def.scalars:
            widget = getattr(self, field.widget, None)
            if widget is not None:
                out[field.widget] = float(widget.value())
        return out

    def _reduction_apply_form_values(
        self,
        gui_key: str,
        operand_values: dict[str, str | None],
        scalar_values: dict[str, float],
    ) -> None:
        op_def = REDUCTION_OP_BY_GUI[gui_key]
        for field in op_def.operands:
            self._reduction_set_combo(
                getattr(self, field.widget, None),
                operand_values.get(field.widget),
            )
        for field in op_def.scalars:
            widget = getattr(self, field.widget, None)
            if widget is not None:
                widget.setValue(float(scalar_values.get(field.widget, 0.0)))

    def _reduction_load_step_into_form(self, step: dict) -> None:
        gui_key, operands, scalars = form_values_from_step(step)
        if gui_key is None or not hasattr(self, "rd_op_combo"):
            return
        self._rd_loading_step = True
        try:
            idx = self.rd_op_combo.findData(gui_key)
            if idx >= 0:
                self.rd_op_combo.setCurrentIndex(idx)
            self._reduction_refresh_operand_combos()
            self._reduction_apply_form_values(gui_key, operands, scalars)
        finally:
            self._rd_loading_step = False
        self._reduction_update_selection_label()
        self._refresh_reduction_actors()
        self._reduction_update_live_preview()

    def _reduction_step_from_form(self, entity_id: str) -> dict:
        op = self._reduction_current_op()
        op_def = REDUCTION_OP_BY_GUI.get(op)
        if op_def is None:
            raise ValueError(f"cannot update with operation {op!r}")
        return build_construct_step(
            op_def,
            entity_id,
            operand_values=self._reduction_read_operand_values(op_def),
            scalar_values=self._reduction_read_scalar_values(op_def),
        )

    def _reduction_on_op_changed(self, *_args):
        op = self._reduction_current_op()
        page = GUI_PAGE_INDEX.get(op, 0)
        if hasattr(self, "rd_stack"):
            self.rd_stack.setCurrentIndex(page)
        self._reduction_sync_apply_button()
        self._reduction_refresh_operand_combos()
        self._reduction_update_selection_label()
        if self._rd_loading_step:
            return
        self._refresh_reduction_actors()
        self._reduction_update_live_preview()

    def _reduction_lock_operation_stack_height(self) -> None:
        """Keep OPERATION card height stable across operation switches."""
        if not hasattr(self, "rd_stack"):
            return
        stack = self.rd_stack
        count = stack.count()
        if count <= 0:
            return
        old_idx = stack.currentIndex()
        max_h = 0
        for i in range(count):
            stack.setCurrentIndex(i)
            page = stack.widget(i)
            if page is None:
                continue
            page_h = page.sizeHint().height()
            if page_h > max_h:
                max_h = page_h
        stack.setCurrentIndex(old_idx)
        if max_h > 0:
            stack.setFixedHeight(max_h)

    def _reduction_is_update_mode(self) -> bool:
        return hasattr(self, "rd_mode_update") and self.rd_mode_update.isChecked()

    def _reduction_on_mode_toggled(self, _id, _checked):
        self._reduction_sync_apply_button()
        self._reduction_refresh_operand_combos()
        self._reduction_update_selection_label()
        self._refresh_reduction_actors()
        self._reduction_update_live_preview()

    def _reduction_on_operand_combo(self, *_args):
        if self._rd_loading_step:
            return
        self._reduction_update_selection_label()
        self._reduction_sync_size_controls_from_selection()
        self._refresh_reduction_actors()
        self._reduction_update_live_preview()

    def _reduction_on_selection_changed(self):
        self._reduction_on_operand_combo()

    def _reduction_update_selection_label(self):
        if not hasattr(self, "rd_selection_label"):
            return
        ids = self._reduction_selected_ids()
        if not ids:
            self.rd_selection_label.setText("Operands: (none)")
            return
        parts = []
        for eid in ids:
            kind = self._reduction.kind_of(eid)
            name = ALIGNED_AXIS_LABELS.get(eid, eid)
            tag = RD_KIND_LABEL.get(kind, kind)
            parts.append(f"{name} [{tag}]")
        self.rd_selection_label.setText("Operands: " + ", ".join(parts))

    def _reduction_entity_color(self, eid: str, *, selected: bool) -> str:
        kind = self._reduction.kind_of(eid)
        if kind == "plane":
            rec = self._reduction.record_of(eid)
            base = RD_PLANE_SCANNED if rec.get("provenance") == "scanned" else RD_PLANE_OFFSET
        elif kind == "line":
            base = RD_AXIS
        else:
            base = RD_POINT
        if selected:
            return "#ffffff" if kind == "line" else base
        return base

    def _reduction_on_offset_spin(self, value: float):
        if self._rd_offset_sync:
            return
        self._rd_offset_sync = True
        try:
            # Keep slider in ±500 mm window when possible.
            ticks = int(round(float(value) * 10.0))
            if hasattr(self, "rd_offset_slider"):
                lo = self.rd_offset_slider.minimum()
                hi = self.rd_offset_slider.maximum()
                self.rd_offset_slider.setValue(max(lo, min(hi, ticks)))
        finally:
            self._rd_offset_sync = False
        if not self._rd_loading_step:
            self._reduction_update_live_preview()

    def _reduction_on_offset_slider(self, ticks: int):
        if self._rd_offset_sync:
            return
        self._rd_offset_sync = True
        try:
            mm = float(ticks) / 10.0
            if hasattr(self, "rd_offset_spin"):
                self.rd_offset_spin.setValue(mm)
        finally:
            self._rd_offset_sync = False
        self._reduction_update_live_preview()

    def _clear_reduction_preview(self):
        for name in (
            "rd_preview_offset",
            "rd_preview_offset_e",
            "rd_preview_axis",
            "rd_preview_point",
            "rd_preview_end0",
            "rd_preview_end1",
        ):
            self.plotter.remove_actor(name, render=False)

    def _reduction_preview_step(self):
        op = self._reduction_current_op()
        if op not in GUI_ID_PREFIX:
            return None
        return preview_construct_step(
            self._reduction,
            self._reduction_step_from_form("__preview__"),
        )

    def _reduction_update_live_preview(self):
        self._clear_reduction_preview()
        try:
            preview = self._reduction_preview_step()
            if preview is None:
                self.plotter.render()
                return
            self._reduction_draw_preview(preview)
        except (ValueError, KeyError, TypeError) as e:
            self._status(f"preview: {e}")
        except Exception:
            traceback.print_exc()
        self.plotter.render()

    def _reduction_draw_preview(self, preview) -> None:
        if preview.kind == "plane" and preview.plane is not None:
            corners = plane_patch_corners(
                preview.plane,
                center=preview.anchor,
                size_mm=preview.overlay_mm,
            )
            corners = self._to_view_points(corners)
            faces = np.array([3, 0, 1, 2, 3, 0, 2, 3], dtype=np.int64)
            mesh = pv.PolyData(corners, faces=faces)
            self.plotter.add_mesh(
                mesh,
                name="rd_preview_offset",
                color="#2ecc71",
                opacity=0.45,
                reset_camera=False,
                pickable=False,
                render=False,
            )
            self.plotter.add_mesh(
                mesh,
                name="rd_preview_offset_e",
                style="wireframe",
                color="#27ae60",
                line_width=2,
                reset_camera=False,
                pickable=False,
                render=False,
            )
            return
        if preview.kind == "line" and preview.line is not None:
            center = preview.anchor if preview.anchor is not None else preview.line.point
            seg = line_segment_points(
                preview.line,
                half_length_mm=float(preview.overlay_mm),
                center=center,
            )
            seg = self._to_view_points(seg)
            self.plotter.add_mesh(
                _line_tube_mesh(seg[0], seg[1], preview.overlay_width_mm),
                name="rd_preview_axis",
                color="#2ecc71",
                reset_camera=False,
                pickable=False,
                render=False,
            )
            return
        if preview.kind == "point" and preview.point is not None:
            if preview.segment_ends is not None:
                end_a, end_b = preview.segment_ends
                diam = float(
                    preview.overlay_width_mm
                    or self._reduction.display_default_mm.get("line_diameter", 1.0)
                )
                self.plotter.add_mesh(
                    _line_tube_mesh(
                        self._to_view_point(end_a),
                        self._to_view_point(end_b),
                        diam,
                    ),
                    name="rd_preview_axis",
                    color="#2ecc71",
                    reset_camera=False,
                    pickable=False,
                    render=False,
                )
                for i, pt in enumerate((end_a, end_b)):
                    self.plotter.add_mesh(
                        pv.Sphere(
                            radius=max(
                                float(self._reduction.display_default_mm.get("point", 4.0)),
                                0.5,
                            ),
                            center=self._to_view_point(pt).tolist(),
                        ),
                        name=f"rd_preview_end{i}",
                        color="#27ae60",
                        reset_camera=False,
                        pickable=False,
                        render=False,
                    )
            r = max(float(self._reduction.display_default_mm.get("point", 4.0)), 0.5)
            self.plotter.add_mesh(
                pv.Sphere(radius=r * 1.4, center=self._to_view_point(preview.point).tolist()),
                name="rd_preview_point",
                color="#2ecc71",
                reset_camera=False,
                pickable=False,
                render=False,
            )
            return

    def _reduction_apply(self):
        eid = self._reduction_editing_id()
        op = self._reduction_current_op()
        if self._reduction_is_update_mode() and eid is not None and op != "bind":
            self._reduction_update_step(eid)
            return
        if op == "bind":
            self._reduction_bind_active()
            return
        prefix = GUI_ID_PREFIX.get(op, "entity")
        entity_id = self._reduction_new_id(prefix)
        step = self._reduction_step_from_form(entity_id)
        self._reduction.apply_step(step)
        self._reduction_after_create(step)

    def _reduction_after_create(self, step: dict) -> None:
        entity_id = str(step["id"])
        self.rd_id_edit.clear()
        self._clear_reduction_preview()
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()
        self._status(f"created {entity_id}")

    def _reduction_status_with_replay(self, message: str) -> None:
        warnings = list(self._reduction.replay_warnings)
        if warnings:
            message = f"{message}  ({'; '.join(warnings)})"
        self._status(message)

    def _reduction_update_step(self, entity_id: str) -> None:
        existing = self._reduction.construct_step(entity_id)
        step = self._reduction_step_from_form(entity_id)
        if existing is not None and str(existing.get("op") or "") != str(step.get("op") or ""):
            raise ValueError(
                f"cannot change operation of {entity_id!r} while updating; "
                "switch to Create new to make a different kind of entity"
            )
        self._reduction.replace_construct_step(entity_id, step)
        self._rd_form_entity_id = entity_id
        self._reduction_refresh_view()
        self._reduction_status_with_replay(f"updated {entity_id}")

    def _reduction_anchor_for_group(self, g: dict) -> np.ndarray:
        if g.get("clicked") is not None:
            return np.asarray(g["clicked"], dtype=np.float64).reshape(3)
        idx = g.get("indices")
        if idx is not None and len(idx) and self.full_points.size:
            return np.mean(self.full_points[idx], axis=0)
        return np.zeros(3, dtype=np.float64)

    def _reduction_bind_active(self):
        key = self._reduction_combo_id(getattr(self, "rd_bind_combo", None))
        if not key:
            raise ValueError("choose a Groups plane")
        gid_s, _, pi_s = key.partition(":")
        g = self._get_group(int(gid_s))
        if g is None or g.get("fit") is None:
            raise ValueError("that Groups plane is no longer available; Fit again")
        planes = g["fit"].get("planes") or []
        pi = int(pi_s or 0)
        p = next((x for x in planes if int(x.get("plane_index", 0)) == pi), None)
        if p is None:
            raise ValueError(f"no plane_index={pi} on {g['name']}")
        alias = self.rd_id_edit.text().strip() or f"{g['name']}_{_plane_id_token(p)}"
        plane = Plane.from_array(p["abcd"])
        quality = {
            "status": p.get("status"),
            "mad_sigma_mm": p.get("mad_sigma_mm"),
            "threshold_mm": p.get("threshold_mm"),
            "n_points": p.get("n_points"),
            "bimodal": p.get("bimodal"),
            "reasons": p.get("reasons"),
        }
        self._reduction.bind_scanned(
            alias,
            plane,
            group_name=str(g["name"]),
            group_id=int(g["id"]),
            plane_index=pi,
            quality=quality,
            anchor=self._reduction_anchor_for_group(g),
        )
        self.rd_id_edit.clear()
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()
        self._reduction_status_with_replay(f"imported {alias!r} ← {g['name']} / p{pi}")

    def _reduction_clear(self):
        self._reduction.clear()
        if self._view_frame is not None:
            self._set_view_frame(None, reset_camera=False)
        self._reduction_refresh_view()
        self._reduction_restore_frame_combos()
        self._status("reduction session cleared")

    def _reduction_refresh_view(self):
        self._clear_reduction_preview()
        self._refresh_reduction_tree()
        self._refresh_measure_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()

    def _measure_operand_meta(self) -> tuple[tuple[str, str], tuple[str, str]]:
        op = self.rd_measure_op.currentData() if hasattr(self, "rd_measure_op") else None
        measure = MEASURE_OP_BY_KEY.get(op)
        if measure is None:
            return (("point", "point"), ("Point A", "Point B"))
        return (
            (measure.operands[0].kind, measure.operands[1].kind),
            (measure.operands[0].label, measure.operands[1].label),
        )

    def _measure_operand_keys(self) -> tuple[str, str]:
        op = self.rd_measure_op.currentData() if hasattr(self, "rd_measure_op") else None
        measure = MEASURE_OP_BY_KEY.get(op)
        if measure is None:
            return ("a", "b")
        return (measure.operands[0].key, measure.operands[1].key)

    def _reduction_on_measure_op(self, *_args) -> None:
        kinds, labels = self._measure_operand_meta()
        if hasattr(self, "rd_measure_a_lbl"):
            self.rd_measure_a_lbl.setText(labels[0])
            self.rd_measure_b_lbl.setText(labels[1])
        if hasattr(self, "rd_measure_a"):
            self._reduction_fill_combo(self.rd_measure_a, kind=kinds[0])
            self._reduction_fill_combo(self.rd_measure_b, kind=kinds[1])
        self._reduction_update_live_measure()

    def _measure_live_spec(self) -> dict | None:
        if not hasattr(self, "rd_measure_op"):
            return None
        op = self.rd_measure_op.currentData()
        a = self._reduction_combo_id(self.rd_measure_a)
        b = self._reduction_combo_id(self.rd_measure_b)
        if not op or not a or not b:
            return None
        ka, kb = self._measure_operand_keys()
        return {"id": "_live", "op": op, ka: a, kb: b}

    def _format_measure_value(self, rec: dict) -> str:
        v = float(rec["value"])
        unit = rec.get("unit", "mm")
        if unit == "deg":
            return f"{v:.3f}°"
        return f"{v:.3f} mm"

    def _reduction_update_live_measure(self, *_args) -> None:
        if not hasattr(self, "rd_measure_live"):
            return
        spec = self._measure_live_spec()
        ok = False
        text = "choose two operands"
        if spec is not None:
            try:
                rec = self._reduction.evaluate_measure(spec)
                text = self._format_measure_value(rec)
                ok = True
            except (KeyError, ValueError, TypeError) as e:
                text = str(e)
        self.rd_measure_live.setText(text)
        if hasattr(self, "rd_measure_add_btn"):
            self.rd_measure_add_btn.setEnabled(ok)
        self._refresh_measure_overlays(render=True)

    def _reduction_add_measure(self) -> None:
        spec = self._measure_live_spec()
        if spec is None:
            raise ValueError("choose two operands")
        spec = dict(spec)
        spec.pop("id", None)
        mid = self._reduction.add_measure(spec)
        self._refresh_measure_tree()
        self._refresh_measure_overlays(render=True)
        rec = self._reduction.evaluate_measure(
            next(m for m in self._reduction.measures if m["id"] == mid)
        )
        self._status(f"measure {mid}: {self._format_measure_value(rec)}")

    def _sync_measure_delete_enabled(self) -> None:
        if not hasattr(self, "rd_measure_del_btn"):
            return
        n = 0
        if hasattr(self, "rd_measure_tree"):
            n = len(self.rd_measure_tree.selectedItems())
        self.rd_measure_del_btn.setEnabled(n > 0)

    def _reduction_delete_measures(self) -> None:
        ids = []
        for item in self.rd_measure_tree.selectedItems():
            mid = item.data(0, Qt.UserRole)
            if mid:
                ids.append(str(mid))
        if not ids:
            raise ValueError("select a measurement to delete")
        for mid in ids:
            self._reduction.remove_measure(mid)
        self._refresh_measure_tree()
        self._refresh_measure_overlays(render=True)
        self._status("deleted " + ", ".join(ids))

    def _refresh_measure_tree(self) -> None:
        if not hasattr(self, "rd_measure_tree"):
            return
        prev = {
            str(item.data(0, Qt.UserRole))
            for item in self.rd_measure_tree.selectedItems()
            if item.data(0, Qt.UserRole)
        }
        self.rd_measure_tree.blockSignals(True)
        _reset_tree_widget(self.rd_measure_tree)
        labels = {
            "distance_points": "point - point",
            "distance_point_plane": "point - plane",
            "distance_point_line": "point - line",
            "angle_planes": "plane - plane",
            "angle_lines": "line - line",
            "angle_line_plane": "line - plane",
        }
        for spec in self._reduction.measures:
            try:
                rec = self._reduction.evaluate_measure(spec)
                val = self._format_measure_value(rec)
            except (KeyError, ValueError, TypeError):
                rec = spec
                val = "—"
            keys = [k for k in ("a", "b", "point", "plane", "line") if spec.get(k)]
            of = " · ".join(str(spec[k]) for k in keys)
            item = QTreeWidgetItem(
                [spec["id"], labels.get(spec["op"], spec["op"]), val, of]
            )
            item.setData(0, Qt.UserRole, spec["id"])
            self.rd_measure_tree.addTopLevelItem(item)
            if spec["id"] in prev:
                item.setSelected(True)
        self.rd_measure_tree.blockSignals(False)
        self.rd_measure_tree.resizeColumnToContents(0)
        self.rd_measure_tree.resizeColumnToContents(1)
        self._sync_measure_delete_enabled()

    def _clear_measure_overlays(self) -> None:
        for name in self._reduction_measure_actor_names:
            self.plotter.remove_actor(name, render=False)
        self._reduction_measure_actor_names = []

    def _measure_segment_survey(self, spec: dict):
        op = spec["op"]
        if op == "distance_points":
            return self._reduction.point(spec["a"]), self._reduction.point(spec["b"])
        if op == "distance_point_plane":
            p = self._reduction.point(spec["point"])
            foot = project_point_to_plane(p, self._reduction.plane(spec["plane"]))
            return p, foot
        if op == "distance_point_line":
            p = self._reduction.point(spec["point"])
            foot = project_point_to_line(p, self._reduction.line(spec["line"]))
            return p, foot
        return None

    def _add_measure_segment(self, name: str, p0, p1, *, color: str, diameter: float) -> None:
        seg = self._to_view_points(np.stack([p0, p1], axis=0))
        if float(np.linalg.norm(seg[1] - seg[0])) < 1e-9:
            return
        mesh = _line_tube_mesh(seg[0], seg[1], diameter)
        self.plotter.add_mesh(
            mesh,
            name=name,
            color=color,
            reset_camera=False,
            pickable=False,
            render=False,
        )
        self._reduction_measure_actor_names.append(name)

    def _refresh_measure_overlays(self, *, render: bool = True) -> None:
        if not hasattr(self, "plotter"):
            return
        self._clear_measure_overlays()
        diam = float(self._reduction.display_default_mm.get("line_diameter", 1.0))
        for spec in self._reduction.measures:
            try:
                ends = self._measure_segment_survey(spec)
            except (KeyError, ValueError, TypeError):
                continue
            if ends is None:
                continue
            self._add_measure_segment(
                f"rd_meas_{spec['id']}", ends[0], ends[1], color=RD_MEASURE, diameter=diam
            )
        live = self._measure_live_spec()
        if live is not None:
            try:
                ends = self._measure_segment_survey(live)
            except (KeyError, ValueError, TypeError):
                ends = None
            if ends is not None:
                self._add_measure_segment(
                    "rd_meas_live", ends[0], ends[1], color="#2ecc71", diameter=diam * 1.2
                )
        if render:
            self.plotter.render()

    def _reduction_try_bind_face(self, alias: str, spec: dict):
        """Resolve a recipe face from in-memory Groups, or None to use disk."""
        if not isinstance(spec, dict):
            raise ValueError(f"faces.{alias}: expected object, got {type(spec).__name__}")
        src = spec.get("from", "group")
        if src != "group":
            raise ValueError(f"faces.{alias}: unsupported from={src!r}")
        name = spec.get("name")
        group_id = spec.get("group_id")
        plane_index = int(spec.get("plane_index", 0))
        if name is not None and group_id is not None:
            raise ValueError(f"faces.{alias}: provide name or group_id, not both")
        if name is None and group_id is None:
            raise ValueError(f"faces.{alias}: need name or group_id")
        g = None
        if name is not None:
            matches = [x for x in self.groups if str(x["name"]) == str(name)]
            if len(matches) > 1:
                raise ValueError(f"faces.{alias}: multiple groups named {name!r}")
            if len(matches) == 1:
                g = matches[0]
        else:
            g = self._get_group(int(group_id))
        if g is None or not g.get("fit"):
            return None
        planes = g["fit"].get("planes") or []
        p = next(
            (x for x in planes if int(x.get("plane_index", 0)) == plane_index),
            None,
        )
        if p is None:
            raise KeyError(
                f"faces.{alias}: no plane_index={plane_index} on {g['name']}"
            )
        plane = Plane.from_array(p["abcd"])
        record = scanned_plane_record(
            plane,
            group_id=int(g["id"]),
            group_name=str(g["name"]),
            plane_index=plane_index,
            quality={
                k: p.get(k)
                for k in (
                    "status",
                    "mad_sigma_mm",
                    "threshold_mm",
                    "n_points",
                    "bimodal",
                    "reasons",
                )
                if p.get(k) is not None
            },
        )
        return plane, record, self._reduction_anchor_for_group(g)

    def _reduction_load_recipe_path(self, path: str | Path, *, confirm: bool = True) -> int:
        path = Path(path)
        if confirm and self._reduction.ids():
            answer = QMessageBox.question(
                self,
                "Load recipe?",
                "This replaces the current reduction session.\n\n"
                f"Load:\n{path}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return 0
        recipe = load_recipe(path)
        self._reduction.apply_recipe(
            recipe,
            project_dir=self.project_dir,
            bind_face=self._reduction_try_bind_face,
        )
        if self._view_frame is not None:
            self._set_view_frame(None, reset_camera=False)
        self._reduction_refresh_view()
        self._reduction_restore_frame_combos()
        return len(self._reduction.ids())

    def _reduction_load_recipe(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load reduction recipe",
            str(self.project_dir / "recipe.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        n = self._reduction_load_recipe_path(path, confirm=True)
        if n:
            if hasattr(self, "reduction_dock"):
                self.reduction_dock.raise_()
            self._status(f"loaded recipe → {path}  ({n} entities)")

    def _reduction_save_recipe(self):
        self._reduction_capture_frame_spec()
        recipe = self._reduction.to_recipe()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save reduction recipe",
            str(self.project_dir / "recipe.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        write_recipe_json(path, recipe)
        self._status(f"saved recipe → {path}")

    def _reduction_export_geometry(self):
        self._reduction_capture_frame_spec()
        want_aligned = bool(
            getattr(self, "rd_export_frame_cb", None) and self.rd_export_frame_cb.isChecked()
        )
        aligned_frame = self._reduction.rigid_frame() if want_aligned else None
        result = export_reduction_result(
            self._reduction,
            source_project=str(self.project_dir.resolve()),
            aligned_frame=aligned_frame,
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export geometry.json",
            str(self.project_dir / "geometry.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        write_geometry_json(path, result)
        # Also write recipe beside it for reproducibility.
        recipe_path = Path(path).with_name(
            Path(path).stem + "_recipe.json"
            if Path(path).name == "geometry.json"
            else Path(path).stem + ".recipe.json"
        )
        self._reduction_capture_frame_spec()
        write_recipe_json(recipe_path, self._reduction.to_recipe())
        if want_aligned and result.aligned is not None:
            frame_note = ", survey + aligned"
        elif want_aligned:
            frame_note = ", survey (set FRAME axis and origin for aligned)"
        else:
            frame_note = ", survey only"
        self._status(
            f"exported {path}  ({len(result.planes)} planes, "
            f"{len(result.lines)} lines, {len(result.points)} points{frame_note})"
        )

    def _refresh_reduction_tree(self):
        if not hasattr(self, "rd_tree"):
            return
        prev_tree = set(self._reduction_tree_selected_ids())
        prev_sel = set(self._reduction_selected_ids())
        self.rd_tree.blockSignals(True)
        _reset_tree_widget(self.rd_tree)
        bold = QFont()
        bold.setBold(True)
        for eid in self._reduction.ids():
            kind = self._reduction.kind_of(eid)
            rec = self._reduction.record_of(eid)
            type_label = RD_KIND_LABEL.get(kind, kind)
            size_txt = f"{self._reduction.overlay_mm(eid):g}"
            item = QTreeWidgetItem([eid, type_label, size_txt, self._reduction_detail_text(eid)])
            item.setData(0, Qt.UserRole, eid)
            item.setFlags(
                item.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsEnabled
                | Qt.ItemIsEditable
                | Qt.ItemIsSelectable
            )
            item.setCheckState(
                0,
                Qt.Checked if self._reduction.visible.get(eid, True) else Qt.Unchecked,
            )
            # Type column colour hint.
            if kind == "plane":
                tint = QColor(RD_PLANE_SCANNED if rec.get("provenance") == "scanned" else RD_PLANE_OFFSET)
            elif kind == "line":
                tint = QColor(RD_AXIS)
            else:
                tint = QColor(RD_POINT)
            item.setForeground(1, tint)
            if eid in prev_sel:
                item.setFont(0, bold)
                item.setFont(1, bold)
                item.setFont(2, bold)
                item.setFont(3, bold)
                for col in range(4):
                    item.setBackground(col, QColor("#e8f0fe"))
            self.rd_tree.addTopLevelItem(item)
            if eid in prev_tree:
                item.setSelected(True)
        muted = QColor("#8b919a")
        italic = QFont()
        italic.setItalic(True)
        for eid in self._reduction_aligned_axis_ids():
            label = ALIGNED_AXIS_LABELS.get(eid, eid)
            item = QTreeWidgetItem([label, "axis", "—", "FRAME triad"])
            item.setData(0, Qt.UserRole, eid)
            item.setFlags(
                Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
            )
            item.setCheckState(
                0,
                Qt.Checked if self._reduction.visible.get(eid, True) else Qt.Unchecked,
            )
            item.setForeground(0, muted)
            item.setForeground(1, QColor(RD_ALIGNED_AXIS.get(eid, RD_AXIS)))
            item.setForeground(2, muted)
            item.setForeground(3, muted)
            for col in range(4):
                item.setFont(col, italic)
            if eid in prev_sel:
                for col in range(4):
                    item.setBackground(col, QColor("#e8f0fe"))
            self.rd_tree.addTopLevelItem(item)
            if eid in prev_tree:
                item.setSelected(True)
        self.rd_tree.blockSignals(False)
        self.rd_tree.resizeColumnToContents(0)
        self.rd_tree.resizeColumnToContents(1)
        self._reduction_update_selection_label()
        self._sync_reduction_entity_actions()

    def _on_reduction_item_changed(self, item: QTreeWidgetItem, column: int):
        eid = item.data(0, Qt.UserRole)
        if not eid:
            return
        eid = str(eid)
        if column == 2:
            if eid not in self._reduction.ids():
                return
            raw = item.text(2).strip().replace("mm", "")
            try:
                size = float(raw)
            except ValueError:
                self.rd_tree.blockSignals(True)
                item.setText(2, f"{self._reduction.overlay_mm(eid):g}")
                self.rd_tree.blockSignals(False)
                self._status("size must be a number (mm)")
                return
            if size <= 0:
                self.rd_tree.blockSignals(True)
                item.setText(2, f"{self._reduction.overlay_mm(eid):g}")
                self.rd_tree.blockSignals(False)
                return
            self._reduction.set_overlay_mm(eid, size)
            self._reduction_sync_size_controls_from_selection()
            self._refresh_reduction_entity_overlays([eid])
            return
        if column in (1, 3):
            self.rd_tree.blockSignals(True)
            if is_aligned_axis_id(eid):
                item.setText(1, "axis")
                item.setText(3, "FRAME triad")
            elif eid in self._reduction.ids():
                kind = self._reduction.kind_of(eid)
                if column == 1:
                    item.setText(1, RD_KIND_LABEL.get(kind, kind))
                else:
                    item.setText(3, self._reduction_detail_text(eid))
            self.rd_tree.blockSignals(False)
            return
        if column != 0:
            return
        vis = item.checkState(0) == Qt.Checked
        was_vis = self._reduction.visible.get(eid, True)
        if is_aligned_axis_id(eid):
            self._reduction.visible[eid] = vis
            if vis != was_vis:
                self._apply_reduction_entity_visibility(eid, vis)
            return
        if eid in self._reduction.ids():
            self._reduction.visible[eid] = vis
        new_id = item.text(0).strip()
        if new_id != eid:
            try:
                self._reduction.rename(eid, new_id)
            except (KeyError, ValueError) as e:
                self.rd_tree.blockSignals(True)
                item.setText(0, eid)
                self.rd_tree.blockSignals(False)
                self._status(str(e))
                return
            item.setData(0, Qt.UserRole, new_id)
            self._reduction_refresh_operand_combos(rename=(eid, new_id))
            self._status(f"renamed {eid!r} → {new_id!r}")
            self._refresh_reduction_actors()
            return
        if vis != was_vis:
            self._apply_reduction_entity_visibility(eid, vis)

    def _sync_reduction_entity_actions(self):
        if not hasattr(self, "rd_delete_btn"):
            return
        real = [
            eid
            for eid in self._reduction_tree_selected_ids()
            if eid in self._reduction.ids()
        ]
        self.rd_delete_btn.setEnabled(bool(real))
        if hasattr(self, "rd_reset_size_btn"):
            has_size = bool(
                self._reduction_size_targets("plane")
                or self._reduction_size_targets("line")
                or self._reduction_size_targets("point")
            )
            self.rd_reset_size_btn.setEnabled(has_size)
        self._reduction_sync_size_controls_from_selection()
        self._reduction_sync_operation_from_selection()
        self._refresh_reduction_entity_overlays(self._reduction_overlay_ids())

    def _reduction_size_spin(self, kind: str) -> QDoubleSpinBox | None:
        return {
            "plane": getattr(self, "rd_patch_spin", None),
            "line": getattr(self, "rd_axis_spin", None),
            "line_diameter": getattr(self, "rd_axis_diam_spin", None),
            "point": getattr(self, "rd_point_spin", None),
        }.get(kind)

    def _reduction_size_targets(self, kind: str) -> list[str]:
        """Entities the DISPLAY slider should edit for ``kind`` (plane/line/point)."""
        def take(ids: list[str]) -> list[str]:
            return [
                eid
                for eid in ids
                if eid in self._reduction.ids() and self._reduction.kind_of(eid) == kind
            ]

        tree = take(self._reduction_tree_selected_ids())
        if tree:
            return tree
        return take(self._reduction_selected_ids())

    def _reduction_apply_overlay_size(self, kind: str, size_mm: float) -> list[str]:
        size_mm = float(size_mm)
        entity_kind = "line" if kind == "line_diameter" else kind
        targets = self._reduction_size_targets(entity_kind)
        if targets:
            for eid in targets:
                if kind == "line_diameter":
                    self._reduction.set_overlay_width_mm(eid, size_mm)
                else:
                    self._reduction.set_overlay_mm(eid, size_mm)
            if kind != "line_diameter":
                self._reduction_update_tree_size_cells(targets)
            return targets
        self._reduction.display_default_mm[kind] = size_mm
        if kind == "line_diameter":
            return [
                eid
                for eid in self._reduction.ids()
                if self._reduction.kind_of(eid) == "line"
                and eid not in self._reduction.display_width_mm
            ]
        affected = [
            eid
            for eid in self._reduction.ids()
            if self._reduction.kind_of(eid) == kind
            and eid not in self._reduction.display_mm
        ]
        self._reduction_update_tree_size_cells(affected)
        return affected

    def _reduction_reset_selected_overlay(self):
        ids = self._reduction_size_targets("plane")
        ids += [e for e in self._reduction_size_targets("line") if e not in ids]
        ids += [e for e in self._reduction_size_targets("point") if e not in ids]
        if not ids:
            ids = self._reduction_tree_selected_ids()
        for eid in ids:
            self._reduction.clear_overlay_mm(eid)
            self._reduction.clear_overlay_width_mm(eid)
        self._reduction_update_tree_size_cells(ids)
        self._reduction_sync_size_controls_from_selection()
        self._refresh_reduction_entity_overlays(ids)
        if ids:
            self._status("reset overlay size for " + ", ".join(ids))

    def _reduction_update_tree_size_cells(self, eids: list[str] | None = None):
        if not hasattr(self, "rd_tree"):
            return
        want = None if eids is None else set(eids)
        self.rd_tree.blockSignals(True)
        for i in range(self.rd_tree.topLevelItemCount()):
            item = self.rd_tree.topLevelItem(i)
            eid = item.data(0, Qt.UserRole)
            if not eid or (want is not None and str(eid) not in want):
                continue
            if str(eid) in self._reduction.ids():
                item.setText(2, f"{self._reduction.overlay_mm(str(eid)):g}")
        self.rd_tree.blockSignals(False)

    def _reduction_sync_size_controls_from_selection(self):
        if not hasattr(self, "rd_patch_spin"):
            return
        by_kind = {
            "plane": self._reduction_size_targets("plane"),
            "line": self._reduction_size_targets("line"),
            "point": self._reduction_size_targets("point"),
        }
        by_kind["line_diameter"] = by_kind["line"]
        labels = {
            "plane": "Plane",
            "line": "Axis ±",
            "line_diameter": "Axis ⌀",
            "point": "Point",
        }
        self._rd_size_loading = True
        try:
            for kind, eids in by_kind.items():
                spin = self._reduction_size_spin(kind)
                lbl = (self.rd_size_lbl or {}).get(kind)
                if not eids:
                    key = kind
                    fallback = 1.0 if kind == "line_diameter" else 200.0
                    val = float(self._reduction.display_default_mm.get(key, fallback))
                    if spin is not None:
                        spin.setValue(val)
                    if lbl is not None:
                        lbl.setText(f"{labels[kind]}  (default)")
                    continue
                if kind == "line_diameter":
                    sizes = [self._reduction.overlay_width_mm(e) for e in eids]
                else:
                    sizes = [self._reduction.overlay_mm(e) for e in eids]
                if spin is not None and len(set(round(s, 4) for s in sizes)) == 1:
                    spin.setValue(sizes[0])
                if lbl is not None:
                    if len(eids) == 1:
                        lbl.setText(f"{labels[kind]}  ·  {eids[0]}")
                    else:
                        lbl.setText(f"{labels[kind]}  ·  {len(eids)} selected")
        finally:
            self._rd_size_loading = False
        if hasattr(self, "rd_size_hint"):
            if any(by_kind[k] for k in ("plane", "line", "point")):
                self.rd_size_hint.setText(
                    "Sliders change the selected entity. Reset selected restores the default."
                )
            else:
                self.rd_size_hint.setText(
                    "No entity selected — sliders set the default size."
                )

    def _reduction_tree_selected_ids(self) -> list[str]:
        ids: list[str] = []
        if not hasattr(self, "rd_tree"):
            return ids
        for item in self.rd_tree.selectedItems():
            eid = item.data(0, Qt.UserRole)
            if eid and str(eid) not in ids:
                ids.append(str(eid))
        return ids

    def _reduction_aligned_axis_ids(self) -> list[str]:
        """FRAME triad ids shown as virtual ENTITIES rows."""
        available = self._reduction.available_aligned_axis_ids()
        return [eid for eid in ALIGNED_AXIS_IDS if eid in available]

    def _reduction_overlay_ids(self) -> list[str]:
        return list(self._reduction.ids()) + self._reduction_aligned_axis_ids()

    def _reduction_has_overlay(self, eid: str) -> bool:
        return eid in self._reduction.ids() or eid in self._reduction_aligned_axis_ids()

    def _reduction_overlay_selected_ids(self) -> set[str]:
        return set(self._reduction_selected_ids()) | set(
            self._reduction_tree_selected_ids()
        )

    def _reduction_delete_selected(self):
        ids = self._reduction_tree_selected_ids()
        if not ids:
            raise ValueError("select an entity in the list to delete")
        real = [eid for eid in ids if eid in self._reduction.ids()]
        if not real:
            raise ValueError("aligned axes cannot be renamed or deleted")
        removed: list[str] = []
        for eid in real:
            removed.extend(self._reduction.remove(eid))
        # Unique, keep order
        seen: set[str] = set()
        ordered = []
        for eid in removed:
            if eid not in seen:
                seen.add(eid)
                ordered.append(eid)
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._clear_reduction_preview()
        self._refresh_reduction_actors()
        if ordered:
            self._status("deleted " + ", ".join(ordered))

    def _reduction_detail_text(self, eid: str) -> str:
        rec = self._reduction.record_of(eid)
        kind = self._reduction.kind_of(eid)
        if self._view_frame is not None:
            rec = transform_record(kind, rec, self._view_frame)
        if kind == "plane" and "distance_mm" in rec:
            return f"offset {rec['distance_mm']:g} mm"
        if kind == "line":
            d = rec.get("direction") or [0, 0, 0]
            return f"dir ({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f})"
        if kind == "point":
            xyz = rec.get("xyz") or [0, 0, 0]
            return f"({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) mm"
        return str(rec.get("provenance", ""))

    def _reduction_actor_name(self, entity_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in entity_id)
        return f"rd_{safe}"

    def _clear_reduction_actors(self):
        for name in self._reduction_actor_names:
            self.plotter.remove_actor(name, render=False)
        self._reduction_actor_names = []

    def _reduction_label_position(self, eid: str) -> np.ndarray:
        if is_aligned_axis_id(eid):
            line = self._reduction.line(eid)
            return axis_arrow_points(line, self._reduction.overlay_mm(eid))[1]
        kind = self._reduction.kind_of(eid)
        anchor = self._reduction.anchors.get(eid)
        size = self._reduction.overlay_mm(eid)
        if kind == "plane":
            plane = self._reduction.plane(eid)
            center = anchor if anchor is not None else -plane.d * plane.normal
            center = center - plane.signed_distances(center.reshape(1, 3))[0] * plane.normal
            return center + plane.normal * (size * 0.08)
        if kind == "line":
            line = self._reduction.line(eid)
            seg = line_segment_points(line, half_length_mm=size, center=anchor)
            return 0.5 * (seg[0] + seg[1])
        return self._reduction.point(eid)

    def _reduction_point_radius_mm(self, eid: str | None = None) -> float:
        if eid is not None and eid in self._reduction.ids():
            return max(self._reduction.overlay_mm(eid), 0.5)
        return max(float(self._reduction.display_default_mm.get("point", 4.0)), 0.5)

    def _reduction_entity_actor_names(self, eid: str) -> list[str]:
        name = self._reduction_actor_name(eid)
        want = {
            name,
            f"{name}_e",
            f"{name}_n",
            f"{name}_ring",
            f"{name}_cap0",
            f"{name}_cap1",
            f"{name}_tip",
        }
        return [n for n in self._reduction_actor_names if n in want]

    def _remove_reduction_entity_overlay(self, eid: str) -> None:
        names = self._reduction_entity_actor_names(eid)
        for n in names:
            self.plotter.remove_actor(n, render=False)
        drop = set(names)
        self._reduction_actor_names = [n for n in self._reduction_actor_names if n not in drop]

    def _add_aligned_axis_overlay(self, eid: str, *, selected: bool) -> None:
        """RGB arrow from FRAME origin along the view +X/+Y/+Z direction."""
        line = self._reduction.line(eid)
        length = max(self._reduction.overlay_mm(eid), 1.0)
        origin, tip = axis_arrow_points(line, length)
        origin_v = self._to_view_point(origin)
        tip_v = self._to_view_point(tip)
        d = tip_v - origin_v
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return
        d = d / n
        tip_len = max(0.16 * n, 8.0)
        if tip_len > 0.4 * n:
            tip_len = 0.4 * n
        shaft_end = tip_v - tip_len * d
        diam = self._reduction.overlay_width_mm(eid)
        if selected:
            diam *= 1.35
        color = RD_ALIGNED_AXIS.get(eid, RD_AXIS)
        name = self._reduction_actor_name(eid)
        self.plotter.add_mesh(
            _line_tube_mesh(origin_v, shaft_end, diam),
            name=name,
            color=color,
            reset_camera=False,
            pickable=False,
            render=False,
        )
        self._reduction_actor_names.append(name)
        cone = pv.Cone(
            center=(tip_v - 0.5 * tip_len * d).tolist(),
            direction=d.tolist(),
            height=float(tip_len),
            radius=max(1.1 * diam, 0.4),
            resolution=16,
        )
        tname = name + "_tip"
        self.plotter.add_mesh(
            cone,
            name=tname,
            color=color,
            reset_camera=False,
            pickable=False,
            render=False,
        )
        self._reduction_actor_names.append(tname)
        if selected:
            cap = pv.Sphere(
                radius=max(0.7 * diam, 0.5),
                center=tip_v.tolist(),
            )
            cname = name + "_cap0"
            self.plotter.add_mesh(
                cap,
                name=cname,
                color=RD_SELECTED_RING,
                reset_camera=False,
                pickable=False,
                render=False,
            )
            self._reduction_actor_names.append(cname)

    def _add_reduction_entity_overlay(self, eid: str, *, selected: bool) -> None:
        kind = self._reduction.kind_of(eid)
        name = self._reduction_actor_name(eid)
        anchor = self._reduction.anchors.get(eid)
        size = self._reduction.overlay_mm(eid)
        try:
            if kind == "plane":
                plane = self._reduction.plane(eid)
                corners_s = plane_patch_corners(
                    plane, center=anchor, size_mm=size
                )
                corners = self._to_view_points(corners_s)
                faces = np.array([3, 0, 1, 2, 3, 0, 2, 3], dtype=np.int64)
                mesh = pv.PolyData(corners, faces=faces)
                rec = self._reduction.record_of(eid)
                color = (
                    RD_PLANE_SCANNED
                    if rec.get("provenance") == "scanned"
                    else RD_PLANE_OFFSET
                )
                opacity = 0.55 if selected else 0.35
                lw = 4 if selected else 2
                self.plotter.add_mesh(
                    mesh,
                    name=name,
                    color=color,
                    opacity=opacity,
                    reset_camera=False,
                    pickable=False,
                    render=False,
                )
                edge_name = name + "_e"
                edge_color = RD_SELECTED_RING if selected else color
                self.plotter.add_mesh(
                    mesh,
                    name=edge_name,
                    style="wireframe",
                    color=edge_color,
                    line_width=lw,
                    reset_camera=False,
                    pickable=False,
                    render=False,
                )
                self._reduction_actor_names.extend([name, edge_name])
                center = corners.mean(axis=0)
                tip = self._to_view_point(
                    corners_s.mean(axis=0) + plane.normal * (size * 0.2)
                )
                nar = pv.Line(center, tip)
                nname = name + "_n"
                self.plotter.add_mesh(
                    nar,
                    name=nname,
                    color=RD_NORMAL if not selected else "#ffffff",
                    line_width=3 if selected else 2,
                    reset_camera=False,
                    pickable=False,
                    render=False,
                )
                self._reduction_actor_names.append(nname)
            elif kind == "line":
                if is_aligned_axis_id(eid):
                    self._add_aligned_axis_overlay(eid, selected=selected)
                    return
                line = self._reduction.line(eid)
                seg = line_segment_points(
                    line, half_length_mm=size, center=anchor
                )
                seg = self._to_view_points(seg)
                diam = self._reduction.overlay_width_mm(eid)
                if selected:
                    diam *= 1.25
                mesh = _line_tube_mesh(seg[0], seg[1], diam)
                self.plotter.add_mesh(
                    mesh,
                    name=name,
                    color=RD_AXIS,
                    reset_camera=False,
                    pickable=False,
                    render=False,
                )
                self._reduction_actor_names.append(name)
                if selected:
                    cap_r = max(0.5 * self._reduction.overlay_width_mm(eid) * 1.6, 0.4)
                    for i, pt in enumerate(seg):
                        cap = pv.Sphere(
                            radius=cap_r, center=pt.tolist()
                        )
                        cname = f"{name}_cap{i}"
                        self.plotter.add_mesh(
                            cap,
                            name=cname,
                            color=RD_SELECTED_RING,
                            reset_camera=False,
                            pickable=False,
                            render=False,
                        )
                        self._reduction_actor_names.append(cname)
            elif kind == "point":
                pt = self._to_view_point(self._reduction.point(eid))
                r = size * (1.4 if selected else 1.0)
                mesh = pv.Sphere(radius=r, center=pt.tolist())
                self.plotter.add_mesh(
                    mesh,
                    name=name,
                    color=RD_POINT,
                    reset_camera=False,
                    pickable=False,
                    render=False,
                )
                self._reduction_actor_names.append(name)
                if selected:
                    ring = pv.Sphere(radius=r * 1.35, center=pt.tolist())
                    rname = name + "_ring"
                    self.plotter.add_mesh(
                        ring,
                        name=rname,
                        style="wireframe",
                        color=RD_SELECTED_RING,
                        line_width=2,
                        reset_camera=False,
                        pickable=False,
                        render=False,
                    )
                    self._reduction_actor_names.append(rname)
        except Exception:
            traceback.print_exc()

    def _refresh_reduction_entity_overlays(
        self, eids: list[str], *, render: bool = True
    ) -> None:
        """Rebuild overlays for ``eids`` only; leave the rest in place."""
        if not eids:
            if render:
                self.plotter.render()
            return
        selected = set(self._reduction_overlay_selected_ids())
        for eid in eids:
            self._remove_reduction_entity_overlay(eid)
            if self._reduction_has_overlay(eid) and self._reduction.visible.get(eid, True):
                self._add_reduction_entity_overlay(eid, selected=eid in selected)
        self._refresh_reduction_labels(render=False)
        if render:
            self.plotter.render()

    def _refresh_reduction_labels(self, *, render: bool = True) -> None:
        self.plotter.remove_actor("rd_labels", render=False)
        self._reduction_actor_names = [
            n for n in self._reduction_actor_names if n != "rd_labels"
        ]
        label_pts: list[np.ndarray] = []
        label_text: list[str] = []
        for eid in self._reduction_overlay_ids():
            if not self._reduction.visible.get(eid, True):
                continue
            try:
                label_pts.append(self._reduction_label_position(eid))
                label_text.append(ALIGNED_AXIS_LABELS.get(eid, eid))
            except Exception:
                traceback.print_exc()
        if label_pts:
            pts = self._to_view_points(np.stack(label_pts, axis=0))
            self.plotter.add_point_labels(
                pts,
                label_text,
                name="rd_labels",
                font_size=14,
                point_size=0,
                shape_opacity=0.75,
                fill_shape=True,
                reset_camera=False,
                pickable=False,
                render=False,
            )
            self._reduction_actor_names.append("rd_labels")
        if render:
            self.plotter.render()

    def _apply_reduction_entity_visibility(self, eid: str, visible: bool) -> None:
        """Show or hide one overlay without rebuilding the rest."""
        names = self._reduction_entity_actor_names(eid)
        actors = getattr(self.plotter, "actors", None) or {}
        if visible and not names:
            selected = eid in self._reduction_overlay_selected_ids()
            self._add_reduction_entity_overlay(eid, selected=selected)
            self._refresh_reduction_labels(render=True)
            return
        for n in names:
            actor = actors.get(n)
            if actor is not None:
                actor.SetVisibility(1 if visible else 0)
        self._refresh_reduction_labels(render=True)

    def _refresh_reduction_actors(self, *, render: bool = True):
        self._clear_reduction_actors()
        selected = self._reduction_overlay_selected_ids()
        for eid in self._reduction_overlay_ids():
            if not self._reduction.visible.get(eid, True):
                continue
            self._add_reduction_entity_overlay(eid, selected=eid in selected)
        self._refresh_reduction_labels(render=False)
        self._refresh_measure_overlays(render=False)
        if render:
            self.plotter.render()
