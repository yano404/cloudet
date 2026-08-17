"""Qt-based interactive plane picker (PySide6 + PyVista/VTK).

Usage: ``cloudet [project_dir] [--cloud <cloud>]``

Layout: 3D view (pyvistaqt QtInteractor) + left dock with a group/plane
tree and a settings form + right docks for residual u–v map and interactive
geometry reduction (offset / intersect → recipe + geometry.json). A FRAME
card can align the 3D view so a chosen axis is global +Z; survey data is
unchanged. Data
model: 1 group = N planes; fitting a group (multi-plane extraction)
populates plane children in the tree.

The project / output folder can be chosen from the CLI or from the GUI
(PROJECT card → Browse...). Groups, settings, and the VTK log are written
there.

Interaction:
    P            pick a plane region at the mouse position (PyVista picking)
    append mode  checkbox: picks add to the active group instead
    Fit          runs multi-plane extraction, shows per-plane QC
    Save All     writes groups/ + manifest (fits computed if missing)
    Cmd/Ctrl+drag on the residual u–v map: select a rectangle for hist/refit

Picking is done on the *displayed* (decimated) cloud to find the click
position, but region extraction, fitting and saving always use the
full-resolution cloud.
"""

from __future__ import annotations

import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv

from PySide6.QtCore import QEvent, Qt, qInstallMessageHandler
from PySide6.QtGui import QColor, QFont, QHelpEvent, QKeySequence, QShortcut
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
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from pyvistaqt import QtInteractor

from cloudet.array_backend import (
    cupy_unavailable_reason,
    device_name,
    resolve_compute_backend,
    set_default_backend,
)
from cloudet.groups import load_groups
from cloudet.mainplane import MainPlaneParams, extract_main_plane
from cloudet.multiplane import MultiPlaneParams, _bimodality_flag, extract_planes
from cloudet.neighbors import (
    VoxelHashGrid,
    depth_layers_along_ray,
    display_xyz,
    resolve_display_backend,
)
from cloudet.picking import PickParams, pick_plane_region
from cloudet.pipeline import residual_uv_map
from cloudet.plane import Plane, mad_sigma
from cloudet.plyio import read_ply_xyz
from cloudet.project import (
    PickerSettings,
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
from cloudet.reduce import ReductionSession, load_recipe, write_geometry_json, write_recipe_json
from cloudet.frame import RigidFrame, transform_record, with_aligned_copy
from cloudet.geometry import line_segment_points, plane_patch_corners
from cloudet.settings_apply import classify_settings_apply

GROUP_COLORS = [
    (0.90, 0.25, 0.25), (0.25, 0.55, 0.95), (0.20, 0.75, 0.35),
    (0.95, 0.75, 0.20), (0.75, 0.35, 0.85), (0.20, 0.80, 0.80),
    (0.95, 0.45, 0.15), (0.60, 0.60, 0.60),
]

# Reduction overlay colours (3D + tree hints).
_RD_PLANE_SCANNED = "#4a90d9"
_RD_PLANE_OFFSET = "#e07b39"
_RD_AXIS = "#c0392b"
_RD_POINT = "#f1c40f"
_RD_SELECTED_RING = "#ffffff"
_RD_NORMAL = "#2ecc71"
_RD_KIND_LABEL = {"plane": "plane", "line": "line", "point": "point"}


DEPTH_TIP = (
    "Depth candidates along the last picked view ray (near to far).\n"
    "The ray does not stop at the visible face, so the back side can appear.\n"
    "When only one candidate exists, < and > are disabled.\n"
    "Press P first, then use < > only when overlap exists."
)

SETTINGS_HELP_DEFAULT = """
<div style='white-space: normal;'>
<b>SETTINGS WORKFLOW</b>
<p>
1. Press <b>P</b> to test a pick with the current values.<br>
2. Change Detection only when face extraction needs adjustment.<br>
3. Change Display to improve view density or interaction speed.<br>
4. Press <b>Apply</b> to use changes; <b>Save as Default</b> keeps them.
</p>
<span style='color: #777;'>Hover a setting name or value to see what it controls.</span>
</div>
"""

# GUI fit ceiling (mm). Adaptive robust fit never exceeds this.
# Residual plot half-range (±) is controlled separately in the DISPLAY card.
FIT_MAX_THRESHOLD_MM = 0.5


def _plane_label(p: dict) -> str:
    """User-facing plane name; defaults to p0, p1, …"""
    name = p.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"p{int(p.get('plane_index', 0))}"


def _plane_id_token(p: dict) -> str:
    """Reduction / file-safe token derived from the display name."""
    token = re.sub(r"[^\w.-]+", "_", _plane_label(p), flags=re.UNICODE).strip("._")
    return token or f"p{int(p.get('plane_index', 0))}"


def _rdbu_r_colormap():
    """Diverging blue–white–red map matching the former matplotlib RdBu_r."""
    import pyqtgraph as pg

    pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    colors = np.array(
        [
            [5, 48, 97, 255],
            [67, 147, 195, 255],
            [247, 247, 247, 255],
            [214, 96, 77, 255],
            [103, 0, 31, 255],
        ],
        dtype=np.ubyte,
    )
    return pg.ColorMap(pos, colors)


class _UVSelectViewBox:
    """Factory: Cmd/Ctrl+drag selects a rectangle; otherwise normal pan/zoom.

    On macOS the Command key is Qt.MetaModifier; Control is still accepted.
    """

    @staticmethod
    def create(on_rect):
        import pyqtgraph as pg
        from PySide6.QtCore import Qt

        class VB(pg.ViewBox):
            def __init__(self):
                super().__init__(enableMenu=False)
                self._on_rect = on_rect
                self._sel_start = None
                self._rubber = None
                self.setAspectLocked(True)

            def mouseDragEvent(self, ev, axis=None):
                # Cmd (Meta) on macOS; Ctrl on Windows/Linux (and macOS Control).
                select_mod = ev.modifiers() & (
                    Qt.ControlModifier | Qt.MetaModifier
                )
                if ev.button() != Qt.LeftButton or not select_mod:
                    return super().mouseDragEvent(ev, axis)
                ev.accept()
                pt = self.mapSceneToView(ev.scenePos())
                if ev.isStart():
                    self._sel_start = pt
                    if self._rubber is not None:
                        try:
                            self.removeItem(self._rubber)
                        except Exception:
                            pass
                    self._rubber = pg.RectROI(
                        [pt.x(), pt.y()],
                        [1e-6, 1e-6],
                        pen=pg.mkPen("#1a4a9a", width=1.5),
                        movable=False,
                        resizable=False,
                    )
                    self.addItem(self._rubber)
                elif self._sel_start is None:
                    return
                elif ev.isFinish():
                    u0, u1 = sorted((self._sel_start.x(), pt.x()))
                    v0, v1 = sorted((self._sel_start.y(), pt.y()))
                    if self._rubber is not None:
                        try:
                            self.removeItem(self._rubber)
                        except Exception:
                            pass
                        self._rubber = None
                    self._sel_start = None
                    if (u1 - u0) > 0 and (v1 - v0) > 0 and self._on_rect is not None:
                        self._on_rect(u0, u1, v0, v1)
                else:
                    u0, u1 = sorted((self._sel_start.x(), pt.x()))
                    v0, v1 = sorted((self._sel_start.y(), pt.y()))
                    if self._rubber is not None:
                        self._rubber.setPos([u0, v0], update=False)
                        self._rubber.setSize(
                            [max(u1 - u0, 1e-6), max(v1 - v0, 1e-6)], update=True
                        )

        return VB()


UI_STYLE = """
QFrame#card {
    background: palette(base);
    border: 1px solid palette(mid);
    border-radius: 8px;
}
QLabel#sectionTitle,
QToolButton#sectionTitle {
    color: palette(mid);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
QToolButton#sectionTitle {
    border: none;
    background: transparent;
    padding: 0px;
}
QToolButton#sectionTitle:hover {
    color: palette(text);
}
QLabel#muted {
    color: palette(mid);
    font-size: 11px;
}
/* A stylesheet on a QPushButton drops the native macOS look entirely, so
   every role has to draw its own background, border and states — without
   them a button reads as disabled. */
QPushButton#primaryBtn {
    font-weight: 700;
    border-radius: 6px;
    padding: 5px 10px;
    border: 1px solid palette(highlight);
    background: palette(highlight);
    color: palette(highlighted-text);
}
QPushButton#primaryBtn:hover:!disabled {
    background: palette(highlight);
    border-color: palette(text);
}
QPushButton#primaryBtn:pressed {
    background: palette(dark);
    border-color: palette(dark);
}
QPushButton#secondaryBtn {
    border-radius: 6px;
    padding: 5px 10px;
    border: 1px solid palette(mid);
    background: palette(button);
    color: palette(button-text);
}
QPushButton#secondaryBtn:hover:!disabled {
    border-color: palette(highlight);
}
QPushButton#secondaryBtn:pressed {
    background: palette(midlight);
}
QPushButton#dangerBtn {
    border-radius: 6px;
    padding: 5px 10px;
    border: 1px solid #a44;
    background: palette(button);
    color: #a44;
}
QPushButton#dangerBtn:hover:!disabled {
    background: #a44;
    color: palette(highlighted-text);
}
QPushButton#dangerBtn:pressed {
    background: #833;
    border-color: #833;
    color: palette(highlighted-text);
}
QPushButton#primaryBtn:disabled,
QPushButton#secondaryBtn:disabled,
QPushButton#dangerBtn:disabled {
    background: palette(window);
    border-color: palette(midlight);
    color: palette(mid);
}
QLabel#badgeOk {
    color: #1f7a1f;
    font-weight: 700;
}
QLabel#badgeWarn {
    color: #a05a00;
    font-weight: 700;
}
QTreeWidget {
    border: 1px solid palette(mid);
    border-radius: 6px;
    alternate-background-color: palette(alternate-base);
}
QTreeWidget::item:selected {
    background: palette(highlight);
    color: palette(highlighted-text);
}
QToolTip {
    padding: 8px;
    border: 1px solid palette(mid);
    border-radius: 5px;
    background: palette(base);
    color: palette(text);
}
"""


def _make_collapsible_card(
    heading: str,
    *,
    expanded: bool = True,
    header_extra: QWidget | None = None,
) -> tuple[QFrame, QVBoxLayout]:
    """Card with a clickable title that shows or hides the body."""
    card = QFrame()
    card.setObjectName("card")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(10, 8, 10, 8)
    outer.setSpacing(6)

    toggle = QToolButton()
    toggle.setObjectName("sectionTitle")
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
    toggle.setText(heading)
    toggle.setCheckable(True)
    toggle.setChecked(expanded)
    toggle.setAutoRaise(True)
    toggle.setCursor(Qt.PointingHandCursor)
    toggle.setToolTip("Click to expand or collapse this section.")

    body = QWidget()
    body_lay = QVBoxLayout(body)
    body_lay.setContentsMargins(0, 0, 0, 0)
    body_lay.setSpacing(6)
    body.setVisible(expanded)

    def _on_toggled(checked: bool, *, t=toggle, b=body) -> None:
        b.setVisible(checked)
        t.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    toggle.toggled.connect(_on_toggled)
    if header_extra is None:
        outer.addWidget(toggle)
    else:
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(6)
        hdr.addWidget(toggle, stretch=1)
        hdr.addWidget(header_extra)
        outer.addLayout(hdr)
    outer.addWidget(body)
    return card, body_lay


def group_color(gid: int) -> np.ndarray:
    return np.asarray(GROUP_COLORS[gid % len(GROUP_COLORS)], dtype=np.float64)


def _reset_tree_widget(tree: QTreeWidget) -> None:
    """Empty a tree without leaving macOS accessibility on a deleted row.

    ``QTreeWidget.clear()`` can make Qt log
    ``qt.accessibility.table: Cell requested for row N is out of bounds
    for table with 0 rows`` (and occasionally crash) because Cocoa still
    holds a ``QAccessibleTableCell`` for the old current index.
    """
    tree.clearSelection()
    tree.setCurrentItem(None)
    while tree.topLevelItemCount():
        tree.takeTopLevelItem(0)


def _reset_combo(combo: QComboBox) -> None:
    """Clear a combo after dropping the current index (same a11y issue)."""
    combo.setCurrentIndex(-1)
    combo.clear()


def _line_tube_mesh(p0, p1, diameter_mm: float) -> pv.PolyData:
    """Finite tube along p0→p1. Diameter is in the same mm units as the cloud."""
    radius = max(0.5 * float(diameter_mm), 0.05)
    return pv.Line(p0, p1).tube(radius=radius, n_sides=10)


_QT_MSG_PREV = None
_QT_MSG_FILTER_INSTALLED = False


def _qt_message_filter(mode, context, message: str) -> None:
    if "out of bounds for table with" in message:
        return
    if _QT_MSG_PREV is not None:
        _QT_MSG_PREV(mode, context, message)
        return
    sys.stderr.write(message + "\n")


def _install_qt_message_filter() -> None:
    """Hide the known-harmless macOS QTreeWidget accessibility warning."""
    global _QT_MSG_PREV, _QT_MSG_FILTER_INSTALLED
    if _QT_MSG_FILTER_INSTALLED:
        return
    _QT_MSG_PREV = qInstallMessageHandler(_qt_message_filter)
    _QT_MSG_FILTER_INSTALLED = True


# vtkOutputWindow only borrows the Python wrappers, so they must outlive the call.
_VTK_LOG_KEEPALIVE: list = []


def route_vtk_messages_to_file(path: Path) -> Path | None:
    """Send VTK's own errors and warnings to *path* instead of the console.

    ``import pyvista`` unconditionally calls ``send_errors_to_logging()``, which
    hands every VTK message to the unconfigured root logger. Long OpenGL
    messages re-enter that handler and bury the terminal under repeated
    ``PyVista error in handling VTK error message``. Writing straight to a file
    removes the logging round-trip and keeps the original text readable.

    ``CLOUDET_VTK_LOG=0`` keeps PyVista's default; any other value is used as the
    log path.
    """
    setting = os.environ.get("CLOUDET_VTK_LOG", "")
    if setting == "0":
        return None
    if setting not in ("", "1"):
        path = Path(setting).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _VTK_LOG_KEEPALIVE.extend(pv.set_error_output_file(path))
    except Exception:
        return None
    return path


class PickerWindow(QMainWindow):
    def __init__(self, project_dir: str, pcd_path: str | None = None):
        super().__init__()
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._vtk_log_path = route_vtk_messages_to_file(self.project_dir / "vtk.log")
        self._fit_log_path = self.project_dir / "fit.log"
        self.settings = load_settings(self.project_dir, warn=self._status)
        set_default_backend(self.settings.detection.compute_backend)

        self.full_points: np.ndarray = np.zeros((0, 3))
        self.grid: VoxelHashGrid | None = None
        self.pcd_path = ""
        self._base_display_xyz: np.ndarray | None = None
        self._n_displayed = 0

        self.groups: list[dict] = []
        self.active_group_id: int | None = None
        self.next_group_id = 0
        # Depth-layer candidates for the last screen pick (front → back).
        self._pick_layers: list[dict] = []
        self._pick_layer_i: int = 0
        self._pick_replace_gid: int | None = None
        self._settings_dirty: bool = False
        self._settings_help_targets: dict[QWidget, str] = {}
        self._status_default: str = "Ready"
        self._active_plane_index: int = 0
        self._tree_focus: str = "group"  # "group" | "plane"
        self._uv_glw = None
        self._uv_plot = None
        self._hist_plot = None
        self._uv_img = None
        self._uv_cbar = None
        self._uv_roi = None
        self._uv_cmap = None
        self._hist_bar = None
        self._hist_lines: list = []
        self._uv_rect = None  # (u0, u1, v0, v1) in mm, or None = whole face
        self._uv_view: dict | None = None
        self._uv_roi_block = False
        self._uv_map_mode = "base"  # "base" | "refit"
        self._reduction = ReductionSession()
        self._reduction_actor_names: list[str] = []
        self._rd_offset_sync = False
        self._view_frame: RigidFrame | None = None

        self.setWindowTitle(f"cloudet - {self.project_dir.name}")
        self.resize(1760, 980)

        # --- 3D view -----------------------------------------------------
        central = QWidget()
        vlay = QVBoxLayout(central)
        vlay.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(central)
        vlay.addWidget(self.plotter)
        self.setCentralWidget(central)
        self.plotter.set_background("white")
        self._axes_widget = self.plotter.add_axes()
        self._place_orientation_axes()
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            show_message="P : Pick",
            use_picker=True,
            show_point=False,
            tolerance=0.01,
        )

        # --- left dock -----------------------------------------------------
        self._build_dock()
        self._build_uv_dock()
        self._build_reduction_dock()
        self._build_shortcuts()
        self._rebuild_status_default()
        self._refresh_frame_overlay()
        self.statusBar().showMessage(self._status_default)

        if pcd_path:
            self.pcd_edit_path = pcd_path
        else:
            manifest = read_manifest(self.project_dir)
            self.pcd_edit_path = (
                manifest.get("source", {}).get("path", "") if manifest else ""
            )
        if self.pcd_edit_path:
            self.cloud_label.setText(Path(self.pcd_edit_path).name)
            self.cloud_label.setToolTip(self.pcd_edit_path)
        self._update_source_meta()
        self._update_project_labels()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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

        # ---- Operation picker -------------------------------------------
        op_card, op_lay = _mini_card("OPERATION")
        self.rd_op_combo = QComboBox()
        for label, key in (
            ("Import plane from Groups", "bind"),
            ("Offset plane", "offset"),
            ("Intersect 2 planes → axis", "intersect_planes"),
            ("Line ∩ plane → point", "intersect_line_plane"),
            ("3 planes → corner", "intersect_three"),
            ("Point + normal → axis", "line_from_point_normal"),
            ("2 points → axis", "line_from_two_points"),
            ("Line ∩ 2 planes → midpoint", "midpoint_line_planes"),
        ):
            self.rd_op_combo.addItem(label, key)
        self.rd_op_combo.currentIndexChanged.connect(self._reduction_on_op_changed)
        op_lay.addWidget(self.rd_op_combo)
        lay.addWidget(op_card)

        # ---- Context panel (one page per operation) ---------------------
        ctx_card, ctx_lay = _mini_card("PARAMETERS")
        self.rd_stack = QStackedWidget()

        # Shared id field lives above the stack so every op can name the result.
        id_form = QFormLayout()
        id_form.setContentsMargins(0, 0, 0, 0)
        id_form.setSpacing(4)
        self.rd_id_edit = QLineEdit()
        self.rd_id_edit.setPlaceholderText("result id (optional, auto if empty)")
        id_form.addRow("New id", self.rd_id_edit)
        ctx_lay.addLayout(id_form)

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

        ctx_lay.addWidget(self.rd_stack)

        self.rd_apply_btn = QPushButton("Apply")
        self.rd_apply_btn.setObjectName("primaryBtn")
        self.rd_apply_btn.clicked.connect(
            lambda: self._guard(self._reduction_apply, busy=False)
        )
        ctx_lay.addWidget(self.rd_apply_btn)
        lay.addWidget(ctx_card)

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
                if not self._rd_size_loading:
                    self._reduction_apply_overlay_size(kind, float(val))
                self._refresh_reduction_actors()
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
                if not self._rd_size_loading:
                    self._reduction_apply_overlay_size(kind, val)
                self._refresh_reduction_actors()
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
            c.currentIndexChanged.connect(self._sync_frame_align_enabled)
        self.rd_frame_flip = QComboBox()
        self.rd_frame_flip.addItem("along axis", False)
        self.rd_frame_flip.addItem("flipped", True)
        frame_form.addRow("Axis", self.rd_frame_axis)
        frame_form.addRow("Origin", self.rd_frame_origin)
        frame_form.addRow("+Z", self.rd_frame_flip)
        frame_lay.addLayout(frame_form)
        frame_btn_row = QHBoxLayout()
        self.rd_frame_align_btn = QPushButton("Align Z")
        self.rd_frame_align_btn.setObjectName("primaryBtn")
        self.rd_frame_align_btn.setEnabled(False)
        self.rd_frame_align_btn.setToolTip(
            "Show the cloud and overlays with this axis as global +Z "
            "and this origin at (0, 0, 0). Survey data is not rewritten."
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
            "When Align Z is active, geometry.json keeps survey numbers and "
            "adds an aligned copy under \"aligned\" plus the frame pose. "
            "Off: survey coordinates only. The recipe is never transformed."
        )
        exp_lay.addWidget(self.rd_export_frame_cb)
        exp_row = QHBoxLayout()
        load_recipe_btn = QPushButton("Load recipe…")
        load_recipe_btn.setObjectName("secondaryBtn")
        load_recipe_btn.setToolTip(
            "Replace this session with a saved recipe.json. "
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

        dock.setWidget(w)
        dock.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.reduction_dock = dock
        if hasattr(self, "uv_dock"):
            self.tabifyDockWidget(self.uv_dock, dock)
            self.uv_dock.raise_()

        self._rd_offset_sync = False
        self._reduction_on_op_changed()
        self._update_frame_controls()

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self, lambda: self._guard(self._save_all))
        QShortcut(QKeySequence("F"), self, lambda: self._guard(self._fit_active))
        QShortcut(QKeySequence("Backspace"), self, lambda: self._guard(self._delete_active))
        rd_del = QShortcut(QKeySequence("Delete"), self.rd_tree)
        rd_del.setContext(Qt.WidgetWithChildrenShortcut)
        rd_del.activated.connect(
            lambda: self._guard(self._reduction_delete_selected, busy=False)
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

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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
        return f"frame: aligned ({axis}, {origin}{extra})"

    def _rebuild_status_default(self) -> None:
        ready = "Ready"
        ready += f"  |  {self._compute_status_suffix()}"
        ready += f"  |  {self._frame_status_text()}"
        if getattr(self, "_vtk_log_path", None) is not None:
            ready += f"  |  VTK messages -> {self._vtk_log_path}"
        ready += f"  |  fit timing -> {self._fit_log_path}"
        self._status_default = ready

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

    def _sync_frame_align_enabled(self) -> None:
        if not hasattr(self, "rd_frame_align_btn"):
            return
        axis = self._reduction_combo_id(getattr(self, "rd_frame_axis", None))
        origin = self._reduction_combo_id(getattr(self, "rd_frame_origin", None))
        self.rd_frame_align_btn.setEnabled(bool(axis and origin))

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
        axis_id = self._reduction_combo_id(getattr(self, "rd_frame_axis", None))
        origin_id = self._reduction_combo_id(getattr(self, "rd_frame_origin", None))
        if not axis_id or not origin_id:
            raise ValueError("choose an axis and an origin")
        flip = bool(self.rd_frame_flip.currentData()) if hasattr(self, "rd_frame_flip") else False
        line = self._reduction.line(axis_id)
        origin = self._reduction.point(origin_id)
        frame = RigidFrame.align_z(
            line.direction,
            origin,
            flip_z=flip,
            axis_id=axis_id,
            origin_id=origin_id,
        )
        self._set_view_frame(frame, reset_camera=True)
        self._status(self._frame_status_text())

    def _frame_survey(self) -> None:
        if self._view_frame is None:
            return
        self._set_view_frame(None, reset_camera=True)
        self._status("frame: survey")

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

    # ------------------------------------------------------------------
    # Interactive geometry reduction
    # ------------------------------------------------------------------

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
    ) -> None:
        ids = self._reduction.ids(kind=kind) if kind else self._reduction.ids()
        combo.blockSignals(True)
        _reset_combo(combo)
        combo.addItem(placeholder, None)
        for eid in ids:
            tag = _RD_KIND_LABEL.get(self._reduction.kind_of(eid), "")
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
        self, *, rename: tuple[str, str] | None = None
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
            "frame_axis": self._reduction_combo_id(getattr(self, "rd_frame_axis", None)),
            "frame_origin": self._reduction_combo_id(
                getattr(self, "rd_frame_origin", None)
            ),
        }
        if rename:
            old, new = rename
            keep = {k: (new if v == old else v) for k, v in keep.items()}
        if hasattr(self, "rd_offset_plane"):
            self._reduction_fill_combo(
                self.rd_offset_plane, kind="plane", keep=keep["offset"]
            )
            self._reduction_fill_combo(self.rd_p2_a, kind="plane", keep=keep["p2a"])
            self._reduction_fill_combo(self.rd_p2_b, kind="plane", keep=keep["p2b"])
            self._reduction_fill_combo(self.rd_lp_line, kind="line", keep=keep["lp_line"])
            self._reduction_fill_combo(
                self.rd_lp_plane, kind="plane", keep=keep["lp_plane"]
            )
            self._reduction_fill_combo(self.rd_p3_a, kind="plane", keep=keep["p3a"])
            self._reduction_fill_combo(self.rd_p3_b, kind="plane", keep=keep["p3b"])
            self._reduction_fill_combo(self.rd_p3_c, kind="plane", keep=keep["p3c"])
            self._reduction_fill_combo(
                self.rd_pn_point, kind="point", keep=keep["pn_point"]
            )
            self._reduction_fill_combo(
                self.rd_pn_plane, kind="plane", keep=keep["pn_plane"]
            )
            if hasattr(self, "rd_pp_a"):
                self._reduction_fill_combo(self.rd_pp_a, kind="point", keep=keep["pp_a"])
                self._reduction_fill_combo(self.rd_pp_b, kind="point", keep=keep["pp_b"])
            if hasattr(self, "rd_mp_line"):
                self._reduction_fill_combo(self.rd_mp_line, kind="line", keep=keep["mp_line"])
                self._reduction_fill_combo(self.rd_mp_a, kind="plane", keep=keep["mp_a"])
                self._reduction_fill_combo(self.rd_mp_b, kind="plane", keep=keep["mp_b"])
            if hasattr(self, "rd_frame_axis"):
                self._reduction_fill_combo(
                    self.rd_frame_axis, kind="line", keep=keep.get("frame_axis")
                )
                self._reduction_fill_combo(
                    self.rd_frame_origin, kind="point", keep=keep.get("frame_origin")
                )
                if rename and self._view_frame is not None:
                    old, new = rename
                    self._view_frame = self._view_frame.relabel(old, new)
                    self._update_frame_controls()
                    self._rebuild_status_default()
                self._sync_frame_align_enabled()

    def _reduction_selected_ids(self) -> list[str]:
        """Operand ids for the current operation (from PARAMETERS combos)."""
        op = self._reduction_current_op()
        ids: list[str] = []
        if op == "offset":
            eid = self._reduction_combo_id(getattr(self, "rd_offset_plane", None))
            if eid:
                ids.append(eid)
        elif op == "intersect_planes":
            for c in (getattr(self, "rd_p2_a", None), getattr(self, "rd_p2_b", None)):
                eid = self._reduction_combo_id(c)
                if eid:
                    ids.append(eid)
        elif op == "intersect_line_plane":
            for c in (getattr(self, "rd_lp_line", None), getattr(self, "rd_lp_plane", None)):
                eid = self._reduction_combo_id(c)
                if eid:
                    ids.append(eid)
        elif op == "intersect_three":
            for c in (
                getattr(self, "rd_p3_a", None),
                getattr(self, "rd_p3_b", None),
                getattr(self, "rd_p3_c", None),
            ):
                eid = self._reduction_combo_id(c)
                if eid:
                    ids.append(eid)
        elif op == "line_from_point_normal":
            for c in (getattr(self, "rd_pn_point", None), getattr(self, "rd_pn_plane", None)):
                eid = self._reduction_combo_id(c)
                if eid:
                    ids.append(eid)
        elif op == "line_from_two_points":
            for c in (getattr(self, "rd_pp_a", None), getattr(self, "rd_pp_b", None)):
                eid = self._reduction_combo_id(c)
                if eid:
                    ids.append(eid)
        elif op == "midpoint_line_planes":
            for c in (
                getattr(self, "rd_mp_line", None),
                getattr(self, "rd_mp_a", None),
                getattr(self, "rd_mp_b", None),
            ):
                eid = self._reduction_combo_id(c)
                if eid:
                    ids.append(eid)
        return ids

    def _reduction_current_op(self) -> str:
        if not hasattr(self, "rd_op_combo"):
            return "bind"
        return str(self.rd_op_combo.currentData())

    def _reduction_on_op_changed(self, *_args):
        op = self._reduction_current_op()
        page = {
            "bind": 0,
            "offset": 1,
            "intersect_planes": 2,
            "intersect_line_plane": 3,
            "intersect_three": 4,
            "line_from_point_normal": 5,
            "line_from_two_points": 6,
            "midpoint_line_planes": 7,
        }.get(op, 0)
        if hasattr(self, "rd_stack"):
            self.rd_stack.setCurrentIndex(page)
        labels = {
            "bind": "Import plane",
            "offset": "Apply offset",
            "intersect_planes": "Create axis",
            "intersect_line_plane": "Create point",
            "intersect_three": "Create corner",
            "line_from_point_normal": "Create axis",
            "line_from_two_points": "Create axis",
            "midpoint_line_planes": "Create midpoint",
        }
        if hasattr(self, "rd_apply_btn"):
            self.rd_apply_btn.setText(labels.get(op, "Apply"))
        self._reduction_refresh_operand_combos()
        self._reduction_update_selection_label()
        self._refresh_reduction_actors()
        self._reduction_update_live_preview()

    def _reduction_on_operand_combo(self, *_args):
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
            tag = _RD_KIND_LABEL.get(kind, kind)
            parts.append(f"{eid} [{tag}]")
        self.rd_selection_label.setText("Operands: " + ", ".join(parts))

    def _reduction_entity_color(self, eid: str, *, selected: bool) -> str:
        kind = self._reduction.kind_of(eid)
        if kind == "plane":
            rec = self._reduction.record_of(eid)
            base = _RD_PLANE_SCANNED if rec.get("provenance") == "scanned" else _RD_PLANE_OFFSET
        elif kind == "line":
            base = _RD_AXIS
        else:
            base = _RD_POINT
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
        self._reduction_update_offset_preview()

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
        self._reduction_update_offset_preview()

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

    def _reduction_update_live_preview(self):
        op = self._reduction_current_op()
        if op == "offset":
            self._reduction_update_offset_preview()
        elif op in ("line_from_point_normal", "line_from_two_points"):
            self._reduction_update_axis_preview()
        elif op == "midpoint_line_planes":
            self._reduction_update_midpoint_preview()
        else:
            self._clear_reduction_preview()
            self.plotter.render()

    def _reduction_update_offset_preview(self):
        """Live-preview an offset plane without committing to the session."""
        self._clear_reduction_preview()
        if self._reduction_current_op() != "offset":
            self.plotter.render()
            return
        ids = self._reduction_selected_ids()
        if len(ids) != 1 or self._reduction.kind_of(ids[0]) != "plane":
            self.plotter.render()
            return
        try:
            from cloudet.geometry import offset_plane

            src = self._reduction.plane(ids[0])
            dist = float(self.rd_offset_spin.value())
            plane = offset_plane(src, dist)
            anchor = self._reduction.anchors.get(ids[0])
            patch = self._reduction.overlay_mm(ids[0])
            corners = plane_patch_corners(plane, center=anchor, size_mm=patch)
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
        except Exception:
            traceback.print_exc()
        self.plotter.render()

    def _reduction_update_axis_preview(self):
        """Live-preview an axis construct without committing."""
        self._clear_reduction_preview()
        op = self._reduction_current_op()
        line = None
        center = None
        extra_half = 0.0
        try:
            if op == "line_from_point_normal":
                from cloudet.geometry import line_from_point_normal

                point_id = self._reduction_combo_id(getattr(self, "rd_pn_point", None))
                plane_id = self._reduction_combo_id(getattr(self, "rd_pn_plane", None))
                if not point_id or not plane_id:
                    self.plotter.render()
                    return
                line = line_from_point_normal(
                    self._reduction.point(point_id),
                    self._reduction.plane(plane_id),
                )
                center = line.point
            elif op == "line_from_two_points":
                from cloudet.geometry import line_from_two_points

                a = self._reduction_combo_id(getattr(self, "rd_pp_a", None))
                b = self._reduction_combo_id(getattr(self, "rd_pp_b", None))
                if not a or not b or a == b:
                    self.plotter.render()
                    return
                pa = self._reduction.point(a)
                pb = self._reduction.point(b)
                line = line_from_two_points(pa, pb)
                center = 0.5 * (pa + pb)
                extra_half = 0.5 * float(np.linalg.norm(pb - pa))
            else:
                self.plotter.render()
                return
            half = max(
                float(self._reduction.display_default_mm.get("line", 300.0)),
                extra_half,
            )
            diam = float(self._reduction.display_default_mm.get("line_diameter", 1.0))
            seg = line_segment_points(line, half_length_mm=half, center=center)
            seg = self._to_view_points(seg)
            self.plotter.add_mesh(
                _line_tube_mesh(seg[0], seg[1], diam),
                name="rd_preview_axis",
                color="#2ecc71",
                reset_camera=False,
                pickable=False,
                render=False,
            )
        except Exception:
            traceback.print_exc()
        self.plotter.render()

    def _reduction_update_midpoint_preview(self):
        """Live-preview the clipped segment and its midpoint."""
        self._clear_reduction_preview()
        if self._reduction_current_op() != "midpoint_line_planes":
            self.plotter.render()
            return
        line_id = self._reduction_combo_id(getattr(self, "rd_mp_line", None))
        a = self._reduction_combo_id(getattr(self, "rd_mp_a", None))
        b = self._reduction_combo_id(getattr(self, "rd_mp_b", None))
        if not line_id or not a or not b or a == b:
            self.plotter.render()
            return
        try:
            from cloudet.geometry import intersect_line_plane, midpoint_line_planes

            line = self._reduction.line(line_id)
            pa = self._reduction.plane(a)
            pb = self._reduction.plane(b)
            end_a = intersect_line_plane(line, pa)
            end_b = intersect_line_plane(line, pb)
            mid = midpoint_line_planes(line, pa, pb)
            diam = float(self._reduction.display_default_mm.get("line_diameter", 1.0))
            self.plotter.add_mesh(
                _line_tube_mesh(self._to_view_point(end_a), self._to_view_point(end_b), diam),
                name="rd_preview_axis",
                color="#2ecc71",
                reset_camera=False,
                pickable=False,
                render=False,
            )
            r = max(float(self._reduction.display_default_mm.get("point", 8.0)), 0.5)
            self.plotter.add_mesh(
                pv.Sphere(radius=r * 1.4, center=self._to_view_point(mid).tolist()),
                name="rd_preview_point",
                color="#2ecc71",
                reset_camera=False,
                pickable=False,
                render=False,
            )
            for i, pt in enumerate((end_a, end_b)):
                self.plotter.add_mesh(
                    pv.Sphere(radius=r * 0.7, center=self._to_view_point(pt).tolist()),
                    name=f"rd_preview_end{i}",
                    color="#27ae60",
                    reset_camera=False,
                    pickable=False,
                    render=False,
                )
        except Exception:
            traceback.print_exc()
        self.plotter.render()

    def _reduction_apply(self):
        op = self._reduction_current_op()
        if op == "bind":
            self._reduction_bind_active()
        elif op == "offset":
            self._reduction_offset()
        elif op == "intersect_planes":
            self._reduction_intersect_planes()
        elif op == "intersect_line_plane":
            self._reduction_intersect_line_plane()
        elif op == "intersect_three":
            self._reduction_intersect_three()
        elif op == "line_from_point_normal":
            self._reduction_line_from_point_normal()
        elif op == "line_from_two_points":
            self._reduction_line_from_two_points()
        elif op == "midpoint_line_planes":
            self._reduction_midpoint_line_planes()
        else:
            raise ValueError(f"unknown operation {op!r}")

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
            raise ValueError("choose a Groups plane in PARAMETERS")
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
        self._status(f"imported {alias!r} ← {g['name']} / p{pi}")

    def _reduction_offset(self):
        ids = self._reduction_selected_ids()
        if len(ids) != 1:
            raise ValueError("Offset needs exactly 1 selected plane")
        of = ids[0]
        if self._reduction.kind_of(of) != "plane":
            raise ValueError(f"{of!r} is not a plane")
        eid = self._reduction_new_id("offset")
        dist = float(self.rd_offset_spin.value())
        self._reduction.offset(eid, of, dist)
        self.rd_id_edit.clear()
        self._clear_reduction_preview()
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()
        self._status(f"offset {eid}: {of} by {dist:g} mm")

    def _reduction_intersect_planes(self):
        ids = self._reduction_selected_ids()
        if len(ids) != 2:
            raise ValueError("Intersect planes needs exactly 2 selected planes")
        for i in ids:
            if self._reduction.kind_of(i) != "plane":
                raise ValueError(f"{i!r} is not a plane")
        eid = self._reduction_new_id("axis")
        self._reduction.intersect_planes(eid, ids[0], ids[1])
        self.rd_id_edit.clear()
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()
        self._status(f"axis {eid}: {ids[0]} ∩ {ids[1]}")

    def _reduction_intersect_line_plane(self):
        ids = self._reduction_selected_ids()
        if len(ids) != 2:
            raise ValueError("Line ∩ plane needs 1 line + 1 plane selected")
        kinds = {i: self._reduction.kind_of(i) for i in ids}
        lines = [i for i, k in kinds.items() if k == "line"]
        planes = [i for i, k in kinds.items() if k == "plane"]
        if len(lines) != 1 or len(planes) != 1:
            raise ValueError("select exactly one line and one plane")
        eid = self._reduction_new_id("point")
        self._reduction.intersect_line_plane(eid, lines[0], planes[0])
        self.rd_id_edit.clear()
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()
        self._status(f"point {eid}: {lines[0]} ∩ {planes[0]}")

    def _reduction_intersect_three(self):
        ids = self._reduction_selected_ids()
        if len(ids) != 3:
            raise ValueError("3 planes → point needs exactly 3 selected planes")
        for i in ids:
            if self._reduction.kind_of(i) != "plane":
                raise ValueError(f"{i!r} is not a plane")
        eid = self._reduction_new_id("corner")
        self._reduction.intersect_three_planes(eid, ids[0], ids[1], ids[2])
        self.rd_id_edit.clear()
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()
        self._status(f"corner {eid}: {ids[0]} ∩ {ids[1]} ∩ {ids[2]}")

    def _reduction_line_from_point_normal(self):
        point_id = self._reduction_combo_id(getattr(self, "rd_pn_point", None))
        plane_id = self._reduction_combo_id(getattr(self, "rd_pn_plane", None))
        if not point_id or not plane_id:
            raise ValueError("Point + normal needs a point and a plane")
        if self._reduction.kind_of(point_id) != "point":
            raise ValueError(f"{point_id!r} is not a point")
        if self._reduction.kind_of(plane_id) != "plane":
            raise ValueError(f"{plane_id!r} is not a plane")
        eid = self._reduction_new_id("axis")
        self._reduction.line_from_point_normal(eid, point_id, plane_id)
        self.rd_id_edit.clear()
        self._clear_reduction_preview()
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()
        self._status(f"axis {eid}: through {point_id}, dir = normal({plane_id})")

    def _reduction_line_from_two_points(self):
        a = self._reduction_combo_id(getattr(self, "rd_pp_a", None))
        b = self._reduction_combo_id(getattr(self, "rd_pp_b", None))
        if not a or not b:
            raise ValueError("2 points → axis needs two points")
        if a == b:
            raise ValueError("choose two different points")
        if self._reduction.kind_of(a) != "point":
            raise ValueError(f"{a!r} is not a point")
        if self._reduction.kind_of(b) != "point":
            raise ValueError(f"{b!r} is not a point")
        eid = self._reduction_new_id("axis")
        self._reduction.line_from_two_points(eid, a, b)
        self.rd_id_edit.clear()
        self._clear_reduction_preview()
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()
        self._status(f"axis {eid}: {a} → {b}")

    def _reduction_midpoint_line_planes(self):
        line_id = self._reduction_combo_id(getattr(self, "rd_mp_line", None))
        a = self._reduction_combo_id(getattr(self, "rd_mp_a", None))
        b = self._reduction_combo_id(getattr(self, "rd_mp_b", None))
        if not line_id or not a or not b:
            raise ValueError("midpoint needs 1 axis and 2 planes")
        if a == b:
            raise ValueError("choose two different planes")
        if self._reduction.kind_of(line_id) != "line":
            raise ValueError(f"{line_id!r} is not an axis")
        if self._reduction.kind_of(a) != "plane":
            raise ValueError(f"{a!r} is not a plane")
        if self._reduction.kind_of(b) != "plane":
            raise ValueError(f"{b!r} is not a plane")
        eid = self._reduction_new_id("mid")
        self._reduction.midpoint_line_planes(eid, line_id, a, b)
        self.rd_id_edit.clear()
        self._clear_reduction_preview()
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()
        self._status(f"midpoint {eid}: {line_id} ∩ {a} / {b}")

    def _reduction_clear(self):
        self._reduction.clear()
        self._reduction_refresh_view()
        self._status("reduction session cleared")

    def _reduction_refresh_view(self):
        self._clear_reduction_preview()
        self._refresh_reduction_tree()
        self._reduction_refresh_operand_combos()
        self._refresh_reduction_actors()

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
        record = {
            "abcd": plane.as_array().tolist(),
            "provenance": "scanned",
            "group_id": int(g["id"]),
            "group_name": str(g["name"]),
            "plane_index": plane_index,
            "quality": {
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
        }
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
        self._reduction_refresh_view()
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
        result = self._reduction.to_result(
            source_project=str(self.project_dir.resolve())
        )
        include_aligned = (
            bool(getattr(self, "rd_export_frame_cb", None) and self.rd_export_frame_cb.isChecked())
            and self._view_frame is not None
        )
        if include_aligned:
            result = with_aligned_copy(result, self._view_frame)
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
        write_recipe_json(recipe_path, self._reduction.to_recipe())
        if include_aligned:
            frame_note = f", survey + {self._frame_status_text()}"
        else:
            frame_note = ", frame: survey"
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
            type_label = _RD_KIND_LABEL.get(kind, kind)
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
                tint = QColor(_RD_PLANE_SCANNED if rec.get("provenance") == "scanned" else _RD_PLANE_OFFSET)
            elif kind == "line":
                tint = QColor(_RD_AXIS)
            else:
                tint = QColor(_RD_POINT)
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
            self._refresh_reduction_actors()
            return
        if column in (1, 3):
            self.rd_tree.blockSignals(True)
            if eid in self._reduction.ids():
                kind = self._reduction.kind_of(eid)
                if column == 1:
                    item.setText(1, _RD_KIND_LABEL.get(kind, kind))
                else:
                    item.setText(3, self._reduction_detail_text(eid))
            self.rd_tree.blockSignals(False)
            return
        if column != 0:
            return
        vis = item.checkState(0) == Qt.Checked
        was_vis = self._reduction.visible.get(eid, True)
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
        n = len(self._reduction_tree_selected_ids()) if hasattr(self, "rd_tree") else 0
        self.rd_delete_btn.setEnabled(n > 0)
        if hasattr(self, "rd_reset_size_btn"):
            has_size = bool(
                self._reduction_size_targets("plane")
                or self._reduction_size_targets("line")
                or self._reduction_size_targets("point")
            )
            self.rd_reset_size_btn.setEnabled(has_size)
        self._reduction_sync_size_controls_from_selection()

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

    def _reduction_apply_overlay_size(self, kind: str, size_mm: float) -> None:
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
            return
        self._reduction.display_default_mm[kind] = size_mm
        if kind == "line_diameter":
            return
        affected = [
            eid
            for eid in self._reduction.ids()
            if self._reduction.kind_of(eid) == kind
            and eid not in self._reduction.display_mm
        ]
        self._reduction_update_tree_size_cells(affected)

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
        self._refresh_reduction_actors()
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

    def _reduction_delete_selected(self):
        ids = self._reduction_tree_selected_ids()
        if not ids:
            raise ValueError("select an entity in the list to delete")
        removed: list[str] = []
        for eid in ids:
            if eid not in self._reduction.ids():
                continue
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
        return max(float(self._reduction.display_default_mm.get("point", 8.0)), 0.5)

    def _reduction_entity_actor_names(self, eid: str) -> list[str]:
        name = self._reduction_actor_name(eid)
        return [
            n for n in self._reduction_actor_names
            if n == name or n.startswith(name + "_")
        ]

    def _refresh_reduction_labels(self, *, render: bool = True) -> None:
        self.plotter.remove_actor("rd_labels", render=False)
        self._reduction_actor_names = [
            n for n in self._reduction_actor_names if n != "rd_labels"
        ]
        label_pts: list[np.ndarray] = []
        label_text: list[str] = []
        for eid in self._reduction.ids():
            if not self._reduction.visible.get(eid, True):
                continue
            try:
                label_pts.append(self._reduction_label_position(eid))
                label_text.append(eid)
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
            self._refresh_reduction_actors()
            return
        for n in names:
            actor = actors.get(n)
            if actor is not None:
                actor.SetVisibility(1 if visible else 0)
        self._refresh_reduction_labels(render=True)

    def _refresh_reduction_actors(self, *, render: bool = True):
        self._clear_reduction_actors()
        selected = set(self._reduction_selected_ids())
        for eid in self._reduction.ids():
            if not self._reduction.visible.get(eid, True):
                continue
            kind = self._reduction.kind_of(eid)
            name = self._reduction_actor_name(eid)
            anchor = self._reduction.anchors.get(eid)
            sel = eid in selected
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
                        _RD_PLANE_SCANNED
                        if rec.get("provenance") == "scanned"
                        else _RD_PLANE_OFFSET
                    )
                    opacity = 0.55 if sel else 0.35
                    lw = 4 if sel else 2
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
                    edge_color = _RD_SELECTED_RING if sel else color
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
                        color=_RD_NORMAL if not sel else "#ffffff",
                        line_width=3 if sel else 2,
                        reset_camera=False,
                        pickable=False,
                        render=False,
                    )
                    self._reduction_actor_names.append(nname)
                elif kind == "line":
                    line = self._reduction.line(eid)
                    seg = line_segment_points(
                        line, half_length_mm=size, center=anchor
                    )
                    seg = self._to_view_points(seg)
                    diam = self._reduction.overlay_width_mm(eid)
                    if sel:
                        diam *= 1.25
                    mesh = _line_tube_mesh(seg[0], seg[1], diam)
                    self.plotter.add_mesh(
                        mesh,
                        name=name,
                        color=_RD_AXIS,
                        reset_camera=False,
                        pickable=False,
                        render=False,
                    )
                    self._reduction_actor_names.append(name)
                    if sel:
                        cap_r = max(0.5 * self._reduction.overlay_width_mm(eid) * 1.6, 0.4)
                        for i, pt in enumerate(seg):
                            cap = pv.Sphere(
                                radius=cap_r, center=pt.tolist()
                            )
                            cname = f"{name}_cap{i}"
                            self.plotter.add_mesh(
                                cap,
                                name=cname,
                                color=_RD_SELECTED_RING,
                                reset_camera=False,
                                pickable=False,
                                render=False,
                            )
                            self._reduction_actor_names.append(cname)
                elif kind == "point":
                    pt = self._to_view_point(self._reduction.point(eid))
                    r = size * (1.4 if sel else 1.0)
                    mesh = pv.Sphere(radius=r, center=pt.tolist())
                    self.plotter.add_mesh(
                        mesh,
                        name=name,
                        color=_RD_POINT,
                        reset_camera=False,
                        pickable=False,
                        render=False,
                    )
                    self._reduction_actor_names.append(name)
                    if sel:
                        ring = pv.Sphere(radius=r * 1.35, center=pt.tolist())
                        rname = name + "_ring"
                        self.plotter.add_mesh(
                            ring,
                            name=rname,
                            style="wireframe",
                            color=_RD_SELECTED_RING,
                            line_width=2,
                            reset_camera=False,
                            pickable=False,
                            render=False,
                        )
                        self._reduction_actor_names.append(rname)
            except Exception:
                traceback.print_exc()
        self._refresh_reduction_labels(render=False)
        if render:
            self.plotter.render()

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
                self._status(
                    f"decimating {len(self.full_points):,} points for display "
                    f"({backend}) ..."
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
        self._status(
            f"decimating {len(self.full_points):,} points for display ({backend}) ..."
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
        self._ensure_grid()

    def _ensure_grid(self) -> VoxelHashGrid:
        if len(self.full_points) == 0:
            raise ValueError("no cloud loaded")
        cell = VoxelHashGrid.cell_size_for_radius(
            self.settings.detection.local_radius_mm
        )
        if self.grid is not None and self.grid.cell_size == cell:
            return self.grid
        self._status(
            f"building spatial index ({len(self.full_points):,} pts, "
            f"cell={cell:.1f} mm) ..."
        )
        QApplication.processEvents()
        self.grid = VoxelHashGrid(self.full_points, cell_size=cell)
        self._status("spatial index ready")
        return self.grid

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _ensure_base_display_xyz(self) -> np.ndarray:
        if self._base_display_xyz is not None:
            return self._base_display_xyz
        v = self.settings.view
        self._base_display_xyz = display_xyz(
            self.full_points,
            v.display_voxel_size_mm,
            int(v.display_max_points),
            backend=v.display_downsample_backend,
        )
        return self._base_display_xyz

    def _refresh_base_actor(self):
        self.plotter.remove_actor("base", render=False)
        self._n_displayed = 0
        if len(self.full_points) == 0:
            self._base_display_xyz = None
            return
        v = self.settings.view
        xyz = self._to_view_points(self._ensure_base_display_xyz())
        self._n_displayed = len(xyz)
        self.plotter.add_points(
            xyz,
            name="base",
            color=(0.82, 0.82, 0.82),
            point_size=v.base_point_size,
            render_points_as_spheres=False,
            reset_camera=False,
            pickable=True,
        )
        self.plotter.render()

    def _clear_active_plane_bbox(self):
        self.plotter.remove_actor("active_plane_bbox", render=False)

    def _refresh_active_plane_bbox(self, *, render: bool = True) -> None:
        """Draw a thin oriented wireframe around the active fitted plane only."""
        self._clear_active_plane_bbox()
        g = self._active_group()
        p = self._active_plane_entry()
        if (
            g is None
            or p is None
            or self.full_points.size == 0
            or len(g.get("indices", [])) == 0
        ):
            if render:
                self.plotter.render()
            return
        if p.get("uv_basis") is None or p.get("uv_local_idx") is None:
            if self._ensure_plane_uv(g, p) is None:
                if render:
                    self.plotter.render()
                return
        basis = p.get("uv_basis") or {}
        if basis.get("basis") != "minrect" or "lo" not in basis or "hi" not in basis:
            if render:
                self.plotter.render()
            return
        u_ax = np.asarray(basis["u"], dtype=np.float64)
        v_ax = np.asarray(basis["v"], dtype=np.float64)
        center = np.asarray(basis["center"], dtype=np.float64)
        lo = np.asarray(basis["lo"], dtype=np.float64).copy()
        hi = np.asarray(basis["hi"], dtype=np.float64).copy()
        n_ax = np.asarray(Plane.from_array(p["abcd"]).normal, dtype=np.float64)
        n_ax = n_ax / max(float(np.linalg.norm(n_ax)), 1e-12)
        if float(np.dot(np.cross(u_ax, v_ax), n_ax)) < 0:
            v_ax = -v_ax
            lo[1], hi[1] = -hi[1], -lo[1]
        span = np.maximum(hi - lo, 1e-6)
        pad = np.maximum(0.01 * span, 0.2)
        mad = float(p.get("mad_sigma_mm", 0.05))
        pad[2] = max(float(pad[2]), 3.0 * mad, 0.15)
        lo = lo - pad
        hi = hi + pad
        box = pv.Box(bounds=(float(lo[0]), float(hi[0]),
                             float(lo[1]), float(hi[1]),
                             float(lo[2]), float(hi[2])))
        tf = np.eye(4, dtype=np.float64)
        tf[:3, 0] = u_ax
        tf[:3, 1] = v_ax
        tf[:3, 2] = n_ax
        tf[:3, 3] = center
        box.transform(tf, inplace=True)
        if self._view_frame is not None:
            box.points = self._to_view_points(np.asarray(box.points, dtype=np.float64))
        self.plotter.add_mesh(
            box,
            name="active_plane_bbox",
            style="wireframe",
            color="#1a4a9a",
            line_width=2.5,
            reset_camera=False,
            pickable=False,
        )
        if render:
            self.plotter.render()

    def _refresh_group_actors(self):
        for g in self.groups:
            self._refresh_group_actor(g)
        self.plotter.render()

    def _refresh_group_actor(self, g):
        name = f"group_{g['id']:03d}"
        self.plotter.remove_actor(name, render=False)
        visible = g["visible"] and not (
            self.solo_cb.isChecked() and g["id"] != self.active_group_id
        )
        if not visible:
            return
        v = self.settings.view
        pts = self.full_points[g["indices"]]
        xyz = display_xyz(
            pts,
            v.display_voxel_size_mm,
            int(v.display_max_points),
            backend=v.display_downsample_backend,
        )
        xyz = self._to_view_points(xyz)
        active = g["id"] == self.active_group_id
        color = tuple(float(c) for c in g["color"])
        self.plotter.add_points(
            xyz,
            name=name,
            color=color,
            point_size=v.active_point_size if active else v.inactive_point_size,
            render_points_as_spheres=False,
            reset_camera=False,
            pickable=False,  # always pick on the base cloud (depth layers)
        )

    def _set_actor_point_sizes_only(self):
        """Update point sizes via VTK properties (no point re-upload)."""
        v = self.settings.view
        actors = getattr(self.plotter, "actors", None) or {}
        base = actors.get("base")
        if base is not None:
            base.GetProperty().SetPointSize(float(v.base_point_size))
        for g in self.groups:
            actor = actors.get(f"group_{g['id']:03d}")
            if actor is None:
                continue
            size = (
                v.active_point_size
                if g["id"] == self.active_group_id
                else v.inactive_point_size
            )
            actor.GetProperty().SetPointSize(float(size))
        self.plotter.render()

    # ------------------------------------------------------------------
    # picking / groups
    # ------------------------------------------------------------------

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
                    _bimodality_flag(
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
            labels.extend(f"{g['name']}/{_plane_label(p)}" for p in gone)
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
            if hasattr(self, "rd_tree"):
                self._refresh_reduction_tree()
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
                    label = _plane_label(p)
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
                        src_label = _plane_label(src_p) if src_p is not None else (
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
                if _plane_label(other) == new_name:
                    self.tree.blockSignals(True)
                    item.setText(0, _plane_label(p))
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


def run_picker_qt(project_dir: str, pcd_path: str | None = None) -> None:
    _install_qt_message_filter()
    app = QApplication.instance() or QApplication([])
    win = PickerWindow(project_dir, pcd_path)
    win.show()
    app.exec()
