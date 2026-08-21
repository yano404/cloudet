"""cloudet desktop main window (PySide6 + PyVista/VTK).

Usage: ``cloudet [project_dir] [--cloud <cloud>]``

Layout: 3D view (pyvistaqt QtInteractor) + left dock with a group/plane
tree and a settings form + right docks for residual u–v map and interactive
geometry reduction (offset / intersect → recipe + geometry.json). A FRAME
card can align the 3D view so a chosen axis is global +Z; survey data is
unchanged. Data model: 1 group = N planes; fitting a group (multi-plane
extraction) populates plane children in the tree.

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

from pathlib import Path

import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from pyvistaqt import QtInteractor

from cloudet.core.array_backend import set_default_backend
from cloudet.reduction.frame import RigidFrame
from cloudet.core.neighbors import VoxelHashGrid
from cloudet.project import load_manifest, load_settings
from cloudet.reduction import ReductionSession
from cloudet.ui.app_common import AppCommonMixin
from cloudet.ui.frame_mixin import FrameMixin
from cloudet.ui.groups_mixin import GroupsMixin
from cloudet.ui.icons import app_icon
from cloudet.ui.qt_helpers import install_qt_message_filter, route_vtk_messages_to_file
from cloudet.ui.reduction_mixin import ReductionMixin
from cloudet.ui.render_mixin import RenderMixin
from cloudet.ui.uv_mixin import UvMixin


class CloudetAppWindow(
    AppCommonMixin,
    GroupsMixin,
    RenderMixin,
    FrameMixin,
    ReductionMixin,
    UvMixin,
    QMainWindow,
):
    """Main application window for cloudet."""

    def __init__(self, project_dir: str, cloud_path: str | None = None):
        super().__init__()
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._vtk_log_path = route_vtk_messages_to_file(self.project_dir / "vtk.log")
        self._fit_log_path = self.project_dir / "fit.log"
        self.settings = load_settings(self.project_dir, warn=self._status)
        set_default_backend(self.settings.detection.compute_backend)

        self.full_points: np.ndarray = np.zeros((0, 3))
        self.grid: VoxelHashGrid | None = None
        self.cloud_path = ""
        self._base_display_xyz: np.ndarray | None = None
        self._n_displayed = 0

        self.groups: list[dict] = []
        self.active_group_id: int | None = None
        self.next_group_id = 0
        # Depth-layer candidates for the last screen pick (front → back).
        self._pick_layers: list[dict] = []
        self._pick_layer_i: int = 0
        self._pick_replace_gid: int | None = None
        # Cylinder 3-point circumference seeds (survey XYZ), length 0–2 while pending.
        self._cyl_seed_points: list[np.ndarray] = []
        self._settings_dirty: bool = False
        self._settings_help_targets: dict = {}
        self._status_default: str = "Ready"
        self._active_plane_index: int = 0
        self._active_circle_index: int = 0
        self._active_cylinder_index: int = 0
        self._tree_focus: str = "group"  # "group" | "plane" | "circle" | "cylinder"
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
        self._reduction_measure_actor_names: list[str] = []
        self._rd_offset_sync = False
        self._rd_loading_step = False
        self._rd_form_entity_id: str | None = None
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

        # --- docks -------------------------------------------------------
        self._build_dock()
        self._build_uv_dock()
        self._build_reduction_dock()
        self._build_measure_dock()
        self._build_shortcuts()
        self._rebuild_status_default()
        self._refresh_frame_overlay()
        self.statusBar().showMessage(self._status_default)

        if cloud_path:
            self.cloud_edit_path = cloud_path
        else:
            manifest = load_manifest(self.project_dir)
            self.cloud_edit_path = (
                manifest.get("source", {}).get("path", "") if manifest else ""
            )
        if self.cloud_edit_path:
            self.cloud_label.setText(Path(self.cloud_edit_path).name)
            self.cloud_label.setToolTip(self.cloud_edit_path)
        self._update_source_meta()
        self._update_project_labels()


def run_cloudet_qt(project_dir: str, cloud_path: str | None = None) -> None:
    """Launch the cloudet Qt application."""
    install_qt_message_filter()
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("cloudet")
    app.setApplicationDisplayName("cloudet")
    icon = app_icon()
    app.setWindowIcon(icon)
    win = CloudetAppWindow(project_dir, cloud_path=cloud_path)
    win.setWindowIcon(icon)
    win.show()
    app.exec()
