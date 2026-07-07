"""Qt-based interactive plane picker (PySide6 + PyVista/VTK).

Usage: ``detpos pick <project_dir> [--pcd <cloud>]``

Layout: 3D view (pyvistaqt QtInteractor) + left dock with a group/plane
tree and a settings form. Data model: 1 group = N planes; fitting a
group (multi-plane extraction) populates plane children in the tree.

Interaction:
    P            pick a plane region at the mouse position (PyVista picking)
    append mode  checkbox: picks add to the active group instead
    Fit          runs multi-plane extraction, shows per-plane QC
    Save All     writes groups/ + manifest (fits computed if missing)

Picking is done on the *displayed* (decimated) cloud to find the click
position, but region extraction, fitting and saving always use the
full-resolution cloud.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import pyvista as pv
from pyvistaqt import QtInteractor

from detpos.groups import load_groups
from detpos.multiplane import MultiPlaneParams, extract_planes
from detpos.neighbors import VoxelHashGrid, display_indices
from detpos.picking import PickParams, pick_plane_region
from detpos.plyio import read_ply_xyz
from detpos.project import (
    PickerSettings,
    SourceInfo,
    ViewSettings,
    load_group_indices,
    load_settings,
    read_manifest,
    save_group,
    save_settings,
    write_manifest,
)

GROUP_COLORS = [
    (0.90, 0.25, 0.25), (0.25, 0.55, 0.95), (0.20, 0.75, 0.35),
    (0.95, 0.75, 0.20), (0.75, 0.35, 0.85), (0.20, 0.80, 0.80),
    (0.95, 0.45, 0.15), (0.60, 0.60, 0.60),
]


def group_color(gid: int) -> np.ndarray:
    return np.asarray(GROUP_COLORS[gid % len(GROUP_COLORS)], dtype=np.float64)


class PickerWindow(QMainWindow):
    def __init__(self, project_dir: str, pcd_path: str | None = None):
        super().__init__()
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.settings = load_settings(self.project_dir, warn=self._status)

        self.full_points: np.ndarray = np.zeros((0, 3))
        self.grid: VoxelHashGrid | None = None
        self.pcd_path = ""

        self.groups: list[dict] = []
        self.active_group_id: int | None = None
        self.next_group_id = 0

        self.setWindowTitle(f"detpos picker - {self.project_dir.name}")
        self.resize(1760, 980)

        # --- 3D view -----------------------------------------------------
        central = QWidget()
        vlay = QVBoxLayout(central)
        vlay.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(central)
        vlay.addWidget(self.plotter)
        self.setCentralWidget(central)
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            show_message="P: pick plane region at mouse position",
            use_picker=True,
            show_point=False,
            tolerance=0.01,
        )

        # --- left dock -----------------------------------------------------
        self._build_dock()
        self._build_shortcuts()
        self.statusBar().showMessage("Ready")

        if pcd_path:
            self.pcd_edit_path = pcd_path
        else:
            manifest = read_manifest(self.project_dir)
            self.pcd_edit_path = (
                manifest.get("source", {}).get("path", "") if manifest else ""
            )
        if self.pcd_edit_path:
            self.cloud_label.setText(self.pcd_edit_path)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_dock(self):
        dock = QDockWidget("detpos", self)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        tabs = QTabWidget()

        # ---- Groups tab ----
        gw = QWidget()
        gl = QVBoxLayout(gw)

        self.cloud_label = QLabel("(no cloud)")
        self.cloud_label.setWordWrap(True)
        gl.addWidget(self.cloud_label)
        row = QHBoxLayout()
        b = QPushButton("Browse...")
        b.clicked.connect(self._browse_cloud)
        row.addWidget(b)
        b = QPushButton("Load Cloud")
        b.clicked.connect(lambda: self._guard(self._load_cloud))
        row.addWidget(b)
        gl.addLayout(row)

        self.append_cb = QCheckBox("Append picks to active group")
        gl.addWidget(self.append_cb)
        self.solo_cb = QCheckBox("Show active group only")
        self.solo_cb.toggled.connect(lambda _: self._refresh_group_actors())
        gl.addWidget(self.solo_cb)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["group / plane", "points", "quality"])
        self.tree.setColumnWidth(0, 170)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(self._on_item_selected)
        gl.addWidget(self.tree, stretch=1)

        grid = QHBoxLayout()
        for text, fn in [
            ("Fit", lambda: self._guard(self._fit_active)),
            ("Fit All", lambda: self._guard(self._fit_all)),
            ("Delete", lambda: self._guard(self._delete_active)),
        ]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            grid.addWidget(b)
        gl.addLayout(grid)
        grid = QHBoxLayout()
        for text, fn in [
            ("Save All", lambda: self._guard(self._save_all)),
            ("Load All", lambda: self._guard(self._load_all)),
            ("Clear", lambda: self._guard(self._clear_all)),
        ]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            grid.addWidget(b)
        gl.addLayout(grid)

        tabs.addTab(gw, "Groups")

        # ---- Settings tab ----
        sw = QWidget()
        form = QFormLayout(sw)
        d, v = self.settings.detection, self.settings.view

        def dspin(value, lo=0.0, hi=1e6, step=0.1, dec=3):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setSingleStep(step)
            w.setValue(value)
            return w

        def ispin(value, lo=0, hi=100_000_000):
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(int(value))
            return w

        self.s_radius = dspin(d.local_radius_mm)
        self.s_locthr = dspin(d.local_distance_threshold_mm)
        self.s_lociter = ispin(d.local_ransac_iterations)
        self.s_minnb = ispin(d.min_neighbor_points)
        self.s_minin = ispin(d.min_local_inliers)
        self.s_accthr = dspin(d.accumulate_threshold_mm)
        self.s_connect = QCheckBox()
        self.s_connect.setChecked(d.connect)
        self.s_cell = dspin(d.cell_size_mm)
        form.addRow(QLabel("<b>Detection (mm)</b>"))
        form.addRow("local_radius", self.s_radius)
        form.addRow("local_distance_threshold", self.s_locthr)
        form.addRow("local_ransac_iterations", self.s_lociter)
        form.addRow("min_neighbor_points", self.s_minnb)
        form.addRow("min_local_inliers", self.s_minin)
        form.addRow("accumulate_threshold", self.s_accthr)
        form.addRow("connect (component only)", self.s_connect)
        form.addRow("cell_size", self.s_cell)

        self.s_voxel = dspin(v.display_voxel_size_mm)
        self.s_maxdisp = ispin(v.display_max_points, lo=100_000)
        self.s_ptsize = dspin(v.base_point_size, lo=0.5, hi=20, step=0.5, dec=1)
        form.addRow(QLabel("<b>View</b>"))
        form.addRow("display_voxel_size (mm)", self.s_voxel)
        form.addRow("display_max_points", self.s_maxdisp)
        form.addRow("point_size", self.s_ptsize)

        row = QHBoxLayout()
        b = QPushButton("Apply")
        b.clicked.connect(lambda: self._guard(self._apply_settings))
        row.addWidget(b)
        b = QPushButton("Save Settings")
        b.clicked.connect(lambda: self._guard(self._save_settings))
        row.addWidget(b)
        wrap = QWidget()
        wrap.setLayout(row)
        form.addRow(wrap)

        tabs.addTab(sw, "Settings")
        dock.setWidget(tabs)
        dock.setMinimumWidth(380)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self, lambda: self._guard(self._save_all))
        QShortcut(QKeySequence("F"), self, lambda: self._guard(self._fit_active))
        QShortcut(QKeySequence("Backspace"), self, lambda: self._guard(self._delete_active))
        QShortcut(QKeySequence("M"), self, self.append_cb.toggle)
        QShortcut(QKeySequence("V"), self, self.solo_cb.toggle)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _status(self, msg):
        self.statusBar().showMessage(str(msg)) if hasattr(self, "statusBar") else print(msg)

    def _guard(self, fn):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                fn()
            finally:
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
    # settings
    # ------------------------------------------------------------------

    def _apply_settings(self):
        self.settings.detection = PickParams(
            local_radius_mm=self.s_radius.value(),
            local_distance_threshold_mm=self.s_locthr.value(),
            local_ransac_iterations=self.s_lociter.value(),
            min_neighbor_points=self.s_minnb.value(),
            min_local_inliers=self.s_minin.value(),
            accumulate_threshold_mm=self.s_accthr.value(),
            connect=self.s_connect.isChecked(),
            cell_size_mm=self.s_cell.value(),
        )
        self.settings.view = ViewSettings(
            base_point_size=self.s_ptsize.value(),
            display_voxel_size_mm=self.s_voxel.value(),
            display_max_points=self.s_maxdisp.value(),
        )
        self.grid = None  # radius may have changed -> rebuild lazily
        self._refresh_base_actor()
        self._refresh_group_actors()
        self._status("settings applied")

    def _save_settings(self):
        self._apply_settings()
        path = save_settings(self.project_dir, self.settings)
        self._status(f"saved settings to {path}")

    # ------------------------------------------------------------------
    # cloud
    # ------------------------------------------------------------------

    def _browse_cloud(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select point cloud", "",
            "Point clouds (*.ply);;All files (*)",
        )
        if path:
            self.pcd_edit_path = path
            self.cloud_label.setText(path)

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
        self.grid = None
        self._clear_all()
        self._refresh_base_actor()
        self.plotter.reset_camera()
        self._status(
            f"loaded {len(self.full_points):,} points "
            f"(displaying {self._n_displayed:,})"
        )

    def _ensure_grid(self) -> VoxelHashGrid:
        if self.grid is None:
            if len(self.full_points) == 0:
                raise ValueError("no cloud loaded")
            self._status("building spatial index ...")
            QApplication.processEvents()
            cell = max(self.settings.detection.local_radius_mm, 1.0)
            self.grid = VoxelHashGrid(self.full_points, cell_size=cell)
            self._status("spatial index ready")
        return self.grid

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _refresh_base_actor(self):
        self.plotter.remove_actor("base", render=False)
        self._n_displayed = 0
        if len(self.full_points) == 0:
            return
        v = self.settings.view
        idx = display_indices(
            self.full_points, v.display_voxel_size_mm, int(v.display_max_points)
        )
        self._n_displayed = len(idx)
        self.plotter.add_points(
            self.full_points[idx],
            name="base",
            color=(0.82, 0.82, 0.82),
            point_size=v.base_point_size,
            render_points_as_spheres=False,
        )
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
        idx = display_indices(pts, v.display_voxel_size_mm, int(v.display_max_points))
        active = g["id"] == self.active_group_id
        color = g["color"]
        if active:
            color = np.clip(0.6 * color + 0.4 * np.array([1.0, 1.0, 0.0]), 0, 1)
        self.plotter.add_points(
            pts[idx],
            name=name,
            color=tuple(color),
            point_size=v.active_point_size if active else v.inactive_point_size,
            render_points_as_spheres=False,
        )

    # ------------------------------------------------------------------
    # picking / groups
    # ------------------------------------------------------------------

    def _on_pick(self, picked_point, *_):
        self._guard(lambda: self._handle_pick(np.asarray(picked_point, dtype=np.float64)))

    def _handle_pick(self, world: np.ndarray):
        if len(self.full_points) == 0:
            raise ValueError("load a cloud first")
        grid = self._ensure_grid()
        nb = grid.radius_indices(world, self.settings.detection.local_radius_mm)
        indices, plane = pick_plane_region(
            self.full_points, world, nb, self.settings.detection
        )

        if self.append_cb.isChecked() and self.active_group_id is not None:
            g = self._active_group()
            before = len(g["indices"])
            g["indices"] = np.union1d(g["indices"], indices)
            g["clicked"] = world
            g["coarse_plane"] = plane.as_array()
            g["fit"] = None
            self._status(
                f"appended {len(g['indices']) - before:,} pts to {g['name']} "
                f"(total {len(g['indices']):,})"
            )
        else:
            gid = self.next_group_id
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
            self._status(f"added {g['name']} with {len(indices):,} points")
        self._refresh_group_actors()
        self._refresh_tree()

    def _fit_group(self, g):
        pts = self.full_points[g["indices"]]
        extracted = extract_planes(
            pts, MultiPlaneParams(), clicked=g["clicked"], coarse_plane=g["coarse_plane"]
        )
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
                }
                for p in extracted
            ]
        }

    def _fit_active(self):
        g = self._active_group()
        if g is None:
            raise ValueError("no active group")
        self._status(f"fitting {g['name']} ...")
        QApplication.processEvents()
        self._fit_group(g)
        self._refresh_tree()
        planes = g["fit"]["planes"]
        self._status(
            f"{g['name']}: {len(planes)} plane(s): "
            + " | ".join(
                f"p{p['plane_index']} {p['mad_sigma_mm']*1e3:.0f}um {p['status']}"
                + (" BIMODAL" if p["bimodal"] else "")
                for p in planes
            )
        )

    def _fit_all(self):
        for g in self.groups:
            self._status(f"fitting {g['name']} ...")
            QApplication.processEvents()
            self._fit_group(g)
        self._refresh_tree()
        self._status("fit all done")

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
                "fit": None,
            })
        if self.groups:
            ids = sorted(g["id"] for g in self.groups)
            self.active_group_id = ids[0]
            self.next_group_id = ids[-1] + 1
        self._refresh_group_actors()
        self._refresh_tree()
        self._status(f"loaded {len(self.groups)} groups")

    def _delete_active(self):
        g = self._active_group()
        if g is None:
            raise ValueError("no active group")
        self.plotter.remove_actor(f"group_{g['id']:03d}", render=False)
        self.groups = [x for x in self.groups if x["id"] != g["id"]]
        self.active_group_id = (
            sorted(x["id"] for x in self.groups)[0] if self.groups else None
        )
        self._refresh_group_actors()
        self._refresh_tree()

    def _clear_all(self):
        for g in self.groups:
            self.plotter.remove_actor(f"group_{g['id']:03d}", render=False)
        self.groups = []
        self.active_group_id = None
        self.next_group_id = 0
        self._refresh_group_actors()
        self._refresh_tree()

    # ------------------------------------------------------------------
    # tree
    # ------------------------------------------------------------------

    def _refresh_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        for g in sorted(self.groups, key=lambda x: x["id"]):
            item = QTreeWidgetItem([g["name"], f"{len(g['indices']):,}", ""])
            item.setData(0, Qt.UserRole, ("group", g["id"]))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(0, Qt.Checked if g["visible"] else Qt.Unchecked)
            c = g["color"]
            item.setBackground(
                0, QColor.fromRgbF(float(c[0]), float(c[1]), float(c[2]), 0.35)
            )
            if g["id"] == self.active_group_id:
                f = item.font(0)
                f.setBold(True)
                item.setFont(0, f)
            if g["fit"] is not None:
                for p in g["fit"]["planes"]:
                    label = (
                        f"p{p['plane_index']}  d={p['abcd'][3]:.3f}"
                    )
                    quality = (
                        f"{p['mad_sigma_mm']*1e3:.0f}um {p['status']}"
                        + (" BIMODAL" if p["bimodal"] else "")
                    )
                    child = QTreeWidgetItem([label, f"{p['n_points']:,}", quality])
                    child.setData(0, Qt.UserRole, ("plane", g["id"], p["plane_index"]))
                    child.setFlags(child.flags() & ~Qt.ItemIsUserCheckable)
                    item.addChild(child)
            self.tree.addTopLevelItem(item)
            item.setExpanded(True)
        self.tree.blockSignals(False)

    def _on_item_selected(self, current, _prev):
        if current is None:
            return
        data = current.data(0, Qt.UserRole)
        if data and data[0] in ("group", "plane"):
            gid = data[1]
            if gid != self.active_group_id:
                self.active_group_id = gid
                self._refresh_group_actors()
                self._refresh_tree()

    def _on_item_changed(self, item, column):
        data = item.data(0, Qt.UserRole)
        if not data or data[0] != "group":
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
    app = QApplication.instance() or QApplication([])
    win = PickerWindow(project_dir, pcd_path)
    win.show()
    app.exec()
