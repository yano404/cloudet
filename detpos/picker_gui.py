"""Interactive plane picker GUI (thin Open3D shell over detpos core).

Usage: ``detpos pick <project_dir> [--pcd <cloud file>]``

All state lives in the project directory (settings.json, groups/,
manifest.json). On every save the current fit quality (main plane
component extraction) is computed and shown, so bad picks are caught
immediately instead of at the batch-fit stage.

Key bindings:
    Alt+Click  add group (or append to active group in append mode)
    Tab        cycle active group
    M          toggle append mode
    V          toggle solo view (active group only)
    S          save all groups (+ fit + manifest)
    L          load all groups from the project
    Del        delete active group
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from detpos.groups import load_groups
from detpos.mainplane import MainPlaneParams, extract_main_plane
from detpos.picking import pick_plane_region
from detpos.project import (
    PickerSettings,
    SourceInfo,
    load_group_indices,
    load_settings,
    read_manifest,
    save_group,
    save_settings,
    write_manifest,
)


def default_group_color(gid: int) -> np.ndarray:
    table = [
        (0.90, 0.25, 0.25), (0.25, 0.55, 0.95), (0.20, 0.75, 0.35),
        (0.95, 0.75, 0.20), (0.75, 0.35, 0.85), (0.20, 0.80, 0.80),
        (0.95, 0.45, 0.15), (0.60, 0.60, 0.60),
    ]
    return np.asarray(table[gid % len(table)], dtype=np.float64)


class PlanePickerApp:
    LEFT_PANEL_WIDTH = 400

    def __init__(self, project_dir: str, pcd_path: str | None = None):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.settings = load_settings(self.project_dir, warn=self._warn_early)

        self.pcd_full = o3d.geometry.PointCloud()
        self.full_points = np.zeros((0, 3))
        self.kdtree = None
        self.pcd_path = ""

        self.groups: list[dict] = []
        self.active_group_id: int | None = None
        self.next_group_id = 0
        self.append_mode = False
        self.solo_mode = False
        self._group_labels: dict[int, object] = {}
        self._cards: list[dict] = []
        self._updating_ui = False
        self._early_warnings: list[str] = getattr(self, "_early_warnings", [])

        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window(
            f"detpos picker - {self.project_dir.name}", 1760, 980
        )
        self.window.set_on_layout(self._on_layout)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])
        self.window.add_child(self.widget3d)

        self.info = gui.Label("")
        self.window.add_child(self.info)

        self._build_panel()

        self.widget3d.set_on_mouse(self._on_mouse)
        self.widget3d.set_on_key(self._on_key)
        self.window.set_focus_widget(self.widget3d)

        if pcd_path:
            self.pcd_path_edit.text_value = pcd_path
            self._safe(self._load_pcd)
        else:
            manifest = read_manifest(self.project_dir)
            if manifest and manifest.get("source", {}).get("path"):
                self.pcd_path_edit.text_value = manifest["source"]["path"]

        for w in self._early_warnings:
            self.info.text = w
        self._update_info()

    def _warn_early(self, msg):
        if not hasattr(self, "_early_warnings"):
            self._early_warnings = []
        self._early_warnings.append(str(msg))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:  # shown in the info bar, never crashes the GUI
            self.info.text = str(e)
            print(e)

    def _num_edit(self, parent, label, value):
        parent.add_child(gui.Label(label))
        if isinstance(value, int):
            w = gui.NumberEdit(gui.NumberEdit.INT)
            w.int_value = int(value)
        else:
            w = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            w.double_value = float(value)
        parent.add_child(w)
        return w

    def _build_panel(self):
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.tabs = gui.TabControl()

        # ---- Settings tab ----
        st = gui.ScrollableVert(margin, gui.Margins(margin, margin, margin, margin))

        pc = gui.CollapsableVert("Point Cloud", 4, gui.Margins(8, 8, 8, 8))
        pc.add_child(gui.Label("cloud file"))
        self.pcd_path_edit = gui.TextEdit()
        pc.add_child(self.pcd_path_edit)
        b = gui.Button("Browse...")
        b.set_on_clicked(self._browse_pcd)
        pc.add_child(b)
        b = gui.Button("Load Cloud")
        b.set_on_clicked(lambda: self._safe(self._load_pcd))
        pc.add_child(b)
        st.add_child(pc)
        st.add_fixed(8)

        det = gui.CollapsableVert("Detection (mm)", 4, gui.Margins(8, 8, 8, 8))
        d = self.settings.detection
        self.e_radius = self._num_edit(det, "local_radius_mm", d.local_radius_mm)
        self.e_locthr = self._num_edit(
            det, "local_distance_threshold_mm", d.local_distance_threshold_mm
        )
        self.e_lociter = self._num_edit(
            det, "local_ransac_iterations", d.local_ransac_iterations
        )
        self.e_minnb = self._num_edit(det, "min_neighbor_points", d.min_neighbor_points)
        self.e_minin = self._num_edit(det, "min_local_inliers", d.min_local_inliers)
        self.e_accthr = self._num_edit(
            det, "accumulate_threshold_mm", d.accumulate_threshold_mm
        )
        self.e_connect = gui.Checkbox("connect (component under click only)")
        self.e_connect.checked = d.connect
        det.add_child(self.e_connect)
        self.e_cell = self._num_edit(det, "cell_size_mm", d.cell_size_mm)
        st.add_child(det)
        st.add_fixed(8)

        vw = gui.CollapsableVert("View", 4, gui.Margins(8, 8, 8, 8))
        v = self.settings.view
        self.e_basesz = self._num_edit(vw, "base_point_size", v.base_point_size)
        self.e_actsz = self._num_edit(vw, "active_point_size", v.active_point_size)
        self.e_inactsz = self._num_edit(vw, "inactive_point_size", v.inactive_point_size)
        self.e_voxel = self._num_edit(
            vw, "display_voxel_size_mm (0=off)", v.display_voxel_size_mm
        )
        self.e_maxdisp = self._num_edit(
            vw, "display_max_points", int(v.display_max_points)
        )
        self.e_axsize = self._num_edit(vw, "axis_size_mm", v.axis_size_mm)
        vw.set_is_open(False)
        st.add_child(vw)
        st.add_fixed(8)

        row = gui.Horiz(4)
        b = gui.Button("Apply")
        b.set_on_clicked(lambda: self._safe(self._apply_settings))
        row.add_child(b)
        b = gui.Button("Save Settings")
        b.set_on_clicked(lambda: self._safe(self._save_settings))
        row.add_child(b)
        st.add_child(row)

        # ---- Groups tab ----
        gt = gui.ScrollableVert(margin, gui.Margins(margin, margin, margin, margin))
        self.groups_header = gui.Label("Groups (0)")
        gt.add_child(self.groups_header)
        self.cards_container = gui.Vert(4)
        gt.add_child(self.cards_container)
        gt.add_fixed(8)

        act = gui.CollapsableVert("Actions", 4, gui.Margins(8, 8, 8, 8))
        grid = gui.VGrid(2, 8, gui.Margins(4, 4, 4, 4))
        for text, fn in [
            ("Fit Active", lambda: self._safe(self._fit_active)),
            ("Rename", self._rename_popup),
            ("Save All (S)", lambda: self._safe(self._save_all)),
            ("Load All (L)", lambda: self._safe(self._load_all)),
            ("Delete (Del)", self._delete_active),
            ("Clear All", self._clear_all),
        ]:
            b = gui.Button(text)
            b.set_on_clicked(fn)
            grid.add_child(b)
        act.add_child(grid)
        gt.add_child(act)

        self.tabs.add_tab("Settings", st)
        self.tabs.add_tab("Groups", gt)
        self.window.add_child(self.tabs)

    def _on_layout(self, ctx):
        r = self.window.content_rect
        w = self.LEFT_PANEL_WIDTH
        info_h = self.info.calc_preferred_size(ctx, gui.Widget.Constraints()).height
        self.tabs.frame = gui.Rect(r.x, r.y, w, r.height)
        self.widget3d.frame = gui.Rect(r.x + w, r.y, r.width - w, r.height - info_h)
        self.info.frame = gui.Rect(r.x + w, r.y + r.height - info_h, r.width - w, info_h)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _apply_settings(self):
        from detpos.picking import PickParams
        from detpos.project import ViewSettings

        self.settings.detection = PickParams(
            local_radius_mm=self.e_radius.double_value,
            local_distance_threshold_mm=self.e_locthr.double_value,
            local_ransac_iterations=self.e_lociter.int_value,
            min_neighbor_points=self.e_minnb.int_value,
            min_local_inliers=self.e_minin.int_value,
            accumulate_threshold_mm=self.e_accthr.double_value,
            connect=self.e_connect.checked,
            cell_size_mm=self.e_cell.double_value,
        )
        self.settings.view = ViewSettings(
            base_point_size=self.e_basesz.double_value,
            active_point_size=self.e_actsz.double_value,
            inactive_point_size=self.e_inactsz.double_value,
            display_voxel_size_mm=self.e_voxel.double_value,
            display_max_points=max(100_000, self.e_maxdisp.int_value),
            axis_size_mm=self.e_axsize.double_value,
        )
        self._refresh_base_cloud()
        self._refresh_all_groups()
        self.info.text = "Settings applied"

    def _save_settings(self):
        self._apply_settings()
        path = save_settings(self.project_dir, self.settings)
        self.info.text = f"Saved settings to {path}"

    # ------------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------------

    def _browse_pcd(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select point cloud", self.window.theme)
        dlg.add_filter(".ply .pcd .xyz .pts", "Point clouds")
        dlg.add_filter("", "All files")
        dlg.set_on_cancel(self.window.close_dialog)

        def done(path):
            self.pcd_path_edit.text_value = path
            self.window.close_dialog()

        dlg.set_on_done(done)
        self.window.show_dialog(dlg)

    def _load_pcd(self):
        path = self.pcd_path_edit.text_value.strip()
        if not path:
            raise ValueError("no cloud file specified")
        if not os.path.exists(path):
            raise FileNotFoundError(f"cloud file not found: {path}")
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise ValueError(f"empty or unreadable cloud: {path}")
        self._clear_all()
        self.pcd_full = pcd
        self.full_points = np.asarray(pcd.points)
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)
        self.pcd_path = path
        self._refresh_base_cloud()
        self.info.text = (
            f"Loaded {len(self.full_points):,} points from {os.path.basename(path)} "
            f"(displaying {getattr(self, '_n_displayed', 0):,})"
        )
        self._update_info()

    def _mat(self, point_size):
        m = rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = float(point_size) * self.window.scaling
        return m

    def _remove_geom(self, name):
        try:
            self.widget3d.scene.remove_geometry(name)
        except Exception:
            pass

    def _for_display(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Downsample for rendering only (never used for pick/fit/save).

        Optional voxel filter, then a hard random-subsample cap: large
        geometries crash the renderer (bus error) if sent unfiltered.
        """
        vox = self.settings.view.display_voxel_size_mm
        if vox > 0:
            cloud = cloud.voxel_down_sample(vox)
        cap = int(self.settings.view.display_max_points)
        n = len(cloud.points)
        if n > cap:
            rng = np.random.default_rng(0)
            keep = rng.choice(n, size=cap, replace=False)
            cloud = cloud.select_by_index(np.sort(keep).tolist())
        return cloud

    def _refresh_base_cloud(self):
        self._remove_geom("base")
        self._remove_geom("axis")
        if len(self.full_points) == 0:
            return
        base = self._for_display(self.pcd_full)
        self._n_displayed = len(base.points)
        base.paint_uniform_color([0.88, 0.88, 0.88])
        self.widget3d.scene.add_geometry(
            "base", base, self._mat(self.settings.view.base_point_size)
        )
        bbox = base.get_axis_aligned_bounding_box()
        self.widget3d.setup_camera(60.0, bbox, bbox.get_center())
        if self.settings.view.axis_size_mm > 0:
            mn = bbox.min_bound
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.settings.view.axis_size_mm,
                origin=[float(x) + self.settings.view.axis_margin_mm for x in mn],
            )
            m = rendering.MaterialRecord()
            m.shader = "defaultLit"
            self.widget3d.scene.add_geometry("axis", axis, m)

    # ------------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------------

    def _get_group(self, gid):
        for g in self.groups:
            if g["id"] == gid:
                return g
        return None

    def _add_group_from_click(self, world):
        if self.kdtree is None:
            raise ValueError("no point cloud loaded")
        _, idx, _ = self.kdtree.search_radius_vector_3d(
            world, self.settings.detection.local_radius_mm
        )
        indices, plane = pick_plane_region(
            self.full_points, world, np.asarray(idx), self.settings.detection
        )

        if self.append_mode and self.active_group_id is not None:
            g = self._get_group(self.active_group_id)
            before = len(g["indices"])
            g["indices"] = np.union1d(g["indices"], indices)
            g["clicked"] = np.asarray(world)
            g["coarse_plane"] = plane.as_array()
            g["fit"] = None
            self.info.text = (
                f"Appended {len(g['indices']) - before:,} points to {g['name']} "
                f"(total {len(g['indices']):,})"
            )
        else:
            gid = self.next_group_id
            self.next_group_id += 1
            g = {
                "id": gid,
                "name": f"G{gid}",
                "visible": True,
                "color": default_group_color(gid),
                "clicked": np.asarray(world),
                "coarse_plane": plane.as_array(),
                "indices": np.asarray(indices, dtype=np.int64),
                "fit": None,
            }
            self.groups.append(g)
            self.active_group_id = gid
            self.info.text = f"Added {g['name']} with {len(indices):,} points"
        self._refresh_all_groups()
        self._update_info()

    def _fit_active(self):
        g = self._get_group(self.active_group_id) if self.active_group_id is not None else None
        if g is None:
            raise ValueError("no active group")
        self._fit_group(g)
        f = g["fit"]
        self.info.text = (
            f"{g['name']}: status={f['status']} mad_sigma={f['mad_sigma_mm']*1e3:.0f} um "
            f"main {f['n_main']:,}/{len(g['indices']):,}  {f.get('reasons') or ''}"
        )
        self._refresh_all_groups()

    def _fit_group(self, g):
        pts = self.full_points[g["indices"]]
        res = extract_main_plane(
            pts, params=MainPlaneParams(), clicked=g["clicked"],
            coarse_plane=g["coarse_plane"],
        )
        g["fit"] = {
            "status": res.status,
            "reasons": res.reasons,
            "plane_abcd": res.plane.as_array().tolist(),
            "mad_sigma_mm": res.fit.stats_inliers["mad_sigma"],
            "n_main": res.n_main,
            "diagnostics": res.diagnostics,
        }

    def _save_all(self):
        if not self.groups:
            raise ValueError("no groups to save")
        for g in self.groups:
            if g["fit"] is None:
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
        worst = max(self.groups, key=lambda g: g["fit"]["mad_sigma_mm"])
        self.info.text = (
            f"Saved {len(self.groups)} groups to {self.project_dir / 'groups'} | "
            f"worst: {worst['name']} {worst['fit']['status']} "
            f"{worst['fit']['mad_sigma_mm']*1e3:.0f} um"
        )
        self._refresh_all_groups()

    def _load_all(self):
        if len(self.full_points) == 0:
            raise ValueError("load the source cloud first")
        manifest = read_manifest(self.project_dir)
        if manifest is not None:
            src_n = manifest.get("source", {}).get("n_points")
            if src_n is not None and src_n != len(self.full_points):
                raise ValueError(
                    f"source cloud mismatch: saved n_points={src_n}, "
                    f"current={len(self.full_points)} "
                    f"(saved source: {manifest['source'].get('path')})"
                )
        infos = load_groups(self.project_dir)
        self._clear_all()
        for info in infos:
            indices = load_group_indices(self.project_dir, info.group_id)
            if indices is None:
                self.info.text = f"{info.name}: no indices file, skipped"
                continue
            if indices.max(initial=-1) >= len(self.full_points):
                raise ValueError(
                    f"{info.name}: indices exceed the current cloud "
                    "(wrong source cloud?)"
                )
            self.groups.append({
                "id": info.group_id,
                "name": info.name,
                "visible": True,
                "color": default_group_color(info.group_id),
                "clicked": info.clicked,
                "coarse_plane": info.coarse_plane,
                "indices": indices,
                "fit": None,
            })
        if self.groups:
            ids = sorted(g["id"] for g in self.groups)
            self.active_group_id = ids[0]
            self.next_group_id = ids[-1] + 1
        self.info.text = f"Loaded {len(self.groups)} groups"
        self._refresh_all_groups()
        self._update_info()

    def _delete_active(self):
        gid = self.active_group_id
        if gid is None:
            self.info.text = "no active group"
            return
        self.groups = [g for g in self.groups if g["id"] != gid]
        self._remove_geom(f"group_{gid:03d}")
        if gid in self._group_labels:
            try:
                self.widget3d.remove_3d_label(self._group_labels.pop(gid))
            except Exception:
                pass
        self.active_group_id = (
            sorted(g["id"] for g in self.groups)[0] if self.groups else None
        )
        self._refresh_all_groups()
        self._update_info()

    def _clear_all(self):
        for g in self.groups:
            self._remove_geom(f"group_{g['id']:03d}")
        for lbl in self._group_labels.values():
            try:
                self.widget3d.remove_3d_label(lbl)
            except Exception:
                pass
        self._group_labels = {}
        self.groups = []
        self.active_group_id = None
        self.next_group_id = 0
        self._refresh_groups_panel()
        self._update_info()

    def _rename_popup(self):
        g = self._get_group(self.active_group_id) if self.active_group_id is not None else None
        if g is None:
            self.info.text = "no active group"
            return
        em = self.window.theme.font_size
        dlg = gui.Dialog("Rename")
        layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        layout.add_child(gui.Label(f"Rename {g['name']}"))
        edit = gui.TextEdit()
        edit.text_value = g["name"]
        layout.add_child(edit)
        row = gui.Horiz(0.5 * em)

        def ok():
            if edit.text_value.strip():
                g["name"] = edit.text_value.strip()
            self.window.close_dialog()
            self._refresh_all_groups()
            self._update_info()

        b = gui.Button("OK")
        b.set_on_clicked(ok)
        row.add_child(b)
        b = gui.Button("Cancel")
        b.set_on_clicked(self.window.close_dialog)
        row.add_child(b)
        layout.add_child(row)
        dlg.add_child(layout)
        self.window.show_dialog(dlg)

    # ------------------------------------------------------------------
    # Rendering of groups / cards
    # ------------------------------------------------------------------

    def _refresh_group_geometry(self, g):
        gid = g["id"]
        name = f"group_{gid:03d}"
        self._remove_geom(name)
        if gid in self._group_labels:
            try:
                self.widget3d.remove_3d_label(self._group_labels.pop(gid))
            except Exception:
                pass
        if not g["visible"]:
            return
        if self.solo_mode and gid != self.active_group_id:
            return
        cloud = self.pcd_full.select_by_index(g["indices"].tolist())
        cloud = self._for_display(cloud)
        active = gid == self.active_group_id
        color = g["color"]
        if active:
            color = np.clip(0.6 * color + 0.4 * np.array([1.0, 1.0, 0.0]), 0, 1)
        cloud.paint_uniform_color(color)
        size = (
            self.settings.view.active_point_size
            if active else self.settings.view.inactive_point_size
        )
        self.widget3d.scene.add_geometry(name, cloud, self._mat(size))
        pts = np.asarray(cloud.points)
        if len(pts):
            self._group_labels[gid] = self.widget3d.add_3d_label(
                pts.mean(axis=0), g["name"]
            )

    def _refresh_all_groups(self):
        for g in self.groups:
            self._refresh_group_geometry(g)
        self._refresh_groups_panel()

    def _make_card(self):
        card = {}
        root = gui.Vert(1)
        root.visible = False
        row = gui.Horiz(3)
        wrap = gui.Vert(0, gui.Margins(12, 16, 0, 0))
        cb = gui.Checkbox("")
        wrap.add_child(cb)
        colorb = gui.Button(" ")
        title = gui.Button("group")
        row.add_child(wrap)
        row.add_child(colorb)
        row.add_child(title)
        root.add_child(row)
        self.cards_container.add_child(root)
        card.update(root=root, cb=cb, colorb=colorb, title=title, gid=None)
        return card

    def _refresh_groups_panel(self):
        self._updating_ui = True
        gs = sorted(self.groups, key=lambda x: x["id"])
        self.groups_header.text = f"Groups ({len(gs)})"
        while len(self._cards) < len(gs):
            self._cards.append(self._make_card())
        for card, g in zip(self._cards, gs):
            gid = g["id"]
            card["gid"] = gid
            card["root"].visible = True
            card["cb"].checked = g["visible"]
            label = f"{g['name']} ({len(g['indices']):,})"
            if g["fit"] is not None:
                label += f" [{g['fit']['status']} {g['fit']['mad_sigma_mm']*1e3:.0f}um]"
            if gid == self.active_group_id:
                label = "* " + label
            card["title"].text = label
            c = g["color"]
            card["colorb"].background_color = gui.Color(float(c[0]), float(c[1]), float(c[2]), 1.0)
            card["cb"].set_on_checked(
                lambda checked, gid=gid: self._set_visible(gid, checked)
            )
            card["title"].set_on_clicked(lambda gid=gid: self._select(gid))
        for card in self._cards[len(gs):]:
            card["gid"] = None
            card["root"].visible = False
        self._updating_ui = False
        self.window.set_needs_layout()

    def _set_visible(self, gid, visible):
        if self._updating_ui:
            return
        g = self._get_group(gid)
        if g is not None:
            g["visible"] = bool(visible)
            self._refresh_group_geometry(g)
            self._refresh_groups_panel()

    def _select(self, gid):
        self.active_group_id = gid
        self._refresh_all_groups()
        self._update_info()

    # ------------------------------------------------------------------
    # Info / input
    # ------------------------------------------------------------------

    def _update_info(self):
        lines = [
            f"Project: {self.project_dir}",
            f"Cloud: {os.path.basename(self.pcd_path) if self.pcd_path else '(none)'} "
            f"({len(self.full_points):,} pts)",
            f"Groups: {len(self.groups)}  Mode: {'APPEND' if self.append_mode else 'NEW'}  "
            f"View: {'SOLO' if self.solo_mode else 'ALL'}",
        ]
        g = self._get_group(self.active_group_id) if self.active_group_id is not None else None
        if g is not None:
            line = f"Active: {g['name']} ({len(g['indices']):,} pts)"
            if g["fit"] is not None:
                line += (
                    f"  fit: {g['fit']['status']} "
                    f"mad_sigma={g['fit']['mad_sigma_mm']*1e3:.0f}um"
                )
            lines.append(line)
        self.info.text = "\n".join(lines)
        self.window.set_needs_layout()

    def _on_mouse(self, event):
        if (
            event.type == gui.MouseEvent.Type.BUTTON_DOWN
            and event.is_button_down(gui.MouseButton.LEFT)
            and event.is_modifier_down(gui.KeyModifier.ALT)
        ):
            def depth_cb(depth_image):
                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                if not (0 <= x < self.widget3d.frame.width
                        and 0 <= y < self.widget3d.frame.height):
                    return
                depth = np.asarray(depth_image)[y, x]
                if depth == 1.0:
                    return
                world = self.widget3d.scene.camera.unproject(
                    x, y, depth, self.widget3d.frame.width, self.widget3d.frame.height
                )
                world = np.asarray(world, dtype=np.float64)
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self._safe(lambda: self._add_group_from_click(world))
                )

            self.widget3d.scene.scene.render_to_depth_image(depth_cb)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.key == gui.KeyName.TAB:
            ids = sorted(g["id"] for g in self.groups)
            if ids:
                if self.active_group_id not in ids:
                    self._select(ids[0])
                else:
                    self._select(ids[(ids.index(self.active_group_id) + 1) % len(ids)])
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.M:
            self.append_mode = not self.append_mode
            self._update_info()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.V:
            self.solo_mode = not self.solo_mode
            self._refresh_all_groups()
            self._update_info()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.S:
            self._safe(self._save_all)
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.L:
            self._safe(self._load_all)
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.F:
            self._safe(self._fit_active)
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key in (gui.KeyName.BACKSPACE, gui.KeyName.DELETE):
            self._delete_active()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def run(self):
        gui.Application.instance.run()


def run_picker(project_dir: str, pcd_path: str | None = None) -> None:
    PlanePickerApp(project_dir, pcd_path).run()
