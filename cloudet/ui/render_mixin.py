"""3D view actor rendering (base cloud, groups, plane bbox, frame overlay)."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from cloudet.core.neighbors import display_xyz
from cloudet.core.plane import Plane
from cloudet.project.schema import plane_from_json
from cloudet.ui.widgets import _line_tube_mesh


class RenderMixin:
    """PyVista actor refresh for base cloud, groups, and active plane bbox."""

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
        self.plotter.remove_actor("active_cylinder_overlay", render=False)
        actors = getattr(self.plotter, "actors", None) or {}
        for name in list(actors.keys()):
            if str(name).startswith("active_circle_overlay"):
                self.plotter.remove_actor(name, render=False)

    def _refresh_active_plane_bbox(self, *, render: bool = True) -> None:
        """Draw wire overlay for the active fitted plane / cylinder / circle."""
        self._clear_active_plane_bbox()
        g = self._active_group()
        if g is None or self.full_points.size == 0 or len(g.get("indices", [])) == 0:
            if render:
                self.plotter.render()
            return

        fit = g.get("fit") or {}
        cylinders = fit.get("cylinders") if isinstance(fit.get("cylinders"), list) else []
        circles = fit.get("circles") if isinstance(fit.get("circles"), list) else []

        if cylinders:
            from cloudet.project.schema import cylinder_from_json

            try:
                cyl = cylinder_from_json(cylinders[0])
            except (KeyError, TypeError, ValueError):
                cyl = None
            if cyl is not None:
                pts = self.full_points[g["indices"]]
                # Segment length from projected extent along axis.
                t = (pts - cyl.point) @ cyl.direction
                t0, t1 = float(np.min(t)), float(np.max(t))
                if t1 - t0 < 1.0:
                    t0, t1 = -50.0, 50.0
                p0 = cyl.point + t0 * cyl.direction
                p1 = cyl.point + t1 * cyl.direction
                mesh = _line_tube_mesh(p0, p1, max(cyl.diameter_mm, 1.0))
                if self._view_frame is not None:
                    mesh.points = self._to_view_points(
                        np.asarray(mesh.points, dtype=np.float64)
                    )
                self.plotter.add_mesh(
                    mesh,
                    name="active_cylinder_overlay",
                    style="wireframe",
                    color="#c45c26",
                    line_width=1.5,
                    opacity=0.55,
                    reset_camera=False,
                    pickable=False,
                )
                if render:
                    self.plotter.render()
                return

        if circles:
            from cloudet.project.schema import circle_from_json

            active_ci = int(getattr(self, "_active_circle_index", 0))
            for entry in circles:
                try:
                    cir = circle_from_json(entry)
                except (KeyError, TypeError, ValueError):
                    continue
                ci = int(entry.get("circle_index", 0))
                n = cir.normal / max(float(np.linalg.norm(cir.normal)), 1e-12)
                tmp = (
                    np.array([1.0, 0.0, 0.0])
                    if abs(n[0]) < 0.9
                    else np.array([0.0, 1.0, 0.0])
                )
                u = np.cross(n, tmp)
                u /= np.linalg.norm(u)
                v = np.cross(n, u)
                theta = np.linspace(0.0, 2.0 * np.pi, 96 if ci == active_ci else 64)
                r = 0.5 * float(cir.diameter_mm)
                ring = cir.center + r * (
                    np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
                )
                if self._view_frame is not None:
                    ring = self._to_view_points(ring)
                poly = pv.lines_from_points(ring, close=True)
                is_active = ci == active_ci
                # Active: tube (VTK ignores line_width on polylines). Inactive: thin ring.
                if is_active:
                    stroke_mm = max(1.5, 0.025 * float(cir.diameter_mm))
                    mesh = poly.tube(radius=0.5 * stroke_mm, n_sides=10)
                    self.plotter.add_mesh(
                        mesh,
                        name=f"active_circle_overlay_{ci}",
                        color="#c45c26",
                        opacity=0.9,
                        reset_camera=False,
                        pickable=False,
                    )
                else:
                    self.plotter.add_mesh(
                        poly,
                        name=f"active_circle_overlay_{ci}",
                        color="#6a8ec0",
                        line_width=1.5,
                        opacity=0.55,
                        reset_camera=False,
                        pickable=False,
                    )

        p = self._active_plane_entry()
        if p is None:
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
        n_ax = np.asarray(plane_from_json(p).normal, dtype=np.float64)
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

