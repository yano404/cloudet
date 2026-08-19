"""Residual u–v map helpers used by the Qt residual QC dock.

Bins signed plane residuals onto an in-plane (u, v) grid aligned to the
face's minimum-area bounding rectangle.
"""

from __future__ import annotations

import numpy as np

from cloudet.array_backend import get_context
from cloudet.plane import Plane

__all__ = ["residual_uv_map"]


def _seed_inplane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Any orthonormal in-plane frame; orientation is arbitrary."""
    n = np.asarray(normal, dtype=np.float64)
    a = np.array([1.0, 0.0, 0.0])
    if abs(n @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _convex_hull_2d(xy: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. Returns CCW hull vertices (N, 2)."""
    pts = np.asarray(xy, dtype=np.float64)
    if len(pts) == 0:
        return pts.reshape(0, 2)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    # Drop exact duplicates that break the chain.
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
    pts = pts[keep]
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _min_area_rect_rotation(xy: np.ndarray) -> np.ndarray:
    """2×2 matrix R with ``xy @ R.T`` axis-aligned to a min-area bounding rect.

    Uses convex-hull edges (rotating-calipers style). Falls back to identity
    when the cloud is too small or degenerate.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 3:
        return np.eye(2)
    # Hull of a capped subsample is enough for orientation and much faster
    # on multi-million-point faces.
    work = xy
    if len(work) > 80_000:
        rng = np.random.default_rng(0)
        work = work[rng.choice(len(work), 80_000, replace=False)]
    hull = _convex_hull_2d(work)
    if len(hull) < 2:
        return np.eye(2)

    best_area = np.inf
    best_R = np.eye(2)
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        elen = float(np.linalg.norm(edge))
        if elen < 1e-12:
            continue
        c = edge[0] / elen
        s = edge[1] / elen
        # Rotate by -atan2 so this hull edge lies on +u.
        R = np.array([[c, s], [-s, c]], dtype=np.float64)
        rot = hull @ R.T
        area = float(
            (rot[:, 0].max() - rot[:, 0].min())
            * (rot[:, 1].max() - rot[:, 1].min())
        )
        if area < best_area:
            best_area = area
            best_R = R
    return best_R


def _aligned_inplane_basis(
    pts: np.ndarray,
    normal: np.ndarray,
    *,
    return_coords: bool = False,
):
    """In-plane basis aligned to the face's minimum-area bounding rectangle.

    Returns ``(u, v, center)``. If ``return_coords`` is True, also returns
    ``(uu, vv)`` — the already-projected in-plane coordinates, avoiding a
    second full ``(pts-center)@axis`` pass in callers.
    """
    pts = np.asarray(pts, dtype=np.float64)
    center = pts.mean(axis=0)
    u0, v0 = _seed_inplane_basis(normal)
    if len(pts) < 3:
        if return_coords:
            zeros = np.zeros(len(pts), dtype=np.float64)
            return u0, v0, center, zeros, zeros
        return u0, v0, center

    xy = np.column_stack([(pts - center) @ u0, (pts - center) @ v0])
    xy -= xy.mean(axis=0)
    R = _min_area_rect_rotation(xy)
    if np.linalg.det(R) < 0:
        R = R.copy()
        R[:, 1] *= -1
    uv = xy @ R.T
    # Put the longer side on u, with deterministic signs.
    if (uv[:, 0].max() - uv[:, 0].min()) < (uv[:, 1].max() - uv[:, 1].min()):
        # Swap axes: [[0,1],[-1,0]] @ R, applied as row: xy @ (S @ R).T
        S = np.array([[0.0, 1.0], [-1.0, 0.0]])
        R = S @ R
        uv = xy @ R.T
    if abs(float(uv[:, 0].min())) > abs(float(uv[:, 0].max())):
        R = R.copy()
        R[0, :] *= -1
        uv[:, 0] *= -1
    if abs(float(uv[:, 1].min())) > abs(float(uv[:, 1].max())):
        R = R.copy()
        R[1, :] *= -1
        uv[:, 1] *= -1
    if np.linalg.det(R) < 0:
        R = R.copy()
        R[1, :] *= -1
        uv[:, 1] *= -1

    # R maps seed-frame coords → aligned coords: [u_seed, v_seed] @ R.T
    # so aligned axes expressed in 3D are rows of R in the (u0, v0) basis.
    u = R[0, 0] * u0 + R[0, 1] * v0
    v = R[1, 0] * u0 + R[1, 1] * v0
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    if np.dot(np.cross(u, v), normal) < 0:
        v = -v
        uv[:, 1] *= -1
    if return_coords:
        return u, v, center, uv[:, 0], uv[:, 1]
    return u, v, center


def _align_inplane_frame(
    u_src: np.ndarray,
    v_src: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a source (u, v) frame onto ``normal``, keeping the same sense.

    Chooses 0° vs 180° about the normal so ``u``/``v`` stay closest to the
    source axes. Does not swap axes (a subset min-rect can do that).
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    u_src = np.asarray(u_src, dtype=np.float64)
    v_src = np.asarray(v_src, dtype=np.float64)
    u = u_src - (u_src @ n) * n
    un = float(np.linalg.norm(u))
    if un < 1e-12:
        return _seed_inplane_basis(n)
    u = u / un
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    if (u @ u_src) + (v @ v_src) < 0.0:
        u = -u
        v = -v
    return u, v


def residual_uv_map(
    points: np.ndarray,
    plane: Plane,
    mask: np.ndarray | None = None,
    bins: int = 200,
    *,
    return_points: bool = False,
    compute_backend: str = "auto",
    u_axis: np.ndarray | None = None,
    v_axis: np.ndarray | None = None,
    center: np.ndarray | None = None,
) -> dict:
    """Bin signed residuals on an in-plane (u, v) grid.

    The in-plane axes follow the face's minimum-area bounding rectangle so
    a rectangular patch appears axis-aligned in the map (an arbitrary seed
    basis otherwise leaves the face diagonally skewed; PCA alone can still
    tilt when sampling density is uneven).

    Pass ``u_axis`` / ``v_axis`` (and optionally ``center``) to lock the
    frame to a previous fit instead of recomputing the min-rect. Used so a
    selection refit keeps the same u–v orientation as the source plane.

    Returns arrays suitable for plotting: per-bin mean signed residual
    and counts. Reveals spatial systematics (registration steps between
    scan passes, surface waviness, residual tilt) that histograms hide.

    Always includes ``r`` (signed residuals), ``u_axis`` / ``v_axis`` /
    ``center``, and ``extents_uvn`` ``(lo, hi)`` in the (u, v, n) frame.
    If ``return_points`` is True, also include per-point ``u`` and ``v``.
    """
    pts = np.asarray(points if mask is None else points[mask], dtype=np.float64)
    ctx = get_context(compute_backend, n_points=len(pts))
    if ctx.name == "cupy":
        xp = ctx.xp
        pts_dev = ctx.to_device(pts)
        normal_dev = ctx.to_device(plane.normal)
        r_g = pts_dev @ normal_dev + plane.d
    else:
        r_g = None
        r = plane.signed_distances(pts)

    if u_axis is not None and v_axis is not None:
        u, v = _align_inplane_frame(u_axis, v_axis, plane.normal)
        if center is None:
            center = pts.mean(axis=0)
        else:
            center = np.asarray(center, dtype=np.float64)
        uu = (pts - center) @ u
        vv = (pts - center) @ v
    else:
        u, v, center, uu, vv = _aligned_inplane_basis(
            pts, plane.normal, return_coords=True
        )
    n_ax = np.asarray(plane.normal, dtype=np.float64)
    nn = (pts - center) @ n_ax

    if ctx.name == "cupy":
        xp = ctx.xp
        uu_g = xp.asarray(uu)
        vv_g = xp.asarray(vv)
        counts, ue, ve = xp.histogram2d(uu_g, vv_g, bins=bins)
        sums, _, _ = xp.histogram2d(uu_g, vv_g, bins=[ue, ve], weights=r_g)
        counts = ctx.asnumpy(counts)
        ue = ctx.asnumpy(ue)
        ve = ctx.asnumpy(ve)
        sums = ctx.asnumpy(sums)
        r = ctx.asnumpy(r_g)
    else:
        counts, ue, ve = np.histogram2d(uu, vv, bins=bins)
        sums, _, _ = np.histogram2d(uu, vv, bins=[ue, ve], weights=r)
    with np.errstate(invalid="ignore"):
        mean_map = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    lo = np.array([uu.min(), vv.min(), nn.min()], dtype=np.float64)
    hi = np.array([uu.max(), vv.max(), nn.max()], dtype=np.float64)

    out = {
        "mean": mean_map,
        "counts": counts,
        "u_edges": ue,
        "v_edges": ve,
        "r": r,
        "u_axis": u,
        "v_axis": v,
        "center": center,
        "extents_uvn": (lo, hi),
        "n_used": int(len(r)),
    }
    if return_points:
        out["u"] = uu
        out["v"] = vv
    return out
