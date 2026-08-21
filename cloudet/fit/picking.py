"""Click-driven region extraction (GUI-independent logic).

**Planes:** fit a local plane near the click, progressively expand the
in-plane region, then accumulate the connected component on the face.

**Cylinders:** filter a seed ball to a radial shell about an estimated or
user-seeded axis (``pick_cylinder_region`` /
``pick_cylinder_region_from_cylinder``), with optional shell thickness
from ``PickParams``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from cloudet.core.array_backend import DevicePoints, get_context
from cloudet.core.cylinder import (
    CYLINDER_PICK_MAX_BALL_POINTS,
    distances_to_axis,
)
from cloudet.fit.mainplane import inplane_basis, label_components
from cloudet.core.plane import Plane, fit_plane_lsq, run_ransac

__all__ = [
    "PickParams",
    "pick_plane_region",
    "pick_ball_region",
    "pick_cylinder_region",
    "pick_cylinder_region_from_cylinder",
    "resolve_cylinder_shell_mm",
]

_EXPAND_MAX_CELLS = 160


@dataclass(frozen=True)
class PickParams:
    """Units: mm."""

    local_radius_mm: float = 10.0
    local_distance_threshold_mm: float = 0.5
    local_ransac_iterations: int = 500
    min_neighbor_points: int = 200
    min_local_inliers: int = 100
    accumulate_threshold_mm: float = 1.0
    connect: bool = True  # restrict to the component containing the click
    cell_size_mm: float = 5.0
    min_points_per_cell: int = 3
    ransac_backend: str = "seeded"  # seeded (GPU) | seeded_cpu | open3d
    compute_backend: str = "auto"  # auto | numpy | cupy (Fit / Pick / UV)
    seed: int = 0
    # Progressive expand + refit (diagonal-cut mitigation)
    expand_step_mm: float = 25.0  # 0 disables progressive refine
    max_expand_rounds: int = 40
    max_inplane_radius_mm: float | None = None  # None = bounded by connectivity only
    refine_max_points: int = 300_000  # subsample cap for progressive rounds
    # Full-resolution accumulate <-> refit passes (corrects a tilted seed)
    final_refit_rounds: int = 3
    final_refit_tolerance: float = 0.01  # stop when the region changes < 1%
    # Cylinder shell pick: 0 → auto from estimated / fixed radius.
    cylinder_shell_half_width_mm: float = 0.0
    cylinder_axial_half_length_mm: float = 0.0


def resolve_cylinder_shell_mm(
    radius_mm: float,
    *,
    shell_half_width_mm: float | None = None,
    axial_half_length_mm: float | None = None,
) -> tuple[float, float]:
    """Resolve cylinder shell half-thickness and axial half-length (mm).

    ``0`` / ``None`` means auto: radial ``max(6, 0.15·r)``, axial
    ``max(40, 1.0·r)``.
    """
    r = max(float(radius_mm), 1e-6)
    if shell_half_width_mm is None or float(shell_half_width_mm) <= 0.0:
        half_w = float(max(6.0, 0.15 * r))
    else:
        half_w = float(shell_half_width_mm)
    if axial_half_length_mm is None or float(axial_half_length_mm) <= 0.0:
        half_len = float(max(40.0, 1.0 * r))
    else:
        half_len = float(axial_half_length_mm)
    return half_w, half_len


def _plane_distances(
    points: np.ndarray,
    plane: Plane,
    compute_backend: str = "auto",
    *,
    device_points: DevicePoints | None = None,
) -> np.ndarray:
    ctx = get_context(compute_backend, n_points=len(points))
    if ctx.name == "cupy":
        xp = ctx.xp
        pts = device_points.pts if device_points is not None else ctx.to_device(points)
        normal = xp.asarray(plane.normal, dtype=xp.float64)
        return ctx.asnumpy(xp.abs(pts @ normal + plane.d))
    return plane.distances(points)


def _slab_candidate_indices(
    points: np.ndarray,
    plane: Plane,
    params: PickParams,
    inplane_radius_mm: float | None,
    clicked: np.ndarray,
    compute_backend: str = "auto",
    *,
    device_points: DevicePoints | None = None,
) -> np.ndarray:
    """Distance slab (+ optional in-plane cap) without connectivity."""
    ctx = get_context(compute_backend, n_points=len(points))
    if ctx.name == "cupy" and device_points is not None:
        xp = ctx.xp
        pts = device_points.pts
        normal = xp.asarray(plane.normal, dtype=xp.float64)
        mask = xp.abs(pts @ normal + plane.d) <= params.accumulate_threshold_mm
        if inplane_radius_mm is not None and np.isfinite(inplane_radius_mm):
            clicked_g = xp.asarray(clicked, dtype=xp.float64)
            u, v = inplane_basis(plane.normal)
            u_g = xp.asarray(u, dtype=xp.float64)
            v_g = xp.asarray(v, dtype=xp.float64)
            delta = pts - clicked_g
            rad = xp.hypot(delta @ u_g, delta @ v_g)
            mask &= rad <= float(inplane_radius_mm)
        return ctx.asnumpy(xp.flatnonzero(mask))

    dists = _plane_distances(
        points, plane, compute_backend, device_points=device_points
    )
    mask = dists <= params.accumulate_threshold_mm
    if inplane_radius_mm is not None and np.isfinite(inplane_radius_mm):
        mask &= _inplane_radius(points, plane, clicked) <= inplane_radius_mm
    return np.flatnonzero(mask)


def _fit_local_plane(
    neighbors: np.ndarray, params: PickParams, *, compute_backend: str = "auto"
) -> Plane:
    if len(neighbors) < params.min_neighbor_points:
        raise ValueError(
            f"too few neighbor points: {len(neighbors)} < {params.min_neighbor_points}"
        )
    plane, inlier_mask = run_ransac(
        neighbors,
        threshold=params.local_distance_threshold_mm,
        n_iterations=params.local_ransac_iterations,
        seed=params.seed,
        backend=params.ransac_backend,
        compute_backend=compute_backend,
    )
    n_in = int(np.count_nonzero(inlier_mask))
    if n_in < params.min_local_inliers:
        raise ValueError(f"too few local inliers: {n_in} < {params.min_local_inliers}")
    return fit_plane_lsq(neighbors[inlier_mask], compute_backend=compute_backend)


def _inplane_radius(points: np.ndarray, plane: Plane, clicked: np.ndarray) -> np.ndarray:
    """In-plane radial distance from ``clicked`` for each point."""
    u, v = inplane_basis(plane.normal)
    clicked = np.asarray(clicked, dtype=np.float64)
    delta = points - clicked
    uu = delta @ u
    vv = delta @ v
    return np.hypot(uu, vv)


def _connected_indices(
    points: np.ndarray,
    candidate_idx: np.ndarray,
    plane: Plane,
    clicked: np.ndarray,
    params: PickParams,
    *,
    compute_backend: str = "auto",
    device_points: DevicePoints | None = None,
) -> np.ndarray:
    """Keep only candidates in the in-plane component containing the click.

    Uses the caller's ``cell_size_mm`` / ``min_points_per_cell`` as-is (no
    auto-coarsening). Intermediate pick stages already skip connectivity;
    this final pass must stay fine enough to separate coplanar faces.
    """
    pts = points[candidate_idx]
    u, v = inplane_basis(plane.normal)
    center = pts.mean(axis=0)
    uu = (pts - center) @ u
    vv = (pts - center) @ v

    n_in = len(candidate_idx)
    ctx = get_context(compute_backend, n_points=n_in)
    cs = float(params.cell_size_mm)
    min_pts = int(params.min_points_per_cell)

    if ctx.name == "cupy" and n_in >= 1_000:
        xp = ctx.xp
        pts_g = device_points.pts if device_points is not None else ctx.to_device(points)
        mask_g = xp.zeros(len(points), dtype=xp.bool_)
        mask_g[candidate_idx] = True
        inpts = pts_g[mask_g]
        u_g = xp.asarray(u, dtype=xp.float64)
        v_g = xp.asarray(v, dtype=xp.float64)
        center_g = inpts.mean(axis=0)
        uu_g = (inpts - center_g) @ u_g
        vv_g = (inpts - center_g) @ v_g
        umin = float(uu_g.min().item())
        vmin = float(vv_g.min().item())
        iu = xp.floor((uu_g - umin) / cs).astype(xp.int64)
        iv = xp.floor((vv_g - vmin) / cs).astype(xp.int64)
        grid_shape = (int(iu.max().item()) + 1, int(iv.max().item()) + 1)
        counts_g = xp.zeros(grid_shape, dtype=xp.int64)
        xp.add.at(counts_g, (iu, iv), 1)
        counts = ctx.asnumpy(counts_g)
        center_np = ctx.asnumpy(center_g)
        iu_np = ctx.asnumpy(iu)
        iv_np = ctx.asnumpy(iv)
        umin_f = umin
        vmin_f = vmin
    else:
        iu_np = np.floor((uu - uu.min()) / cs).astype(np.int64)
        iv_np = np.floor((vv - vv.min()) / cs).astype(np.int64)
        counts = np.zeros((int(iu_np.max()) + 1, int(iv_np.max()) + 1), dtype=np.int64)
        np.add.at(counts, (iu_np, iv_np), 1)
        center_np = center
        umin_f = float(uu.min())
        vmin_f = float(vv.min())

    occupied = counts >= min_pts
    labels = label_components(occupied)
    if labels.max() == 0:
        return candidate_idx

    clicked = np.asarray(clicked, dtype=np.float64)
    ci = int(np.floor(((clicked - center_np) @ u - umin_f) / cs))
    cj = int(np.floor(((clicked - center_np) @ v - vmin_f) / cs))
    main = 0
    if 0 <= ci < labels.shape[0] and 0 <= cj < labels.shape[1]:
        main = int(labels[ci, cj])
    if main == 0:
        sizes = np.bincount(labels.ravel(), weights=counts.ravel())
        main = int(np.argmax(sizes[1:])) + 1

    keep = labels[iu_np, iv_np] == main
    return candidate_idx[keep]


def _select_candidates(
    points: np.ndarray,
    plane: Plane,
    clicked: np.ndarray,
    params: PickParams,
    inplane_radius_mm: float | None,
    *,
    connect: bool | None = None,
    compute_backend: str = "auto",
    device_points: DevicePoints | None = None,
) -> np.ndarray:
    """Slab (+ optional in-plane radius) (+ optional connectivity)."""
    candidate_idx = _slab_candidate_indices(
        points,
        plane,
        params,
        inplane_radius_mm,
        clicked,
        compute_backend,
        device_points=device_points,
    )
    if len(candidate_idx) == 0:
        return candidate_idx
    do_connect = params.connect if connect is None else bool(connect)
    if do_connect:
        candidate_idx = _connected_indices(
            points,
            candidate_idx,
            plane,
            clicked,
            params,
            compute_backend=compute_backend,
            device_points=device_points,
        )
    return candidate_idx


def _sample_indices(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``size`` distinct indices from ``0..n-1`` without ``choice`` on huge n."""
    size = int(min(size, n))
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if size >= n:
        return np.arange(n, dtype=np.int64)
    if size * 8 < n:
        buf = np.empty(0, dtype=np.int64)
        while len(buf) < size:
            need = size - len(buf)
            draw = rng.integers(0, n, size=max(need * 2, need + 1024), dtype=np.int64)
            buf = np.unique(np.concatenate([buf, draw]))
        rng.shuffle(buf)
        return buf[:size]
    return rng.choice(n, size=size, replace=False).astype(np.int64, copy=False)


def _fit_plane_on_indices(
    points: np.ndarray,
    idx: np.ndarray,
    params: PickParams,
    *,
    compute_backend: str = "auto",
) -> Plane:
    """LSQ plane fit, subsampling when the candidate set is huge."""
    n = len(idx)
    cap = int(params.refine_max_points)
    if n <= cap:
        return fit_plane_lsq(points[idx], compute_backend=compute_backend)
    rng = np.random.default_rng(params.seed)
    sub = _sample_indices(n, cap, rng)
    return fit_plane_lsq(points[idx[sub]], compute_backend=compute_backend)


def _refine_subset(
    points: np.ndarray,
    clicked: np.ndarray,
    params: PickParams,
) -> np.ndarray:
    """Downsample for progressive rounds."""
    n = len(points)
    cap = int(params.refine_max_points)
    if n <= cap:
        return points

    rng = np.random.default_rng(params.seed)
    pool = min(n, max(cap * 4, cap))
    cand = _sample_indices(n, pool, rng)
    clicked = np.asarray(clicked, dtype=np.float64)
    d = np.linalg.norm(points[cand] - clicked, axis=1)
    order = np.argsort(d)
    n_near = cap // 2
    near = cand[order[:n_near]]
    far = cand[order[n_near:]]
    n_far = cap - len(near)
    if len(far) > n_far:
        far = far[_sample_indices(len(far), n_far, rng)]
    return points[np.concatenate([near, far])]


def _expand_params(params: PickParams, radius_mm: float) -> PickParams:
    cs = max(params.cell_size_mm, (2.0 * radius_mm) / _EXPAND_MAX_CELLS)
    return replace(params, cell_size_mm=cs, min_points_per_cell=1)


def _progressive_refine_plane(
    points: np.ndarray,
    clicked: np.ndarray,
    plane: Plane,
    params: PickParams,
    *,
    compute_backend: str = "auto",
) -> tuple[Plane, float]:
    """Expand in-plane radius from the click, refitting the plane each round."""
    R0 = float(params.local_radius_mm)
    if params.expand_step_mm <= 0 or params.max_expand_rounds <= 0:
        cap = params.max_inplane_radius_mm
        return plane, float(cap) if cap is not None else R0

    work = _refine_subset(points, clicked, params)
    R = R0
    max_R = params.max_inplane_radius_mm
    if max_R is not None and max_R < R:
        max_R = R

    prev_n = np.asarray(plane.normal, dtype=np.float64)
    for _ in range(int(params.max_expand_rounds)):
        idx = _select_candidates(
            work,
            plane,
            clicked,
            _expand_params(params, R),
            inplane_radius_mm=R,
            connect=False,
            compute_backend=compute_backend,
        )
        if len(idx) < 3:
            break
        plane = fit_plane_lsq(work[idx], compute_backend=compute_backend)

        nrm = np.asarray(plane.normal, dtype=np.float64)
        settled = float(np.abs(nrm @ prev_n)) > 0.999995
        prev_n = nrm

        at_cap = max_R is not None and R >= max_R - 1e-12
        if at_cap:
            break

        R_next = R + float(params.expand_step_mm)
        if max_R is not None:
            R_next = min(R_next, float(max_R))

        idx_next = _select_candidates(
            work,
            plane,
            clicked,
            _expand_params(params, R_next),
            inplane_radius_mm=R_next,
            connect=False,
            compute_backend=compute_backend,
        )
        if len(idx_next) <= len(idx):
            break
        if settled and len(idx_next) < int(1.05 * len(idx)):
            if max_R is not None:
                R = float(max_R)
                idx_cap = _select_candidates(
                    work,
                    plane,
                    clicked,
                    _expand_params(params, R),
                    inplane_radius_mm=R,
                    connect=False,
                    compute_backend=compute_backend,
                )
                if len(idx_cap) >= 3:
                    plane = fit_plane_lsq(work[idx_cap], compute_backend=compute_backend)
            break
        R = R_next

    return plane, float(R)


def _accumulate_with_refit(
    points: np.ndarray,
    clicked: np.ndarray,
    plane: Plane,
    params: PickParams,
    *,
    compute_backend: str = "auto",
    device_points: DevicePoints | None = None,
    timings: dict | None = None,
) -> tuple[np.ndarray, Plane]:
    """Accumulate on the full cloud, refitting until the region stabilises."""
    cap = params.max_inplane_radius_mm
    dist_s = lsq_s = connect_s = 0.0

    t0 = time.perf_counter()
    idx = _select_candidates(
        points,
        plane,
        clicked,
        params,
        inplane_radius_mm=cap,
        connect=False,
        compute_backend=compute_backend,
        device_points=device_points,
    )
    dist_s += time.perf_counter() - t0
    if len(idx) == 0:
        raise ValueError("accumulation selected no points")

    for _ in range(max(0, int(params.final_refit_rounds))):
        if len(idx) < 3:
            break
        t1 = time.perf_counter()
        new_plane = _fit_plane_on_indices(
            points, idx, params, compute_backend=compute_backend
        )
        lsq_s += time.perf_counter() - t1
        t2 = time.perf_counter()
        new_idx = _select_candidates(
            points,
            new_plane,
            clicked,
            params,
            inplane_radius_mm=cap,
            connect=False,
            compute_backend=compute_backend,
            device_points=device_points,
        )
        dist_s += time.perf_counter() - t2
        if len(new_idx) < 3:
            break
        grew = abs(len(new_idx) - len(idx)) / max(len(idx), 1)
        plane, idx = new_plane, new_idx
        if grew < params.final_refit_tolerance:
            break

    if params.connect and len(idx):
        t3 = time.perf_counter()
        idx = _connected_indices(
            points,
            idx,
            plane,
            clicked,
            params,
            compute_backend=compute_backend,
            device_points=device_points,
        )
        connect_s = time.perf_counter() - t3

    if timings is not None:
        timings["accumulate_dist_s"] = timings.get("accumulate_dist_s", 0.0) + dist_s
        timings["accumulate_lsq_s"] = timings.get("accumulate_lsq_s", 0.0) + lsq_s
        timings["accumulate_connect_s"] = (
            timings.get("accumulate_connect_s", 0.0) + connect_s
        )
        timings["n_candidates"] = int(len(idx))
    return idx, plane


def pick_ball_region(
    points: np.ndarray,
    clicked: np.ndarray,
    ball_idx: np.ndarray,
    *,
    timings: dict | None = None,
) -> tuple[np.ndarray, Plane]:
    """Keep points in a ball around the click (legacy cylinder seed).

    Prefer ``pick_cylinder_region`` for ducts/pipes (radial shell filter).
    """
    points = np.asarray(points, dtype=np.float64)
    ball_idx = np.asarray(ball_idx, dtype=np.int64)
    clicked = np.asarray(clicked, dtype=np.float64).reshape(3)
    t0 = time.perf_counter()
    if len(ball_idx) < 3:
        raise ValueError(
            f"cylinder pick: too few points in seed ball ({len(ball_idx)}; "
            "increase Pick local radius or click a denser area)"
        )
    rng = np.random.default_rng(0)
    if len(ball_idx) > 5000:
        sample = ball_idx[rng.choice(len(ball_idx), size=5000, replace=False)]
    else:
        sample = ball_idx
    plane = fit_plane_lsq(points[sample])
    if timings is not None:
        timings.update({
            "local_fit_s": time.perf_counter() - t0,
            "progressive_s": 0.0,
            "accumulate_s": 0.0,
            "n_candidates": int(len(ball_idx)),
            "pick_mode": "ball",
        })
    return ball_idx.astype(np.int64, copy=False), plane


def pick_cylinder_region(
    points: np.ndarray,
    clicked: np.ndarray,
    ball_idx: np.ndarray,
    *,
    diameter_mm: float | None = None,
    shell_half_width_mm: float | None = None,
    axial_half_length_mm: float | None = None,
    max_ball_points: int = CYLINDER_PICK_MAX_BALL_POINTS,
    timings: dict | None = None,
) -> tuple[np.ndarray, Plane]:
    """Filter a seed ball to a cylindrical shell around a PCA axis.

    1. PCA axis from the ball (largest-variance direction).
    2. Seed radius = ``diameter_mm/2`` if given, else median distance to axis
       among points near the click.
    3. Keep points with ``|radial − r| ≤ shell_half_width`` and axial distance
       from the click projection within ``axial_half_length``.

    Returns filtered indices and a coarse LSQ plane for group bookkeeping.

    Never promotes the full cloud to float64 — only the seed-ball subset
    is copied (and subsampled if larger than ``max_ball_points``).
    """
    # Index into the caller's array as-is (often float32); do not cast all points.
    points_all = np.asarray(points)
    ball_idx = np.asarray(ball_idx, dtype=np.int64)
    clicked = np.asarray(clicked, dtype=np.float64).reshape(3)
    t0 = time.perf_counter()
    if len(ball_idx) < 3:
        raise ValueError(
            f"cylinder pick: too few points in seed ball ({len(ball_idx)}; "
            "increase Pick local radius or click a denser area)"
        )

    max_ball = max(3, int(max_ball_points))
    n_ball_raw = int(len(ball_idx))
    rng = np.random.default_rng(0)
    if n_ball_raw > max_ball:
        ball_idx = ball_idx[rng.choice(n_ball_raw, size=max_ball, replace=False)]

    # Promote only the (possibly subsampled) ball — not the whole cloud.
    pts = np.asarray(points_all[ball_idx], dtype=np.float64, order="C")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, int(np.argmax(evals))]
    u = u / np.linalg.norm(u)

    # Prefer axis through the click neighborhood: axis point near click.
    axis_point = clicked - float((clicked - centroid) @ u) * u
    radial = distances_to_axis(pts, axis_point, u)
    # Local ring estimate: points within 25 mm of the click (Euclidean).
    near = np.linalg.norm(pts - clicked, axis=1) <= 25.0
    if int(np.count_nonzero(near)) >= 20:
        seed_r = float(np.median(radial[near]))
    else:
        seed_r = float(np.median(radial))
    if diameter_mm is not None and float(diameter_mm) > 0:
        seed_r = 0.5 * float(diameter_mm)
    if not np.isfinite(seed_r) or seed_r <= 0:
        seed_r = 40.0

    half_w, half_len = resolve_cylinder_shell_mm(
        seed_r,
        shell_half_width_mm=shell_half_width_mm,
        axial_half_length_mm=axial_half_length_mm,
    )

    t_ax = (pts - clicked) @ u
    shell = (np.abs(radial - seed_r) <= half_w) & (np.abs(t_ax) <= half_len)
    keep = ball_idx[shell]
    if len(keep) < 50:
        # Loosen shell once so tiny ducts / sparse scans still get a seed.
        shell = (np.abs(radial - seed_r) <= 2.0 * half_w) & (
            np.abs(t_ax) <= 2.0 * half_len
        )
        keep = ball_idx[shell]
    if len(keep) < 20:
        # Last resort: subsample the ball — never return millions of points.
        if len(ball_idx) > 80_000:
            keep = ball_idx[rng.choice(len(ball_idx), size=80_000, replace=False)]
        else:
            keep = ball_idx

    sample = keep
    if len(keep) > 5000:
        sample = keep[rng.choice(len(keep), size=5000, replace=False)]
    plane = fit_plane_lsq(np.asarray(points_all[sample], dtype=np.float64))
    if timings is not None:
        timings.update({
            "local_fit_s": time.perf_counter() - t0,
            "progressive_s": 0.0,
            "accumulate_s": 0.0,
            "n_candidates": int(len(keep)),
            "n_ball": n_ball_raw,
            "n_ball_used": int(len(ball_idx)),
            "seed_radius_mm": float(seed_r),
            "shell_half_width_mm": float(half_w),
            "axial_half_length_mm": float(half_len),
            "pick_mode": "cylinder_shell",
        })
    return keep.astype(np.int64, copy=False), plane


def pick_cylinder_region_from_cylinder(
    points: np.ndarray,
    ball_idx: np.ndarray,
    cylinder,
    *,
    anchor: np.ndarray | None = None,
    shell_half_width_mm: float | None = None,
    axial_half_length_mm: float | None = None,
    max_ball_points: int = CYLINDER_PICK_MAX_BALL_POINTS,
    timings: dict | None = None,
) -> tuple[np.ndarray, Plane]:
    """Filter a seed ball using a known cylinder axis / radius.

    Used after a 3-point circumference seed: PCA is skipped; the shell is
    centered on ``cylinder.point`` / ``cylinder.direction`` / radius.
    ``anchor`` (default: cylinder axis point) limits axial extent.
    """
    from cloudet.core.cylinder import Cylinder

    if not isinstance(cylinder, Cylinder):
        raise TypeError("cylinder must be a Cylinder")
    points_all = np.asarray(points)
    ball_idx = np.asarray(ball_idx, dtype=np.int64)
    t0 = time.perf_counter()
    if len(ball_idx) < 3:
        raise ValueError(
            f"cylinder pick: too few points in seed ball ({len(ball_idx)})"
        )

    max_ball = max(3, int(max_ball_points))
    n_ball_raw = int(len(ball_idx))
    rng = np.random.default_rng(0)
    if n_ball_raw > max_ball:
        ball_idx = ball_idx[rng.choice(n_ball_raw, size=max_ball, replace=False)]

    pts = np.asarray(points_all[ball_idx], dtype=np.float64, order="C")
    u = np.asarray(cylinder.direction, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    axis_point = np.asarray(cylinder.point, dtype=np.float64).reshape(3)
    seed_r = float(cylinder.radius_mm)
    if anchor is None:
        anchor = axis_point
    else:
        anchor = np.asarray(anchor, dtype=np.float64).reshape(3)

    half_w, half_len = resolve_cylinder_shell_mm(
        seed_r,
        shell_half_width_mm=shell_half_width_mm,
        axial_half_length_mm=axial_half_length_mm,
    )

    radial = distances_to_axis(pts, axis_point, u)
    t_ax = (pts - anchor) @ u
    shell = (np.abs(radial - seed_r) <= half_w) & (np.abs(t_ax) <= half_len)
    keep = ball_idx[shell]
    if len(keep) < 50:
        shell = (np.abs(radial - seed_r) <= 2.0 * half_w) & (
            np.abs(t_ax) <= 2.0 * half_len
        )
        keep = ball_idx[shell]
    if len(keep) < 20:
        # Last resort: subsample the ball — never return millions of points.
        if len(ball_idx) > 80_000:
            keep = ball_idx[rng.choice(len(ball_idx), size=80_000, replace=False)]
        else:
            keep = ball_idx

    sample = keep
    if len(keep) > 5000:
        sample = keep[rng.choice(len(keep), size=5000, replace=False)]
    plane = fit_plane_lsq(np.asarray(points_all[sample], dtype=np.float64))
    if timings is not None:
        timings.update({
            "local_fit_s": time.perf_counter() - t0,
            "progressive_s": 0.0,
            "accumulate_s": 0.0,
            "n_candidates": int(len(keep)),
            "n_ball": n_ball_raw,
            "n_ball_used": int(len(ball_idx)),
            "seed_radius_mm": float(seed_r),
            "shell_half_width_mm": float(half_w),
            "axial_half_length_mm": float(half_len),
            "pick_mode": "cylinder_shell_3pt",
        })
    return keep.astype(np.int64, copy=False), plane


def pick_plane_region(
    points: np.ndarray,
    clicked: np.ndarray,
    neighbor_idx: np.ndarray,
    params: PickParams = PickParams(),
    *,
    compute_backend: str = "auto",
    timings: dict | None = None,
) -> tuple[np.ndarray, Plane]:
    """Extract a plane-candidate region around a click."""
    points = np.asarray(points, dtype=np.float64)
    neighbor_idx = np.asarray(neighbor_idx, dtype=np.int64)
    clicked = np.asarray(clicked, dtype=np.float64)
    device_points = DevicePoints.create(points, compute_backend)

    t0 = time.perf_counter()
    plane = _fit_local_plane(
        points[neighbor_idx], params, compute_backend=compute_backend
    )
    local_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    plane, _final_R = _progressive_refine_plane(
        points, clicked, plane, params, compute_backend=compute_backend
    )
    progressive_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    idx, plane = _accumulate_with_refit(
        points,
        clicked,
        plane,
        params,
        compute_backend=compute_backend,
        device_points=device_points,
        timings=timings,
    )
    accumulate_s = time.perf_counter() - t2

    if timings is not None:
        timings.update({
            "local_fit_s": local_s,
            "progressive_s": progressive_s,
            "accumulate_s": accumulate_s,
        })
    return idx, plane
