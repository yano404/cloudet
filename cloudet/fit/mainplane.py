"""Main plane component extraction from rough picker groups.

Picker groups are rough: they may contain several surfaces, curved
regions, and "infinite slab" contamination (points that are coplanar
with the target but belong to spatially distant structures). This
stage isolates the *main planar component*:

1. seed plane via RANSAC (picker's coarse plane used as fallback seed)
2. robust refit with a bounded threshold (an unbounded adaptive
   threshold diverges on non-planar groups)
3. connected-component analysis of the inliers on an in-plane (u, v)
   occupancy grid; the component containing the picker's click is the
   main one (largest component as fallback)
4. final robust refit on the main component only
5. quality gates -> status "ok" / "suspect" / "fail" with reasons

Pure numpy for grid labelling (BFS flood fill on the occupancy grid,
at most ``grid_bins**2`` cells). In-plane projection and cell counting
use the compute backend when set to cupy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cloudet.core.array_backend import DevicePoints, get_context
from cloudet.core.plane import FitResult, Plane, robust_fit_plane, run_ransac

__all__ = [
    "MainPlaneParams",
    "MainPlaneResult",
    "extract_main_plane",
    "inplane_basis",
    "label_components",
]


@dataclass(frozen=True)
class MainPlaneParams:
    """All lengths in mm."""

    # step 1: RANSAC seed
    ransac_threshold_mm: float = 0.3
    ransac_iterations: int = 1000
    ransac_backend: str = "seeded"  # seeded (GPU) | seeded_cpu | open3d
    seed: int = 0
    # step 2/4: robust refit
    sigma_factor: float = 3.0
    max_threshold_mm: float = 0.3  # ceiling for the adaptive threshold
    max_iterations: int = 100
    compute_backend: str = "auto"  # auto | numpy | cupy
    # step 3: connectivity
    cell_size_mm: float = 5.0
    min_points_per_cell: int = 5
    # step 5: quality gates
    ok_mad_sigma_mm: float = 0.10  # <= nominal-ish noise -> ok
    fail_mad_sigma_mm: float = 0.25  # above this -> fail
    min_main_fraction: float = 0.2  # main component vs all inliers
    min_points: int = 1000
    fit_max_points: int = 300_000  # subsample cap for iterative robust refit
    skip_ransac_min_points: int = 50_000  # reuse picker coarse plane above this


@dataclass
class MainPlaneResult:
    plane: Plane
    fit: FitResult  # final fit on the main component
    main_mask: np.ndarray  # bool mask on the input points
    status: str  # "ok" | "suspect" | "fail"
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_main(self) -> int:
        return int(np.count_nonzero(self.main_mask))


def inplane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0])
    if abs(normal @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, a)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def label_components(occupied: np.ndarray) -> np.ndarray:
    """4-connectivity labelling of a boolean grid. Returns int labels, 0 = empty.

    Vectorised max-label propagation: a per-cell Python BFS was the picker's
    dominant cost once the in-plane grid grew to hundreds of cells per side.
    """
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.size == 0 or not occupied.any():
        return np.zeros(occupied.shape, dtype=np.int32)

    lab = np.where(
        occupied,
        np.arange(1, occupied.size + 1, dtype=np.int64).reshape(occupied.shape),
        0,
    )
    while True:
        m = lab.copy()
        m[1:, :] = np.maximum(m[1:, :], lab[:-1, :])
        m[:-1, :] = np.maximum(m[:-1, :], lab[1:, :])
        m[:, 1:] = np.maximum(m[:, 1:], lab[:, :-1])
        m[:, :-1] = np.maximum(m[:, :-1], lab[:, 1:])
        m[~occupied] = 0
        if np.array_equal(m, lab):
            break
        lab = m

    uniq = np.unique(lab[occupied])
    remap = np.zeros(int(lab.max()) + 1, dtype=np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1, dtype=np.int32)
    return remap[lab]


def _select_main_component(
    iu,
    iv,
    counts,
    labels,
    clicked,
    center,
    u,
    v,
    cs: float,
    umin: float,
    vmin: float,
) -> tuple[int, list[str], float]:
    """Pick the connected component under ``clicked``, else the largest."""
    reasons: list[str] = []
    n_components = int(labels.max())
    grid_shape = labels.shape
    main_label = 0
    if clicked is not None:
        cu = (np.asarray(clicked, dtype=np.float64) - center) @ u
        cv = (np.asarray(clicked, dtype=np.float64) - center) @ v
        ci = int(np.floor((cu - umin) / cs))
        cj = int(np.floor((cv - vmin) / cs))
        if 0 <= ci < grid_shape[0] and 0 <= cj < grid_shape[1]:
            main_label = int(labels[ci, cj])
        if main_label == 0:
            reasons.append("click_not_on_component_using_largest")
    if main_label == 0:
        sizes = np.bincount(
            labels.ravel(), weights=counts.ravel(), minlength=n_components + 1
        )
        main_label = int(np.argmax(sizes[1:])) + 1
    point_in_main = labels[iu, iv] == main_label
    return main_label, reasons, float(point_in_main.mean())


def _connectivity_main_mask(
    points: np.ndarray,
    inlier_mask: np.ndarray,
    normal: np.ndarray,
    clicked: np.ndarray | None,
    params: MainPlaneParams,
    device_points: DevicePoints | None = None,
) -> tuple[np.ndarray, dict, list[str]]:
    """In-plane grid connectivity: return ``main_mask`` on full ``points``."""
    n = len(points)
    u, v = inplane_basis(normal)
    n_in = int(np.count_nonzero(inlier_mask))
    ctx = (
        device_points.ctx
        if device_points is not None
        else get_context(params.compute_backend, n_points=n_in)
    )
    cs = params.cell_size_mm

    if ctx.name == "cupy" and n_in >= 1_000:
        xp = ctx.xp
        pts_g = device_points.pts if device_points is not None else ctx.to_device(points)
        mask_g = ctx.to_device_bool(inlier_mask)
        inpts = pts_g[mask_g]
        u_g = xp.asarray(u, dtype=xp.float64)
        v_g = xp.asarray(v, dtype=xp.float64)
        center = inpts.mean(axis=0)
        uu = (inpts - center) @ u_g
        vv = (inpts - center) @ v_g
        umin = float(uu.min().item())
        vmin = float(vv.min().item())
        iu = xp.floor((uu - umin) / cs).astype(xp.int64)
        iv = xp.floor((vv - vmin) / cs).astype(xp.int64)
        grid_shape = (int(iu.max().item()) + 1, int(iv.max().item()) + 1)
        counts_g = xp.zeros(grid_shape, dtype=xp.int64)
        xp.add.at(counts_g, (iu, iv), 1)
        counts = ctx.asnumpy(counts_g)
        center_np = ctx.asnumpy(center)
        iu_np = ctx.asnumpy(iu)
        iv_np = ctx.asnumpy(iv)
    else:
        inpts = points[inlier_mask]
        center_np = inpts.mean(axis=0)
        uu_np = (inpts - center_np) @ u
        vv_np = (inpts - center_np) @ v
        umin = float(uu_np.min())
        vmin = float(vv_np.min())
        iu_np = np.floor((uu_np - umin) / cs).astype(np.int64)
        iv_np = np.floor((vv_np - vmin) / cs).astype(np.int64)
        grid_shape = (int(iu_np.max()) + 1, int(iv_np.max()) + 1)
        counts = np.zeros(grid_shape, dtype=np.int64)
        np.add.at(counts, (iu_np, iv_np), 1)

    occupied = counts >= params.min_points_per_cell
    labels = label_components(occupied)
    n_components = int(labels.max())
    diag = {"n_components": n_components}

    if n_components == 0:
        main_mask = np.zeros(n, dtype=bool)
        main_mask[inlier_mask] = True
        return main_mask, diag, ["no_occupied_cells"]

    main_label, reasons, main_fraction = _select_main_component(
        iu_np, iv_np, counts, labels, clicked, center_np, u, v, cs, umin, vmin
    )
    diag["main_fraction_of_inliers"] = main_fraction
    point_in_main = labels[iu_np, iv_np] == main_label
    main_mask = np.zeros(n, dtype=bool)
    main_mask[np.flatnonzero(inlier_mask)] = point_in_main
    return main_mask, diag, reasons


def _bounded_robust_fit(
    points,
    params: MainPlaneParams,
    init: Plane,
    *,
    device_points: DevicePoints | None = None,
    fit_mask: np.ndarray | None = None,
) -> FitResult:
    """Adaptive robust fit, but never let the threshold exceed the ceiling."""
    kw = dict(
        threshold=None,
        sigma_factor=params.sigma_factor,
        max_iterations=params.max_iterations,
        init=init,
        min_inlier_fraction=0.0,
        compute_backend=params.compute_backend,
        device_points=device_points,
        fit_mask=fit_mask,
        max_fit_points=params.fit_max_points,
        seed=params.seed,
    )
    fit = robust_fit_plane(points, **kw)
    if fit.threshold > params.max_threshold_mm:
        fit = robust_fit_plane(
            points,
            threshold=params.max_threshold_mm,
            max_iterations=params.max_iterations,
            init=init,
            min_inlier_fraction=0.0,
            compute_backend=params.compute_backend,
            device_points=device_points,
            fit_mask=fit_mask,
            max_fit_points=params.fit_max_points,
            seed=params.seed,
        )
    return fit


def extract_main_plane(
    points: np.ndarray,
    params: MainPlaneParams = MainPlaneParams(),
    clicked: np.ndarray | None = None,
    coarse_plane: np.ndarray | None = None,
) -> MainPlaneResult:
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    reasons: list[str] = []
    diag: dict = {}
    device_points = DevicePoints.create(points, params.compute_backend)

    # --- 1. seed ---------------------------------------------------------
    if coarse_plane is not None and n >= int(params.skip_ransac_min_points):
        seed_plane = Plane.from_array(coarse_plane)
        diag["ransac_skipped"] = True
    else:
        try:
            seed_plane, _ = run_ransac(
                points,
                threshold=params.ransac_threshold_mm,
                n_iterations=params.ransac_iterations,
                seed=params.seed,
                backend=params.ransac_backend,
                compute_backend=params.compute_backend,
                device_points=device_points,
                return_inlier_mask=False,
            )
            diag["ransac_skipped"] = False
        except ValueError:
            if coarse_plane is None:
                raise
            seed_plane = Plane.from_array(coarse_plane)
            diag["ransac_skipped"] = True

    # --- 2. bounded robust refit (all points) ----------------------------
    fit0 = _bounded_robust_fit(
        points, params, init=seed_plane, device_points=device_points
    )
    inlier_mask = fit0.inlier_mask
    diag["stage2_threshold_mm"] = fit0.threshold
    diag["stage2_n_inliers"] = fit0.n_inliers

    if fit0.n_inliers < max(3, params.min_points):
        return MainPlaneResult(
            plane=fit0.plane, fit=fit0, main_mask=inlier_mask,
            status="fail", reasons=["too_few_inliers_after_refit"], diagnostics=diag,
        )

    # --- 3. in-plane connectivity ----------------------------------------
    main_mask, conn_diag, conn_reasons = _connectivity_main_mask(
        points,
        inlier_mask,
        fit0.plane.normal,
        clicked,
        params,
        device_points=device_points,
    )
    reasons.extend(conn_reasons)
    diag.update(conn_diag)
    main_fraction = conn_diag.get("main_fraction_of_inliers", 1.0)

    if conn_diag.get("n_components", 0) == 0:
        return MainPlaneResult(
            plane=fit0.plane, fit=fit0, main_mask=main_mask,
            status="fail", reasons=reasons, diagnostics=diag,
        )

    if np.count_nonzero(main_mask) < max(3, params.min_points):
        return MainPlaneResult(
            plane=fit0.plane, fit=fit0, main_mask=main_mask,
            status="fail", reasons=reasons + ["main_component_too_small"], diagnostics=diag,
        )

    # --- 4. final refit on the main component (mask on device) -----------
    fit = _bounded_robust_fit(
        points,
        params,
        init=fit0.plane,
        device_points=device_points,
        fit_mask=main_mask,
    )
    final_mask = fit.inlier_mask
    fit_full = FitResult(
        plane=fit.plane,
        inlier_mask=final_mask,
        n_iterations=fit.n_iterations,
        converged=fit.converged,
        threshold=fit.threshold,
        stats_inliers=fit.stats_inliers,
        stats_all=fit.stats_all,
    )

    # --- 5. quality gates -------------------------------------------------
    mad = fit.stats_inliers["mad_sigma"]
    status = "ok"
    if not fit.converged:
        status = "suspect"
        reasons.append("not_converged")
    if main_fraction < params.min_main_fraction:
        status = "suspect"
        reasons.append("main_component_minor_fraction")
    if mad > params.ok_mad_sigma_mm:
        status = "suspect" if status == "ok" else status
        reasons.append("mad_sigma_above_ok")
    if mad > params.fail_mad_sigma_mm:
        status = "fail"
        reasons.append("mad_sigma_above_fail")

    return MainPlaneResult(
        plane=fit.plane, fit=fit_full, main_mask=final_mask,
        status=status, reasons=reasons, diagnostics=diag,
    )
