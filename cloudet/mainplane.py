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

Pure numpy; the grid labelling is a BFS flood fill on the occupancy
grid, which is at most ``grid_bins**2`` cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cloudet.plane import FitResult, Plane, fit_plane_lsq, robust_fit_plane, run_ransac

__all__ = ["MainPlaneParams", "MainPlaneResult", "extract_main_plane"]


@dataclass(frozen=True)
class MainPlaneParams:
    """All lengths in mm."""

    # step 1: RANSAC seed
    ransac_threshold_mm: float = 0.3
    ransac_iterations: int = 1000
    ransac_backend: str = "numpy"  # "numpy" (seeded, reproducible) or "open3d"
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


def _inplane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0])
    if abs(normal @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, a)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _label_components(occupied: np.ndarray) -> np.ndarray:
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


def _bounded_robust_fit(points, params: MainPlaneParams, init: Plane) -> FitResult:
    """Adaptive robust fit, but never let the threshold exceed the ceiling."""
    kw = dict(
        threshold=None,
        sigma_factor=params.sigma_factor,
        max_iterations=params.max_iterations,
        init=init,
        min_inlier_fraction=0.0,
        compute_backend=params.compute_backend,
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

    # --- 1. seed ---------------------------------------------------------
    try:
        seed_plane, _ = run_ransac(
            points,
            threshold=params.ransac_threshold_mm,
            n_iterations=params.ransac_iterations,
            seed=params.seed,
            backend=params.ransac_backend,
            compute_backend=params.compute_backend,
        )
    except ValueError:
        if coarse_plane is None:
            raise
        seed_plane = Plane.from_array(coarse_plane)

    # --- 2. bounded robust refit (all points) ----------------------------
    fit0 = _bounded_robust_fit(points, params, init=seed_plane)
    inlier_mask = fit0.inlier_mask
    diag["stage2_threshold_mm"] = fit0.threshold
    diag["stage2_n_inliers"] = fit0.n_inliers

    if fit0.n_inliers < max(3, params.min_points):
        return MainPlaneResult(
            plane=fit0.plane, fit=fit0, main_mask=inlier_mask,
            status="fail", reasons=["too_few_inliers_after_refit"], diagnostics=diag,
        )

    # --- 3. in-plane connectivity ----------------------------------------
    normal = fit0.plane.normal
    u, v = _inplane_basis(normal)
    inpts = points[inlier_mask]
    center = inpts.mean(axis=0)
    uu = (inpts - center) @ u
    vv = (inpts - center) @ v

    cs = params.cell_size_mm
    iu = np.floor((uu - uu.min()) / cs).astype(np.int64)
    iv = np.floor((vv - vv.min()) / cs).astype(np.int64)
    grid_shape = (int(iu.max()) + 1, int(iv.max()) + 1)
    counts = np.zeros(grid_shape, dtype=np.int64)
    np.add.at(counts, (iu, iv), 1)
    occupied = counts >= params.min_points_per_cell
    labels = _label_components(occupied)
    n_components = int(labels.max())
    diag["n_components"] = n_components

    if n_components == 0:
        return MainPlaneResult(
            plane=fit0.plane, fit=fit0, main_mask=inlier_mask,
            status="fail", reasons=["no_occupied_cells"], diagnostics=diag,
        )

    # choose the component: the one under the picker click, else largest
    main_label = 0
    if clicked is not None:
        cu = (np.asarray(clicked, dtype=np.float64) - center) @ u
        cv = (np.asarray(clicked, dtype=np.float64) - center) @ v
        ci = int(np.floor((cu - uu.min()) / cs))
        cj = int(np.floor((cv - vv.min()) / cs))
        if 0 <= ci < grid_shape[0] and 0 <= cj < grid_shape[1]:
            main_label = int(labels[ci, cj])
        if main_label == 0:
            reasons.append("click_not_on_component_using_largest")
    if main_label == 0:
        sizes = np.bincount(labels.ravel(), weights=counts.ravel(), minlength=n_components + 1)
        main_label = int(np.argmax(sizes[1:])) + 1

    point_in_main = labels[iu, iv] == main_label
    main_mask = np.zeros(n, dtype=bool)
    main_mask[np.flatnonzero(inlier_mask)] = point_in_main
    main_fraction = point_in_main.mean()
    diag["main_fraction_of_inliers"] = float(main_fraction)

    if np.count_nonzero(main_mask) < max(3, params.min_points):
        return MainPlaneResult(
            plane=fit0.plane, fit=fit0, main_mask=main_mask,
            status="fail", reasons=reasons + ["main_component_too_small"], diagnostics=diag,
        )

    # --- 4. final refit on the main component ----------------------------
    fit = _bounded_robust_fit(points[main_mask], params, init=fit0.plane)
    # rebase masks onto the full input
    final_mask = np.zeros(n, dtype=bool)
    final_mask[np.flatnonzero(main_mask)] = fit.inlier_mask
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
