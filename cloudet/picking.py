"""Click-driven plane-region extraction (GUI-independent logic).

Given a clicked 3D position on the cloud, fit a local plane to the
neighbourhood and accumulate points belonging to that surface.

Wide planar faces are easy to cut diagonally if the local normal is
even slightly tilted: a thin infinite slab then selects a band that
crosses the true face. To reduce that failure mode this module:

1. Fits a local plane near the click (RANSAC + LSQ).
2. Progressively expands an in-plane radius from the click, refitting
   the plane each round so the normal can correct before the region
   grows across the whole face. Intermediate rounds skip connectivity
   (radius + slab only) so they stay cheap on large clouds.
3. Accumulates on the full cloud, restricted by default to the in-plane
   connected component containing the click, then refits and re-accumulates
   until the region stabilises. The contract is one click -> one connected
   physical face, so the region is bounded by connectivity, not by radius.

This module is pure numpy so it can be unit-tested without a GUI;
neighbour search is injected by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cloudet.mainplane import _inplane_basis, _label_components
from cloudet.plane import Plane, fit_plane_lsq, run_ransac

__all__ = ["PickParams", "pick_plane_region"]


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
    ransac_backend: str = "numpy"  # seeded (GPU when compute=cupy) or open3d
    seed: int = 0
    # Progressive expand + refit (diagonal-cut mitigation)
    expand_step_mm: float = 25.0  # 0 disables progressive refine
    max_expand_rounds: int = 40
    max_inplane_radius_mm: float | None = None  # None = bounded by connectivity only
    refine_max_points: int = 300_000  # subsample cap for progressive rounds
    # Full-resolution accumulate <-> refit passes (corrects a tilted seed)
    final_refit_rounds: int = 3
    final_refit_tolerance: float = 0.01  # stop when the region changes < 1%


def _fit_local_plane(neighbors: np.ndarray, params: PickParams) -> Plane:
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
    )
    n_in = int(np.count_nonzero(inlier_mask))
    if n_in < params.min_local_inliers:
        raise ValueError(f"too few local inliers: {n_in} < {params.min_local_inliers}")
    return fit_plane_lsq(neighbors[inlier_mask])


def _inplane_radius(points: np.ndarray, plane: Plane, clicked: np.ndarray) -> np.ndarray:
    """In-plane radial distance from ``clicked`` for each point."""
    u, v = _inplane_basis(plane.normal)
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
) -> np.ndarray:
    """Keep only candidates in the in-plane component containing the click."""
    pts = points[candidate_idx]
    u, v = _inplane_basis(plane.normal)
    center = pts.mean(axis=0)
    uu = (pts - center) @ u
    vv = (pts - center) @ v

    cs = params.cell_size_mm
    iu = np.floor((uu - uu.min()) / cs).astype(np.int64)
    iv = np.floor((vv - vv.min()) / cs).astype(np.int64)
    counts = np.zeros((int(iu.max()) + 1, int(iv.max()) + 1), dtype=np.int64)
    np.add.at(counts, (iu, iv), 1)
    occupied = counts >= params.min_points_per_cell
    labels = _label_components(occupied)
    if labels.max() == 0:
        return candidate_idx

    clicked = np.asarray(clicked, dtype=np.float64)
    ci = int(np.floor(((clicked - center) @ u - uu.min()) / cs))
    cj = int(np.floor(((clicked - center) @ v - vv.min()) / cs))
    main = 0
    if 0 <= ci < labels.shape[0] and 0 <= cj < labels.shape[1]:
        main = int(labels[ci, cj])
    if main == 0:
        sizes = np.bincount(labels.ravel(), weights=counts.ravel())
        main = int(np.argmax(sizes[1:])) + 1

    keep = labels[iu, iv] == main
    return candidate_idx[keep]


def _select_candidates(
    points: np.ndarray,
    plane: Plane,
    clicked: np.ndarray,
    params: PickParams,
    inplane_radius_mm: float | None,
    *,
    connect: bool | None = None,
) -> np.ndarray:
    """Slab (+ optional in-plane radius) (+ optional connectivity)."""
    dists = plane.distances(points)
    mask = dists <= params.accumulate_threshold_mm
    if inplane_radius_mm is not None and np.isfinite(inplane_radius_mm):
        mask &= _inplane_radius(points, plane, clicked) <= inplane_radius_mm
    candidate_idx = np.flatnonzero(mask)
    if len(candidate_idx) == 0:
        return candidate_idx
    do_connect = params.connect if connect is None else bool(connect)
    if do_connect:
        candidate_idx = _connected_indices(
            points, candidate_idx, plane, clicked, params
        )
    return candidate_idx


def _sample_indices(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``size`` distinct indices from ``0..n-1`` without ``choice`` on huge n.

    ``numpy.random.Generator.choice(..., replace=False)`` on tens of millions
    of points can allocate a full permutation and appear to hang.
    """
    size = int(min(size, n))
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if size >= n:
        return np.arange(n, dtype=np.int64)
    if size * 8 < n:
        # Rejection sampling: expected collisions stay small.
        buf = np.empty(0, dtype=np.int64)
        while len(buf) < size:
            need = size - len(buf)
            draw = rng.integers(0, n, size=max(need * 2, need + 1024), dtype=np.int64)
            buf = np.unique(np.concatenate([buf, draw]))
        rng.shuffle(buf)
        return buf[:size]
    return rng.choice(n, size=size, replace=False).astype(np.int64, copy=False)


def _refine_subset(
    points: np.ndarray,
    clicked: np.ndarray,
    params: PickParams,
) -> np.ndarray:
    """Downsample for progressive rounds.

    Must stay cheap on tens of millions of points: never voxelize / permute
    the full cloud (that hung the picker). Take a random pool, then keep a
    click-biased mix so early expand rounds still see the local surface.
    """
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


_EXPAND_MAX_CELLS = 160


def _expand_params(params: PickParams, radius_mm: float) -> PickParams:
    """Connectivity settings for one expand round.

    The grid is coarsened so its side never exceeds ``_EXPAND_MAX_CELLS``
    cells: labelling cost then stays bounded no matter how far the region
    grows. ``min_points_per_cell`` drops to 1 because expand rounds run on
    a subsampled working set.
    """
    cs = max(params.cell_size_mm, (2.0 * radius_mm) / _EXPAND_MAX_CELLS)
    return replace(params, cell_size_mm=cs, min_points_per_cell=1)


def _progressive_refine_plane(
    points: np.ndarray,
    clicked: np.ndarray,
    plane: Plane,
    params: PickParams,
) -> tuple[Plane, float]:
    """Expand in-plane radius from the click, refitting the plane each round.

    Returns ``(refined_plane, final_radius_mm)``. Connectivity stays on so
    a tilted seed slab cannot pull in points from structures that merely
    happen to intersect the slab within the current radius.
    """
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
            work, plane, clicked, _expand_params(params, R), inplane_radius_mm=R
        )
        if len(idx) < 3:
            break
        plane = fit_plane_lsq(work[idx])

        nrm = np.asarray(plane.normal, dtype=np.float64)
        settled = float(np.abs(nrm @ prev_n)) > 0.999995  # ~0.18 deg
        prev_n = nrm

        at_cap = max_R is not None and R >= max_R - 1e-12
        if at_cap:
            break

        R_next = R + float(params.expand_step_mm)
        if max_R is not None:
            R_next = min(R_next, float(max_R))

        idx_next = _select_candidates(
            work, plane, clicked, _expand_params(params, R_next), inplane_radius_mm=R_next
        )
        if len(idx_next) <= len(idx):
            break
        if settled and len(idx_next) < int(1.05 * len(idx)):
            if max_R is not None:
                R = float(max_R)
                idx_cap = _select_candidates(
                    work, plane, clicked, _expand_params(params, R), inplane_radius_mm=R
                )
                if len(idx_cap) >= 3:
                    plane = fit_plane_lsq(work[idx_cap])
            break
        R = R_next

    return plane, float(R)


def _accumulate_with_refit(
    points: np.ndarray,
    clicked: np.ndarray,
    plane: Plane,
    params: PickParams,
) -> tuple[np.ndarray, Plane]:
    """Accumulate on the full cloud, refitting until the region stabilises.

    The seed normal is never exact. A single slab pass with a tilted normal
    selects a diagonal band across a wide face; refitting on that band
    recovers the true plane (the band still lies on the real surface), so
    one or two extra passes turn the band into the whole face.

    The region is bounded by connectivity, not by a radius: capping it here
    would split one physical face across several picks.
    """
    cap = params.max_inplane_radius_mm
    # Connectivity on tens of millions of points dominates runtime.
    # Do broad slab/refit iterations first, then apply connectivity once.
    idx = _select_candidates(
        points,
        plane,
        clicked,
        params,
        inplane_radius_mm=cap,
        connect=False,
    )
    if len(idx) == 0:
        raise ValueError("accumulation selected no points")

    for _ in range(max(0, int(params.final_refit_rounds))):
        if len(idx) < 3:
            break
        new_plane = fit_plane_lsq(points[idx])
        new_idx = _select_candidates(
            points,
            new_plane,
            clicked,
            params,
            inplane_radius_mm=cap,
            connect=False,
        )
        if len(new_idx) < 3:
            break
        grew = abs(len(new_idx) - len(idx)) / max(len(idx), 1)
        plane, idx = new_plane, new_idx
        if grew < params.final_refit_tolerance:
            break

    if params.connect and len(idx):
        idx = _connected_indices(points, idx, plane, clicked, params)
    return idx, plane


def pick_plane_region(
    points: np.ndarray,
    clicked: np.ndarray,
    neighbor_idx: np.ndarray,
    params: PickParams = PickParams(),
) -> tuple[np.ndarray, Plane]:
    """Extract a plane-candidate region around a click.

    Parameters
    ----------
    points : (N, 3) full cloud
    clicked : (3,) clicked world position
    neighbor_idx : indices of ``points`` within ``local_radius_mm`` of the
        click (computed by the caller, e.g. with a KDTree)

    Returns
    -------
    (indices, coarse_plane): indices into ``points`` of the accumulated
    region, and the (possibly refined) plane used for accumulation.
    """
    points = np.asarray(points, dtype=np.float64)
    neighbor_idx = np.asarray(neighbor_idx, dtype=np.int64)
    clicked = np.asarray(clicked, dtype=np.float64)

    plane = _fit_local_plane(points[neighbor_idx], params)
    plane, _final_R = _progressive_refine_plane(points, clicked, plane, params)
    return _accumulate_with_refit(points, clicked, plane, params)
