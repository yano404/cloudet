"""Click-driven plane-region extraction (GUI-independent logic).

Given a clicked 3D position on the cloud, fit a local plane to the
neighbourhood and accumulate all points close to that plane. Unlike the
legacy picker, accumulation is restricted to the in-plane connected
component containing the click, so coplanar-but-distant structures are
not swept in ("infinite slab" fix).

This module is pure numpy so it can be unit-tested without a GUI;
neighbour search is injected by the caller (the GUI uses Open3D's
KDTree, tests use brute force).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detpos.mainplane import _inplane_basis, _label_components
from detpos.plane import Plane, fit_plane_lsq, run_ransac

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
    ransac_backend: str = "numpy"  # "numpy" (seeded, reproducible) or "open3d"
    seed: int = 0


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
    region, and the local plane used for accumulation.
    """
    points = np.asarray(points, dtype=np.float64)
    neighbor_idx = np.asarray(neighbor_idx, dtype=np.int64)

    plane = _fit_local_plane(points[neighbor_idx], params)

    dists = plane.distances(points)
    candidate_idx = np.flatnonzero(dists <= params.accumulate_threshold_mm)
    if len(candidate_idx) == 0:
        raise ValueError("accumulation selected no points")

    if params.connect:
        candidate_idx = _connected_indices(points, candidate_idx, plane, clicked, params)

    return candidate_idx, plane
