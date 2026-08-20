"""Sequential multi-plane extraction from a single group.

Rough picker groups can contain several planes, including parallel
surfaces only ~0.5 mm apart (visually inseparable when picking).
``extract_planes`` peels them off one at a time:

    extract_main_plane -> record -> remove its inliers -> repeat

To resolve close parallel planes the per-plane thresholds must be
small compared to the plane separation; the defaults here (RANSAC
0.1 mm, ceiling 0.15 mm) resolve >= ~0.4 mm separations with FARO-level
noise. A plane straddling two real surfaces reveals itself by a bimodal
signed-residual distribution -- reported via the ``bimodal`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cloudet.fit.mainplane import MainPlaneParams, MainPlaneResult, extract_main_plane

__all__ = ["MultiPlaneParams", "extract_planes", "bimodality_flag"]


@dataclass(frozen=True)
class MultiPlaneParams:
    """Units: mm. ``plane`` holds the per-plane extraction parameters
    (deliberately tighter than the single-plane defaults so that close
    parallel surfaces are resolved instead of merged)."""

    plane: MainPlaneParams = MainPlaneParams(
        ransac_threshold_mm=0.1,
        max_threshold_mm=0.15,
    )
    max_planes: int = 5
    min_plane_points: int = 5_000
    min_remaining_fraction: float = 0.02  # stop when the rest is this small
    accept_suspect: bool = True  # suspect planes are recorded (flagged), fail stops


def bimodality_flag(signed_residuals: np.ndarray, mad_sigma: float) -> bool:
    """Crude bimodality check: is there a second mode beyond 3*sigma?

    Detects a straddled pair of surfaces: the histogram of signed
    residuals then shows two peaks instead of one centred at zero.
    """
    r = signed_residuals
    if len(r) < 1000 or mad_sigma <= 0:
        return False
    hist, edges = np.histogram(r, bins=81, range=(-6 * mad_sigma, 6 * mad_sigma))
    if hist.max() == 0:
        return False
    peak = int(np.argmax(hist))
    # look for a second local maximum at least 2 sigma away and >30% of the main peak
    away = np.abs(0.5 * (edges[:-1] + edges[1:]) - 0.5 * (edges[peak] + edges[peak + 1]))
    candidates = np.flatnonzero((away > 2 * mad_sigma) & (hist > 0.3 * hist.max()))
    for c in candidates:
        lo, hi = max(c - 2, 0), min(c + 3, len(hist))
        if hist[c] == hist[lo:hi].max() and hist[c] >= hist[max(c - 3, 0):c + 1].min() + 0:
            return True
    return False


# Backward-compatible alias.
_bimodality_flag = bimodality_flag


def extract_planes(
    points: np.ndarray,
    params: MultiPlaneParams = MultiPlaneParams(),
    clicked: np.ndarray | None = None,
    coarse_plane: np.ndarray | None = None,
) -> list[dict]:
    """Extract all planes from a group, dominant first.

    Returns a list of dicts, one per plane, each with keys:
    ``plane_index``, ``result`` (MainPlaneResult), ``mask`` (bool mask on
    the input points), ``n_points``, ``bimodal``.
    The picker click is only used to anchor the first plane.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    remaining = np.ones(n, dtype=bool)
    planes: list[dict] = []

    for k in range(params.max_planes):
        n_rem = int(np.count_nonzero(remaining))
        if n_rem < max(params.min_plane_points, 3):
            break
        if n_rem < params.min_remaining_fraction * n:
            break

        sub = points[remaining]
        pp = params.plane
        if pp.min_points > params.min_plane_points:
            pp = replace(pp, min_points=params.min_plane_points)
        try:
            res: MainPlaneResult = extract_main_plane(
                sub,
                params=pp,
                clicked=clicked if k == 0 else None,
                coarse_plane=coarse_plane if k == 0 else None,
            )
        except ValueError:
            break

        n_in = res.n_main
        if n_in < params.min_plane_points:
            break
        if res.status == "fail" and not planes:
            # even the dominant surface is not a plane: report it and stop
            pass
        elif res.status == "fail":
            break

        mask = np.zeros(n, dtype=bool)
        mask[np.flatnonzero(remaining)] = res.main_mask

        r_signed = res.plane.signed_distances(points[mask])
        bimodal = bimodality_flag(
            r_signed, res.fit.stats_inliers["mad_sigma"]
        )

        planes.append({
            "plane_index": k,
            "result": res,
            "mask": mask,
            "n_points": int(np.count_nonzero(mask)),
            "bimodal": bool(bimodal),
        })

        if res.status == "fail":
            break
        remaining &= ~mask

    return planes
