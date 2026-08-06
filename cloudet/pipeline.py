"""Batch plane fitting pipeline: groups -> fits.

For each group this stage writes ``fit_xxx.json`` (plane, quality,
parameters, provenance incl. input SHA256) plus an overall
``fits_summary.csv``. Optionally a residual u-v map PNG per group
(requires matplotlib).

Everything needed to reproduce or invalidate a fit is baked into the
output; the pipeline never depends on a mutable settings file.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cloudet
from cloudet.groups import GroupInfo, load_groups
from cloudet.mainplane import MainPlaneParams, extract_main_plane
from cloudet.multiplane import MultiPlaneParams, extract_planes
from cloudet.plane import FitResult, Plane, robust_fit_plane, run_ransac

__all__ = ["FitParams", "fit_group", "fit_groupset", "residual_uv_map"]


@dataclass(frozen=True)
class FitParams:
    """Parameters of the RANSAC + robust-refit pipeline (units: mm)."""

    ransac_threshold_mm: float = 0.5
    ransac_iterations: int = 1000
    ransac_backend: str = "numpy"  # "numpy" (seeded, reproducible) or "open3d"
    seed: int = 0
    strict_threshold_mm: float | None = None  # None -> adaptive 3*mad_sigma
    sigma_factor: float = 3.0
    max_iterations: int = 100
    min_inlier_fraction: float = 0.05


def fit_group(points: np.ndarray, params: FitParams = FitParams()) -> FitResult:
    """RANSAC (selector) followed by robust orthogonal LSQ refit."""
    init, _ = run_ransac(
        points,
        threshold=params.ransac_threshold_mm,
        n_iterations=params.ransac_iterations,
        seed=params.seed,
        backend=params.ransac_backend,
    )
    return robust_fit_plane(
        points,
        threshold=params.strict_threshold_mm,
        sigma_factor=params.sigma_factor,
        max_iterations=params.max_iterations,
        init=init,
        min_inlier_fraction=params.min_inlier_fraction,
    )


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


def residual_uv_map(
    points: np.ndarray,
    plane: Plane,
    mask: np.ndarray | None = None,
    bins: int = 200,
    *,
    return_points: bool = False,
) -> dict:
    """Bin signed residuals on an in-plane (u, v) grid.

    The in-plane axes follow the face's minimum-area bounding rectangle so
    a rectangular patch appears axis-aligned in the map (an arbitrary seed
    basis otherwise leaves the face diagonally skewed; PCA alone can still
    tilt when sampling density is uneven).

    Returns arrays suitable for plotting: per-bin mean signed residual
    and counts. Reveals spatial systematics (registration steps between
    scan passes, surface waviness, residual tilt) that histograms hide.

    Always includes ``r`` (signed residuals), ``u_axis`` / ``v_axis`` /
    ``center``, and ``extents_uvn`` ``(lo, hi)`` in the (u, v, n) frame.
    If ``return_points`` is True, also include per-point ``u`` and ``v``.
    """
    pts = points if mask is None else points[mask]
    r = plane.signed_distances(pts)

    u, v, center, uu, vv = _aligned_inplane_basis(
        pts, plane.normal, return_coords=True
    )
    n_ax = np.asarray(plane.normal, dtype=np.float64)
    nn = (pts - center) @ n_ax

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


def _save_uv_map_png(uv: dict, path: Path, title: str, vlim_mm: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(
        uv["u_edges"],
        uv["v_edges"],
        uv["mean"].T * 1e3,  # um
        cmap="RdBu_r",
        vmin=-vlim_mm * 1e3,
        vmax=vlim_mm * 1e3,
    )
    fig.colorbar(im, ax=ax, label="mean signed residual [um]")
    ax.set_xlabel("u [mm]")
    ax.set_ylabel("v [mm]")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fit_record(group: GroupInfo, fit: FitResult | None, params) -> dict:
    if fit is None:
        return {
            "version": 1,
            "units": "mm",
            "group": {
                "id": group.group_id,
                "name": group.name,
                "ply_file": group.ply_path.name,
                "ply_sha256": group.sha256(),
                "n_points": group.num_points,
            },
            "params": asdict(params),
            "plane": None,
            "quality": None,
            "software": {"cloudet": cloudet.__version__},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    return {
        "version": 1,
        "units": "mm",
        "group": {
            "id": group.group_id,
            "name": group.name,
            "ply_file": group.ply_path.name,
            "ply_sha256": group.sha256(),
            "n_points": group.num_points,
        },
        "params": asdict(params),
        "plane": {
            "normal": fit.plane.normal.tolist(),
            "d_mm": fit.plane.d,
            "abcd": fit.plane.as_array().tolist(),
        },
        "quality": {
            "n_inliers": fit.n_inliers,
            "inlier_fraction": fit.n_inliers / group.num_points,
            "converged": fit.converged,
            "n_iterations": fit.n_iterations,
            "strict_threshold_mm": fit.threshold,
            "stats_inliers": fit.stats_inliers,
            "stats_all": fit.stats_all,
        },
        "software": {"cloudet": cloudet.__version__},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


_CSV_COLUMNS = [
    "group_id", "name", "plane_index", "n_planes", "status", "bimodal",
    "group_points", "plane_points", "converged", "nx", "ny", "nz", "d_mm",
    "strict_threshold_mm", "inlier_mad_sigma_mm", "inlier_std_mm",
    "inlier_p95_abs_mm",
]


def _plane_entry(index: int, fit: FitResult, n_points: int, status: str,
                 reasons: list, bimodal: bool) -> dict:
    si = fit.stats_inliers
    return {
        "plane_index": index,
        "plane": {
            "normal": fit.plane.normal.tolist(),
            "d_mm": fit.plane.d,
            "abcd": fit.plane.as_array().tolist(),
        },
        "n_points": int(n_points),
        "status": status,
        "reasons": reasons,
        "bimodal": bool(bimodal),
        "converged": fit.converged,
        "strict_threshold_mm": fit.threshold,
        "stats_inliers": si,
    }


def fit_groupset(
    groups_path: str | Path,
    out_dir: str | Path,
    params: FitParams | MainPlaneParams | MultiPlaneParams = MultiPlaneParams(),
    uv_maps: bool = True,
    uv_bins: int = 200,
    log=print,
) -> list[dict]:
    """Fit every group and write fit_xxx.json + fits_summary.csv to out_dir.

    Parameter type selects the method:
    - ``MultiPlaneParams`` (default): sequential multi-plane extraction;
      groups may contain several planes (e.g. parallel surfaces ~0.5 mm
      apart), all are recorded in the ``planes`` list, dominant first.
    - ``MainPlaneParams``: single main plane component + QC.
    - ``FitParams``: plain RANSAC + robust refit (no QC).
    """
    groups = load_groups(groups_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    rows = []
    for g in groups:
        pts = g.load_points()

        if isinstance(params, MultiPlaneParams):
            extracted = extract_planes(
                pts, params, clicked=g.clicked, coarse_plane=g.coarse_plane
            )
            plane_entries = [
                (
                    _plane_entry(
                        p["plane_index"], p["result"].fit, p["n_points"],
                        p["result"].status, p["result"].reasons, p["bimodal"],
                    ),
                    p["result"].fit, p["mask"],
                )
                for p in extracted
            ]
        elif isinstance(params, MainPlaneParams):
            res = extract_main_plane(
                pts, params=params, clicked=g.clicked, coarse_plane=g.coarse_plane
            )
            plane_entries = [(
                _plane_entry(0, res.fit, res.n_main, res.status, res.reasons, False),
                res.fit, res.fit.inlier_mask,
            )]
        else:
            fit = fit_group(pts, params)
            plane_entries = [(
                _plane_entry(0, fit, fit.n_inliers, "unchecked", [], False),
                fit, fit.inlier_mask,
            )]

        rec = _fit_record(g, plane_entries[0][1] if plane_entries else None, params)
        rec["version"] = 2
        rec["planes"] = [e[0] for e in plane_entries]
        rec["n_planes"] = len(plane_entries)
        records.append(rec)

        with open(out_dir / f"fit_{g.group_id:03d}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

        for entry, fit, mask in plane_entries:
            k = entry["plane_index"]
            if uv_maps:
                uv = residual_uv_map(pts, fit.plane, mask=mask, bins=uv_bins)
                suffix = "" if k == 0 else f"_p{k}"
                try:
                    _save_uv_map_png(
                        uv,
                        out_dir / f"fit_{g.group_id:03d}{suffix}_uvmap.png",
                        title=(
                            f"{g.name} plane {k} [{entry['status']}] mad_sigma="
                            f"{entry['stats_inliers']['mad_sigma']*1e3:.0f} um"
                        ),
                        vlim_mm=3 * entry["stats_inliers"]["mad_sigma"],
                    )
                except ImportError:
                    log("matplotlib not available; skipping u-v maps")
                    uv_maps = False
            si = entry["stats_inliers"]
            n = entry["plane"]["normal"]
            rows.append([
                g.group_id, g.name, k, len(plane_entries), entry["status"],
                entry["bimodal"], g.num_points, entry["n_points"],
                entry["converged"], *(round(x, 9) for x in n),
                round(entry["plane"]["d_mm"], 6),
                round(entry["strict_threshold_mm"], 6),
                round(si["mad_sigma"], 6), round(si["std"], 6),
                round(si["p95_abs"], 6),
            ])

        summary = " + ".join(
            f"p{e['plane_index']}:{e['n_points']:,}pts "
            f"{e['stats_inliers']['mad_sigma']*1e3:.0f}um {e['status']}"
            + (" BIMODAL" if e["bimodal"] else "")
            for e, _, _ in plane_entries
        ) or "no plane found"
        log(f"{g.name}: {g.num_points:,} pts -> {len(plane_entries)} plane(s): {summary}")

    with open(out_dir / "fits_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLUMNS)
        w.writerows(rows)

    return records
