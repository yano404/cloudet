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

import detpos
from detpos.groups import GroupInfo, load_groups
from detpos.mainplane import MainPlaneParams, extract_main_plane
from detpos.multiplane import MultiPlaneParams, extract_planes
from detpos.plane import FitResult, Plane, ransac_plane, robust_fit_plane

__all__ = ["FitParams", "fit_group", "fit_groupset", "residual_uv_map"]


@dataclass(frozen=True)
class FitParams:
    """Parameters of the RANSAC + robust-refit pipeline (units: mm)."""

    ransac_threshold_mm: float = 0.5
    ransac_iterations: int = 1000
    seed: int = 0
    strict_threshold_mm: float | None = None  # None -> adaptive 3*mad_sigma
    sigma_factor: float = 3.0
    max_iterations: int = 100
    min_inlier_fraction: float = 0.05


def fit_group(points: np.ndarray, params: FitParams = FitParams()) -> FitResult:
    """RANSAC (selector) followed by robust orthogonal LSQ refit."""
    init, _ = ransac_plane(
        points,
        threshold=params.ransac_threshold_mm,
        n_iterations=params.ransac_iterations,
        seed=params.seed,
    )
    return robust_fit_plane(
        points,
        threshold=params.strict_threshold_mm,
        sigma_factor=params.sigma_factor,
        max_iterations=params.max_iterations,
        init=init,
        min_inlier_fraction=params.min_inlier_fraction,
    )


def residual_uv_map(
    points: np.ndarray,
    plane: Plane,
    mask: np.ndarray | None = None,
    bins: int = 200,
) -> dict:
    """Bin signed residuals on an in-plane (u, v) grid.

    Returns arrays suitable for plotting: per-bin mean signed residual
    and counts. Reveals spatial systematics (registration steps between
    scan passes, surface waviness, residual tilt) that histograms hide.
    """
    pts = points if mask is None else points[mask]
    r = plane.signed_distances(pts)

    n = plane.normal
    a = np.array([1.0, 0.0, 0.0])
    if abs(n @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    center = pts.mean(axis=0)
    uu = (pts - center) @ u
    vv = (pts - center) @ v

    counts, ue, ve = np.histogram2d(uu, vv, bins=bins)
    sums, _, _ = np.histogram2d(uu, vv, bins=[ue, ve], weights=r)
    with np.errstate(invalid="ignore"):
        mean_map = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    return {"mean": mean_map, "counts": counts, "u_edges": ue, "v_edges": ve}


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
            "software": {"detpos": detpos.__version__},
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
        "software": {"detpos": detpos.__version__},
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
