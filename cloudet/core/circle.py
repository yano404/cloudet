"""Planar circle fitting (project to plane UV, then 2D RANSAC / LSQ).

Circle model: center ``c``, plane unit normal ``n``, diameter ``Φ`` (mm).
Residual is the distance in the plane from ``c`` to the projected point,
minus ``Φ/2`` (plus a small out-of-plane penalty when scoring 3D points).

API / JSON use **diameter_mm**. With ``diameter_fixed``, only the center
(and optionally the supporting plane) is estimated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cloudet.core.cylinder import circle_from_three_points
from cloudet.core.plane import Plane, fit_plane_lsq, mad_sigma, residual_stats, robust_fit_plane

__all__ = [
    "Circle",
    "CircleFitResult",
    "fit_circle_2d_lsq",
    "ransac_circle",
    "robust_fit_circle",
]


def _sample_indices(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    size = int(min(size, n))
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if size >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=size, replace=False).astype(np.int64, copy=False)


def _uv_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _to_uv(points: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    w = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    return np.column_stack([w @ u, w @ v])


def fit_circle_2d_lsq(
    xy: np.ndarray,
    *,
    diameter_mm: float | None = None,
    diameter_fixed: bool = False,
) -> tuple[np.ndarray, float]:
    """Algebraic 2D circle fit. Returns ``(center_xy, radius)``."""
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2)")
    if len(pts) < 3:
        raise ValueError("need at least 3 points")

    if diameter_fixed:
        if diameter_mm is None or float(diameter_mm) <= 0:
            raise ValueError("diameter_fixed requires diameter_mm > 0")
        r = 0.5 * float(diameter_mm)
        # Minimize sum (||x-c||^2 - r^2)^2 ≈ geometric; use centroid of
        # points scaled toward circle: geometric center ≈ mean for uniform arc.
        # Better: solve for center with fixed r via iterative update.
        c = pts.mean(axis=0)
        for _ in range(20):
            d = pts - c
            dist = np.linalg.norm(d, axis=1)
            # Avoid zero
            dist = np.maximum(dist, 1e-12)
            # Move center so mean radial error is zero.
            c = c + np.mean(((dist - r) / dist)[:, None] * d, axis=0)
        return c, r

    # Kåsa / algebraic fit: x^2+y^2 + D x + E y + F = 0
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(pts))])
    b = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError as e:
        raise ValueError("degenerate circle fit") from e
    d, e, f = sol
    cx = -0.5 * d
    cy = -0.5 * e
    r2 = cx * cx + cy * cy - f
    if not np.isfinite(r2) or r2 <= 0:
        raise ValueError("degenerate circle fit")
    return np.array([cx, cy], dtype=np.float64), float(np.sqrt(r2))


@dataclass(frozen=True)
class Circle:
    """Circle in 3D: center, plane normal, diameter (mm)."""

    center: np.ndarray
    normal: np.ndarray
    diameter_mm: float
    diameter_fixed: bool = False

    def __post_init__(self):
        c = np.asarray(self.center, dtype=np.float64).reshape(3)
        n = np.asarray(self.normal, dtype=np.float64).reshape(3)
        nn = np.linalg.norm(n)
        if not np.isfinite(nn) or nn == 0.0:
            raise ValueError("circle normal must be finite and non-zero")
        n = n / nn
        k = int(np.argmax(np.abs(n)))
        if n[k] < 0:
            n = -n
        diam = float(self.diameter_mm)
        if not np.isfinite(diam) or diam <= 0.0:
            raise ValueError("diameter_mm must be finite and > 0")
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "normal", n)
        object.__setattr__(self, "diameter_mm", diam)
        object.__setattr__(self, "diameter_fixed", bool(self.diameter_fixed))

    @property
    def radius_mm(self) -> float:
        return 0.5 * self.diameter_mm

    @property
    def plane(self) -> Plane:
        return Plane(self.normal, float(-self.normal @ self.center))

    def residuals(self, points: np.ndarray) -> np.ndarray:
        """In-plane radial residual (ignores out-of-plane for the signed value)."""
        pts = np.asarray(points, dtype=np.float64)
        u, v = _uv_basis(self.normal)
        xy = _to_uv(pts, self.center, u, v)
        return np.linalg.norm(xy, axis=1) - self.radius_mm

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.abs(self.residuals(points))


@dataclass
class CircleFitResult:
    circle: Circle
    inlier_mask: np.ndarray
    n_iterations: int
    converged: bool
    threshold: float
    stats_inliers: dict
    stats_all: dict
    status: str = "ok"
    reasons: list[str] | None = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []

    @property
    def n_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))


def ransac_circle(
    points: np.ndarray,
    threshold: float,
    n_iterations: int = 800,
    seed: int = 0,
    *,
    plane: Plane | None = None,
    diameter_mm: float | None = None,
    diameter_fixed: bool = False,
    diameter_tol_mm: float | None = None,
) -> tuple[Circle, np.ndarray]:
    """RANSAC circle in a supporting plane (fitted if ``plane`` is None)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 3:
        raise ValueError("need at least 3 points")
    fixed = bool(diameter_fixed)
    if fixed and (diameter_mm is None or float(diameter_mm) <= 0):
        raise ValueError("diameter_fixed requires diameter_mm > 0")

    if plane is None:
        plane = fit_plane_lsq(pts)
    u, v = _uv_basis(plane.normal)
    origin = -plane.d * plane.normal
    xy = _to_uv(pts, origin, u, v)

    thresh = float(threshold)
    diam_tol = float(2.0 * thresh if diameter_tol_mm is None else diameter_tol_mm)
    fixed_diam = None if diameter_mm is None else float(diameter_mm)
    rng = np.random.default_rng(int(seed))
    best_count = -1
    best: Circle | None = None
    best_mask = np.zeros(len(pts), dtype=bool)

    for _ in range(max(1, int(n_iterations))):
        idx = _sample_indices(len(pts), 3, rng)
        # Lift three UV samples back... easier use 3D circumcircle and check
        # normal alignment with plane.
        circ = circle_from_three_points(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if circ is None:
            continue
        center, normal, radius = circ
        if abs(float(normal @ plane.normal)) < 0.85:
            continue
        hyp_diam = 2.0 * radius
        if fixed:
            assert fixed_diam is not None
            if abs(hyp_diam - fixed_diam) > diam_tol:
                continue
            diam = fixed_diam
        else:
            diam = hyp_diam
        # Project center onto supporting plane.
        center = center - plane.signed_distances(center.reshape(1, 3))[0] * plane.normal
        cir = Circle(
            center=center,
            normal=plane.normal,
            diameter_mm=diam,
            diameter_fixed=fixed,
        )
        # Combined residual: radial + out-of-plane
        radial = cir.distances(pts)
        plane_d = plane.distances(pts)
        mask = (radial <= thresh) & (plane_d <= thresh)
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best = cir
            best_mask = mask

    if best is None:
        c_xy, r = fit_circle_2d_lsq(
            xy, diameter_mm=diameter_mm, diameter_fixed=fixed
        )
        center = origin + c_xy[0] * u + c_xy[1] * v
        best = Circle(
            center=center,
            normal=plane.normal,
            diameter_mm=2.0 * r if not fixed else float(fixed_diam),
            diameter_fixed=fixed,
        )
        radial = best.distances(pts)
        plane_d = plane.distances(pts)
        best_mask = (radial <= thresh) & (plane_d <= thresh)

    return best, best_mask


def robust_fit_circle(
    points: np.ndarray,
    threshold: float | None = None,
    sigma_factor: float = 3.0,
    max_iterations: int = 40,
    *,
    init: Circle | None = None,
    plane: Plane | None = None,
    diameter_mm: float | None = None,
    diameter_fixed: bool = False,
    ransac_iterations: int = 600,
    seed: int = 0,
    min_inlier_fraction: float = 0.05,
) -> CircleFitResult:
    """Fit a circle: supporting plane + 2D circle (free or fixed diameter)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 3:
        raise ValueError("need at least 3 points")

    fixed = bool(diameter_fixed)
    if fixed and (diameter_mm is None or float(diameter_mm) <= 0):
        raise ValueError("diameter_fixed requires diameter_mm > 0")

    reasons: list[str] = []
    status = "ok"

    if plane is None:
        if init is not None:
            plane = init.plane
        else:
            pr = robust_fit_plane(pts, threshold=threshold, seed=seed)
            plane = pr.plane

    if init is None:
        boot = 1.0 if threshold is None else float(threshold)
        cir0, mask0 = ransac_circle(
            pts,
            boot,
            n_iterations=ransac_iterations,
            seed=seed,
            plane=plane,
            diameter_mm=diameter_mm,
            diameter_fixed=fixed,
        )
        n_ransac = ransac_iterations
    else:
        cir0 = Circle(
            center=init.center,
            normal=plane.normal,
            diameter_mm=float(diameter_mm)
            if (fixed and diameter_mm is not None)
            else float(init.diameter_mm),
            diameter_fixed=fixed,
        )
        mask0 = np.ones(len(pts), dtype=bool)
        n_ransac = 0

    cir = cir0
    mask = mask0
    thresh = float(threshold) if threshold is not None else None
    converged = False
    u, v = _uv_basis(plane.normal)

    for it in range(int(max_iterations)):
        if int(np.count_nonzero(mask)) < 3:
            reasons.append("too_few_inliers")
            status = "fail"
            break
        inliers = pts[mask]
        # Refit plane lightly on inliers
        try:
            plane = fit_plane_lsq(inliers)
        except ValueError:
            pass
        u, v = _uv_basis(plane.normal)
        origin = -plane.d * plane.normal
        xy = _to_uv(inliers, origin, u, v)
        try:
            c_xy, r = fit_circle_2d_lsq(
                xy, diameter_mm=diameter_mm, diameter_fixed=fixed
            )
        except ValueError:
            reasons.append("circle_fit_failed")
            status = "fail"
            break
        center = origin + c_xy[0] * u + c_xy[1] * v
        cir = Circle(
            center=center,
            normal=plane.normal,
            diameter_mm=2.0 * r,
            diameter_fixed=fixed,
        )
        radial = cir.residuals(pts)
        plane_d = plane.signed_distances(pts)
        # Use radial residual for adaptive threshold; gate with plane distance.
        if thresh is None:
            mad = mad_sigma(radial[mask] if np.any(mask) else radial, seed=seed + it)
            if not np.isfinite(mad) or mad <= 0:
                mad = float(np.median(np.abs(radial))) + 1e-9
            cur_thresh = float(sigma_factor * mad)
        else:
            cur_thresh = float(thresh)
        new_mask = (np.abs(radial) <= cur_thresh) & (np.abs(plane_d) <= cur_thresh)
        if np.array_equal(new_mask, mask):
            converged = True
            mask = new_mask
            thresh = cur_thresh
            break
        mask = new_mask
        thresh = cur_thresh
    else:
        thresh = float(thresh if thresh is not None else 1.0)

    if thresh is None:
        thresh = 1.0

    frac = float(np.count_nonzero(mask)) / float(len(pts))
    if frac < float(min_inlier_fraction):
        reasons.append("low_inlier_fraction")
        status = "suspect" if status == "ok" else status

    resid_all = cir.residuals(pts)
    resid_in = resid_all[mask] if np.any(mask) else resid_all[:0]
    return CircleFitResult(
        circle=cir,
        inlier_mask=mask,
        n_iterations=n_ransac + int(max_iterations),
        converged=converged,
        threshold=float(thresh),
        stats_inliers=residual_stats(resid_in),
        stats_all=residual_stats(resid_all),
        status=status,
        reasons=reasons,
    )
