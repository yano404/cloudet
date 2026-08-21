"""Infinite cylinder fitting (RANSAC + iterative refine).

Cylinder model: axis point ``p``, unit direction ``u``, diameter ``Φ`` (mm).
Radial residual for a point ``x`` is ``| |(x-p)×u| − Φ/2 |``.

API and on-disk fields use **diameter_mm** (Φ). Internally radius is ``Φ/2``.
When ``diameter_fixed`` is true, only the axis is estimated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cloudet.core.plane import mad_sigma, residual_stats

__all__ = [
    "Cylinder",
    "CylinderFitResult",
    "circle_from_three_points",
    "cylinder_from_three_points",
    "distances_to_axis",
    "ransac_cylinder",
    "refine_cylinder",
    "refine_cylinder_geometric",
    "robust_fit_cylinder",
]

# RANSAC inlier boot when ``threshold`` is None (adaptive refine afterward).
CYLINDER_DEFAULT_RANSAC_MM = 3.0
# Cap points used while scoring RANSAC hypotheses (full cloud scored once at end).
CYLINDER_RANSAC_MAX_SCORE_POINTS = 80_000
# Cap points kept in the cylinder seed ball before shell filtering.
CYLINDER_PICK_MAX_BALL_POINTS = 400_000


def _sample_indices(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    size = int(min(size, n))
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if size >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=size, replace=False).astype(np.int64, copy=False)


def distances_to_axis(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
) -> np.ndarray:
    """Perpendicular distance from each point to the infinite axis.

    Uses ``sqrt(|w|^2 - (w·u)^2)`` instead of ``np.cross`` to avoid a
    temporary ``(N, 3)`` allocation (important for large RANSAC clouds).
    """
    pts = np.asarray(points, dtype=np.float64, order="C")
    p = np.asarray(axis_point, dtype=np.float64).reshape(3)
    u = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    nu = np.linalg.norm(u)
    if not np.isfinite(nu) or nu == 0.0:
        raise ValueError("axis direction must be finite and non-zero")
    u = u / nu
    w = pts - p
    # |w × u|^2 = |w|^2 |u|^2 - (w·u)^2 with |u|=1
    w2 = np.einsum("ij,ij->i", w, w)
    wu = w @ u
    return np.sqrt(np.maximum(w2 - wu * wu, 0.0))


def circle_from_three_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Circumcircle of three non-collinear points.

    Returns ``(center, unit_normal, radius)`` or ``None`` if degenerate.
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    c = np.asarray(c, dtype=np.float64).reshape(3)
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    nlen = np.linalg.norm(normal)
    if not np.isfinite(nlen) or nlen < 1e-12:
        return None
    normal = normal / nlen
    # Circumcenter in plane: solve linear system in ab/ac basis.
    # Using the formula: perpendicular bisector intersection.
    ab2 = float(ab @ ab)
    ac2 = float(ac @ ac)
    abac = float(ab @ ac)
    denom = ab2 * ac2 - abac * abac
    if abs(denom) < 1e-18:
        return None
    # Center = a + α ab + β ac
    alpha = 0.5 * (ac2 * (ab2 - abac)) / denom
    beta = 0.5 * (ab2 * (ac2 - abac)) / denom
    center = a + alpha * ab + beta * ac
    radius = float(np.linalg.norm(center - a))
    if not np.isfinite(radius) or radius <= 0.0:
        return None
    return center, normal, radius


def cylinder_from_three_points(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    diameter_mm: float | None = None,
    diameter_fixed: bool = False,
) -> Cylinder | None:
    """Build a cylinder seed from three surface points (circumcircle axis).

    The three points should lie near one cross-section of the cylinder (not
    collinear). When ``diameter_fixed`` is true, the circumcircle supplies
    axis + center and ``diameter_mm`` locks Φ.
    """
    circ = circle_from_three_points(a, b, c)
    if circ is None:
        return None
    center, normal, radius = circ
    fixed = bool(diameter_fixed)
    if fixed:
        if diameter_mm is None or float(diameter_mm) <= 0:
            raise ValueError("diameter_fixed requires diameter_mm > 0")
        diam = float(diameter_mm)
    else:
        diam = float(diameter_mm) if diameter_mm is not None else 2.0 * radius
    if not np.isfinite(diam) or diam <= 0.0:
        return None
    return Cylinder(
        point=center,
        direction=normal,
        diameter_mm=diam,
        diameter_fixed=fixed,
    )


@dataclass(frozen=True)
class Cylinder:
    """Infinite cylinder: point on axis, unit direction, diameter (mm)."""

    point: np.ndarray  # (3,)
    direction: np.ndarray  # (3,) unit
    diameter_mm: float
    diameter_fixed: bool = False

    def __post_init__(self):
        p = np.asarray(self.point, dtype=np.float64).reshape(3)
        u = np.asarray(self.direction, dtype=np.float64).reshape(3)
        nu = np.linalg.norm(u)
        if not np.isfinite(nu) or nu == 0.0:
            raise ValueError("cylinder direction must be finite and non-zero")
        u = u / nu
        # Sign: largest-magnitude component of direction is non-negative.
        k = int(np.argmax(np.abs(u)))
        if u[k] < 0:
            u = -u
        diam = float(self.diameter_mm)
        if not np.isfinite(diam) or diam <= 0.0:
            raise ValueError("diameter_mm must be finite and > 0")
        object.__setattr__(self, "point", p)
        object.__setattr__(self, "direction", u)
        object.__setattr__(self, "diameter_mm", diam)
        object.__setattr__(self, "diameter_fixed", bool(self.diameter_fixed))

    @property
    def radius_mm(self) -> float:
        return 0.5 * self.diameter_mm

    def radial_distances(self, points: np.ndarray) -> np.ndarray:
        return distances_to_axis(points, self.point, self.direction)

    def residuals(self, points: np.ndarray) -> np.ndarray:
        """Signed radial residual: dist_to_axis − radius."""
        return self.radial_distances(points) - self.radius_mm

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.abs(self.residuals(points))


@dataclass
class CylinderFitResult:
    cylinder: Cylinder
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


def _axis_through_centroid(
    points: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit direction (signed) and axis point at the cloud centroid projection."""
    pts = np.asarray(points, dtype=np.float64)
    u = np.asarray(direction, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    centroid = pts.mean(axis=0)
    # Keep a point on the axis nearest the centroid.
    return centroid, u


def refine_cylinder(
    points: np.ndarray,
    init: Cylinder,
    *,
    max_iterations: int = 10,
    diameter_mm: float | None = None,
    diameter_fixed: bool | None = None,
    update_direction: bool = True,
) -> Cylinder:
    """Refine axis point and diameter; optionally refresh direction via PCA.

    Cross-section center comes from an algebraic 2D circle fit in the plane
    perpendicular to the current direction. Diameter uses that fit (or the
    median radial distance) unless fixed. When ``update_direction`` is true,
    each iteration also re-estimates the axis from the inlier covariance
    (largest-variance eigenvector), which helps after a good RANSAC seed.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 3:
        raise ValueError("need at least 3 points")

    fixed = init.diameter_fixed if diameter_fixed is None else bool(diameter_fixed)
    diam = float(init.diameter_mm if diameter_mm is None else diameter_mm)
    u = np.asarray(init.direction, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    p = np.asarray(init.point, dtype=np.float64).reshape(3).copy()

    for _ in range(int(max_iterations)):
        if update_direction and len(pts) >= 3:
            centered0 = pts - pts.mean(axis=0)
            cov = centered0.T @ centered0
            evals, evecs = np.linalg.eigh(cov)
            u_new = evecs[:, int(np.argmax(evals))]
            u_new = u_new / np.linalg.norm(u_new)
            if float(u_new @ u) < 0.0:
                u_new = -u_new
            u = u_new

        tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(u, tmp)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)

        centroid = pts.mean(axis=0)
        centered = pts - centroid
        xy = np.column_stack([centered @ e1, centered @ e2])
        x = xy[:, 0]
        y = xy[:, 1]
        A = np.column_stack([x, y, np.ones(len(xy))])
        bvec = -(x * x + y * y)
        try:
            sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
            d_coe, e_coe, f_coe = sol
            cx = -0.5 * d_coe
            cy = -0.5 * e_coe
            r2 = cx * cx + cy * cy - f_coe
            if np.isfinite(r2) and r2 > 0:
                c_xy = np.array([cx, cy], dtype=np.float64)
                if not fixed:
                    diam = 2.0 * float(np.sqrt(r2))
            else:
                c_xy = xy.mean(axis=0)
        except np.linalg.LinAlgError:
            c_xy = xy.mean(axis=0)
        p = centroid + c_xy[0] * e1 + c_xy[1] * e2
        if not fixed:
            med = float(np.median(distances_to_axis(pts, p, u)))
            if med > 0.0 and np.isfinite(med):
                diam = 0.5 * (diam + 2.0 * med)

    return Cylinder(point=p, direction=u, diameter_mm=diam, diameter_fixed=fixed)


def _cylinder_tangents(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal pair perpendicular to unit direction ``u``."""
    tmp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


def refine_cylinder_geometric(
    points: np.ndarray,
    init: Cylinder,
    *,
    max_iterations: int = 25,
    diameter_mm: float | None = None,
    diameter_fixed: bool | None = None,
    lock_direction: bool = False,
    damping: float = 1e-3,
    max_points: int = 50_000,
    seed: int = 0,
) -> Cylinder:
    """Nonlinear least-squares refine minimizing radial residuals ``ρ − r``.

    Unlike :func:`refine_cylinder` (algebraic 2D circle), this directly
    minimizes geometric distance to the cylinder surface. Better on short
    arcs / partial ducts. With ``lock_direction``, only the axis point (and
    optionally diameter) move in the plane perpendicular to the seed axis.
    """
    pts_all = np.asarray(points, dtype=np.float64, order="C")
    if pts_all.ndim != 2 or pts_all.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts_all) < 3:
        raise ValueError("need at least 3 points")

    if len(pts_all) > int(max_points):
        rng = np.random.default_rng(int(seed))
        pts = pts_all[
            rng.choice(len(pts_all), size=int(max_points), replace=False)
        ]
    else:
        pts = pts_all

    fixed = init.diameter_fixed if diameter_fixed is None else bool(diameter_fixed)
    if fixed and diameter_mm is not None and float(diameter_mm) <= 0:
        raise ValueError("diameter_fixed requires diameter_mm > 0")
    diam0 = float(init.diameter_mm if diameter_mm is None else diameter_mm)
    u = np.asarray(init.direction, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    p = np.asarray(init.point, dtype=np.float64).reshape(3).copy()
    centroid = pts.mean(axis=0)
    p = p + float((centroid - p) @ u) * u
    r = max(0.5 * diam0, 1e-6)

    # State: unit direction u, axis point p (near centroid), radius r.
    n = len(pts)
    lam = float(damping)
    eps = 1e-7

    def pack() -> np.ndarray:
        e1, e2 = _cylinder_tangents(u)
        a = float((p - centroid) @ e1)
        b = float((p - centroid) @ e2)
        if lock_direction:
            return (
                np.array([a, b], dtype=np.float64)
                if fixed
                else np.array([a, b, r], dtype=np.float64)
            )
        # Direction as small offsets in the current tangent frame (always ~0
        # after each successful step that reabsorbs into u).
        if fixed:
            return np.array([a, b, 0.0, 0.0], dtype=np.float64)
        return np.array([a, b, 0.0, 0.0, r], dtype=np.float64)

    def apply(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        e1, e2 = _cylinder_tangents(u)
        uu = u
        if not lock_direction:
            uu = u + float(x[2]) * e1 + float(x[3]) * e2
            nu = float(np.linalg.norm(uu))
            if nu < 1e-12:
                uu = u
            else:
                uu = uu / nu
            if float(uu @ u) < 0.0:
                uu = -uu
            e1, e2 = _cylinder_tangents(uu)
        pp = centroid + float(x[0]) * e1 + float(x[1]) * e2
        pp = pp + float((centroid - pp) @ uu) * uu
        rr = r if fixed else max(float(x[-1]), 1e-6)
        return pp, uu, rr

    def residual_vec(x: np.ndarray) -> np.ndarray:
        pp, uu, rr = apply(x)
        return distances_to_axis(pts, pp, uu) - rr

    x = pack()
    prev_cost = float(np.dot(residual_vec(x), residual_vec(x)))

    for _ in range(int(max_iterations)):
        res = residual_vec(x)
        pdim = len(x)
        J = np.empty((n, pdim), dtype=np.float64)
        for j in range(pdim):
            x2 = x.copy()
            step = eps * max(1.0, abs(float(x[j])))
            x2[j] += step
            J[:, j] = (residual_vec(x2) - res) / step
        JTJ = J.T @ J
        g = J.T @ res
        JTJ.flat[:: pdim + 1] += lam
        try:
            delta = -np.linalg.solve(JTJ, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)) or float(np.linalg.norm(delta)) < 1e-10:
            break
        x_trial = x + delta
        if not lock_direction:
            x_trial[2] = float(np.clip(x_trial[2], -0.4, 0.4))
            x_trial[3] = float(np.clip(x_trial[3], -0.4, 0.4))
        if not fixed:
            x_trial[-1] = max(float(x_trial[-1]), 1e-6)
        cost = float(np.dot(residual_vec(x_trial), residual_vec(x_trial)))
        if cost < prev_cost * (1.0 - 1e-12):
            p, u, r = apply(x_trial)
            prev_cost = cost
            lam = max(lam * 0.4, 1e-8)
            x = pack()  # reabsorb direction into u; reset d1,d2
        else:
            lam = min(lam * 2.5, 1e6)
            if lam >= 1e5:
                break

    return Cylinder(
        point=p,
        direction=u,
        diameter_mm=(2.0 * r) if not fixed else diam0,
        diameter_fixed=fixed,
    )


def ransac_cylinder(
    points: np.ndarray,
    threshold: float,
    n_iterations: int = 1000,
    seed: int = 0,
    *,
    diameter_mm: float | None = None,
    diameter_fixed: bool = False,
    diameter_tol_mm: float | None = None,
    min_diameter_mm: float = 1e-3,
    max_diameter_mm: float | None = None,
    max_score_points: int = CYLINDER_RANSAC_MAX_SCORE_POINTS,
) -> tuple[Cylinder, np.ndarray]:
    """RANSAC cylinder with **per-iteration axis sampling**.

    Each hypothesis takes three points, builds their circumcircle, and uses
    that circle's plane normal as the cylinder axis (same geometric idea as
    pyransac3d; reimplemented in-house). This avoids locking the axis to a
    single PCA of a contaminated seed ball.

    With ``diameter_fixed``, only hypotheses whose circum-diameter is near
    ``diameter_mm`` are kept (axis + center estimated).

    Hypotheses are scored on a random subsample (``max_score_points``) to
    bound memory/time; the winning model is then evaluated on all points.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 3:
        raise ValueError("need at least 3 points")
    if diameter_fixed and (diameter_mm is None or float(diameter_mm) <= 0):
        raise ValueError("diameter_fixed requires diameter_mm > 0")

    thresh = float(threshold)
    fixed = bool(diameter_fixed)
    fixed_diam = None if diameter_mm is None else float(diameter_mm)
    diam_tol = float(2.0 * thresh if diameter_tol_mm is None else diameter_tol_mm)
    min_d = float(min_diameter_mm)
    max_d = None if max_diameter_mm is None else float(max_diameter_mm)

    rng = np.random.default_rng(int(seed))
    n_all = len(pts)
    max_score = max(3, int(max_score_points))
    if n_all > max_score:
        score_idx = rng.choice(n_all, size=max_score, replace=False).astype(
            np.int64, copy=False
        )
        score_pts = pts[score_idx]
    else:
        score_idx = None
        score_pts = pts

    # PCA fallback axis (used only if no 3-point hypothesis succeeds).
    centroid = score_pts.mean(axis=0)
    centered = score_pts - centroid
    cov = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    u_pca = evecs[:, int(np.argmax(evals))]
    u_pca = u_pca / np.linalg.norm(u_pca)

    best_count = -1
    best_cyl: Cylinder | None = None
    n_iter = max(1, int(n_iterations))
    # Sample hypotheses from the scoring pool (same pool used for counting).
    n_score = len(score_pts)

    for _ in range(n_iter):
        idx = _sample_indices(n_score, 3, rng)
        circ = circle_from_three_points(
            score_pts[idx[0]], score_pts[idx[1]], score_pts[idx[2]]
        )
        if circ is None:
            continue
        center, normal, radius = circ
        hyp_diam = 2.0 * radius
        if hyp_diam < min_d or (max_d is not None and hyp_diam > max_d):
            continue
        if fixed:
            assert fixed_diam is not None
            if abs(hyp_diam - fixed_diam) > diam_tol:
                continue
            diam = fixed_diam
        else:
            diam = hyp_diam
        # Inline residual count without building a Cylinder each time.
        radial = distances_to_axis(score_pts, center, normal)
        count = int(np.count_nonzero(np.abs(radial - 0.5 * diam) <= thresh))
        if count > best_count:
            best_count = count
            best_cyl = Cylinder(
                point=center,
                direction=normal,
                diameter_mm=diam,
                diameter_fixed=fixed,
            )

    if best_cyl is None:
        dists = distances_to_axis(score_pts, centroid, u_pca)
        diam = (
            float(fixed_diam)
            if fixed and fixed_diam is not None
            else float(2.0 * np.median(dists))
        )
        if not np.isfinite(diam) or diam <= 0:
            diam = max(min_d, 1.0)
        best_cyl = Cylinder(
            point=centroid,
            direction=u_pca,
            diameter_mm=diam,
            diameter_fixed=fixed,
        )

    # One full-cloud evaluation for the winner only.
    best_mask = best_cyl.distances(pts) <= thresh
    return best_cyl, best_mask


def robust_fit_cylinder(
    points: np.ndarray,
    threshold: float | None = None,
    sigma_factor: float = 3.0,
    max_iterations: int = 40,
    *,
    init: Cylinder | None = None,
    diameter_mm: float | None = None,
    diameter_fixed: bool = False,
    ransac_iterations: int = 800,
    seed: int = 0,
    min_inlier_fraction: float = 0.05,
    max_refine_points: int = 200_000,
) -> CylinderFitResult:
    """RANSAC (optional) then iterative inlier refine for a cylinder.

    Pass ``diameter_mm`` with ``diameter_fixed=True`` to lock Φ (drawing
    diameter). Otherwise diameter is estimated from inliers.
    """
    pts = np.asarray(points, dtype=np.float64, order="C")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 3:
        raise ValueError("need at least 3 points")

    fixed = bool(diameter_fixed)
    if fixed and (diameter_mm is None or float(diameter_mm) <= 0):
        raise ValueError("diameter_fixed requires diameter_mm > 0")

    reasons: list[str] = []
    status = "ok"
    rng = np.random.default_rng(int(seed))

    if init is None:
        # Bootstrap threshold for RANSAC if adaptive.
        boot_thresh = (
            float(CYLINDER_DEFAULT_RANSAC_MM)
            if threshold is None
            else float(threshold)
        )
        cyl0, mask0 = ransac_cylinder(
            pts,
            boot_thresh,
            n_iterations=ransac_iterations,
            seed=seed,
            diameter_mm=diameter_mm,
            diameter_fixed=fixed,
        )
        n_ransac = ransac_iterations
        # RANSAC axis is noisy; allow PCA / geometric direction updates.
        lock_direction = False
    else:
        cyl0 = Cylinder(
            point=init.point,
            direction=init.direction,
            diameter_mm=float(diameter_mm)
            if (fixed and diameter_mm is not None)
            else float(init.diameter_mm),
            diameter_fixed=fixed,
        )
        mask0 = np.ones(len(pts), dtype=bool)
        n_ransac = 0
        # 3-point / user seeds: keep the axis. PCA on a short rim ring
        # otherwise replaces the cylinder axis with an in-plane direction.
        lock_direction = True

    cyl = cyl0
    mask = mask0
    thresh = float(threshold) if threshold is not None else None
    converged = False
    max_ref = max(3, int(max_refine_points))

    for it in range(int(max_iterations)):
        # Refine on a generous pool so the cross-section is well sampled.
        if thresh is None:
            resid0 = cyl.residuals(pts)
            mad0 = mad_sigma(resid0, seed=seed + it)
            if not np.isfinite(mad0) or mad0 <= 0:
                mad0 = float(np.median(np.abs(resid0))) + 1e-9
            pool_thresh = float(max(sigma_factor * mad0, 1.0))
        else:
            pool_thresh = float(max(3.0 * thresh, thresh))
        pool_mask = cyl.distances(pts) <= pool_thresh
        if int(np.count_nonzero(pool_mask)) < 3:
            pool_mask = np.ones(len(pts), dtype=bool)
        if int(np.count_nonzero(pool_mask)) < 3:
            reasons.append("too_few_inliers")
            status = "fail"
            break
        pool_idx = np.flatnonzero(pool_mask)
        if len(pool_idx) > max_ref:
            pool_idx = rng.choice(pool_idx, size=max_ref, replace=False)
        cyl = refine_cylinder(
            pts[pool_idx],
            cyl,
            diameter_mm=diameter_mm if fixed else None,
            diameter_fixed=fixed,
            update_direction=not lock_direction,
        )
        # Geometric polish: minimize ρ − r (helps short arcs / partial ducts).
        try:
            cyl = refine_cylinder_geometric(
                pts[pool_idx],
                cyl,
                diameter_mm=diameter_mm if fixed else None,
                diameter_fixed=fixed,
                lock_direction=lock_direction,
                seed=seed + it,
            )
        except (ValueError, np.linalg.LinAlgError):
            pass
        resid = cyl.residuals(pts)
        if thresh is None:
            mad = mad_sigma(resid[pool_idx], seed=seed + it)
            if not np.isfinite(mad) or mad <= 0:
                mad = float(np.median(np.abs(resid[pool_idx]))) + 1e-9
            cur_thresh = float(sigma_factor * mad)
        else:
            cur_thresh = float(thresh)
        new_mask = np.abs(resid) <= cur_thresh
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

    resid_all = cyl.residuals(pts)
    resid_in = resid_all[mask] if np.any(mask) else resid_all[:0]
    return CylinderFitResult(
        cylinder=cyl,
        inlier_mask=mask,
        n_iterations=n_ransac + int(max_iterations),
        converged=converged,
        threshold=float(thresh),
        stats_inliers=residual_stats(resid_in),
        stats_all=residual_stats(resid_all),
        status=status,
        reasons=reasons,
    )
