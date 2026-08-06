"""High-precision plane fitting core.

Plane model: n . x + d = 0 with |n| = 1 (Hesse normal form).
All distances are signed distances along the unit normal, in the same
unit as the input coordinates (millimetres for FARO exports).

Design notes
------------
* RANSAC is used only to select points (outlier rejection); the final
  plane always comes from an orthogonal least-squares fit that is fully
  under our control (see ``robust_fit_plane``).
* Statistics are reported both for the inlier set (truncated
  distribution -- underestimates the true noise) and for the full input
  set, plus a truncation-robust sigma estimate (``mad_sigma``).
* All randomness is controlled by an explicit seed for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cloudet.array_backend import ArrayContext, get_context

__all__ = [
    "Plane",
    "FitResult",
    "fit_plane_lsq",
    "ransac_plane",
    "ransac_plane_open3d",
    "run_ransac",
    "robust_fit_plane",
    "residual_stats",
    "mad_sigma",
]


# ----------------------------------------------------------------------
# Plane model
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Plane:
    """Plane in Hesse normal form: normal . x + d = 0, |normal| = 1."""

    normal: np.ndarray  # shape (3,), unit length
    d: float

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=np.float64)
        norm = np.linalg.norm(n)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("plane normal must be finite and non-zero")
        n = n / norm
        # Sign convention: largest-magnitude component of the normal is
        # positive, so that the same physical plane always gets the same
        # representation.
        k = int(np.argmax(np.abs(n)))
        sign = 1.0 if n[k] >= 0 else -1.0
        object.__setattr__(self, "normal", sign * n)
        object.__setattr__(self, "d", float(sign * self.d / norm))

    def signed_distances(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return points @ self.normal + self.d

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.abs(self.signed_distances(points))

    def as_array(self) -> np.ndarray:
        """Return [a, b, c, d] with (a, b, c) the unit normal."""
        return np.array([*self.normal, self.d], dtype=np.float64)

    @staticmethod
    def from_array(abcd) -> "Plane":
        abcd = np.asarray(abcd, dtype=np.float64)
        if abcd.shape != (4,):
            raise ValueError("expected [a, b, c, d]")
        return Plane(abcd[:3], float(abcd[3]))

    def angle_to(self, other: "Plane") -> float:
        """Angle between plane normals in radians, in [0, pi/2]."""
        c = np.clip(np.abs(self.normal @ other.normal), -1.0, 1.0)
        return float(np.arccos(c))


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------


def mad_sigma(
    signed_residuals: np.ndarray,
    *,
    max_samples: int | None = None,
    seed: int = 0,
    ctx: ArrayContext | None = None,
) -> float:
    """Robust sigma estimate: 1.4826 * median absolute deviation.

    Insensitive to outliers and much less biased by threshold
    truncation than the plain standard deviation.

    If ``max_samples`` is set and the array is larger, MAD is estimated
    on a seeded subsample (for iterative loops). Final QC should leave
    ``max_samples=None``.
    """
    if ctx is not None and ctx.name == "cupy":
        xp = ctx.xp
        r = ctx.to_device(signed_residuals).ravel()
        if r.size == 0:
            return float("nan")
        if max_samples is not None and r.size > int(max_samples):
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(int(r.size), size=int(max_samples), replace=False)
            r = r[idx]
        med = xp.median(r)
        return float(1.4826 * xp.median(xp.abs(r - med)))

    r = np.asarray(signed_residuals, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    if max_samples is not None and r.size > int(max_samples):
        rng = np.random.default_rng(int(seed))
        r = r[rng.choice(r.size, size=int(max_samples), replace=False)]
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def residual_stats(
    signed_residuals: np.ndarray,
    *,
    lite: bool = False,
) -> dict:
    """Summary residual statistics.

    ``lite=True`` skips ``std`` / ``rms`` / ``p95_abs`` (keeps ``mad_sigma``
    and ``max_abs``) for hot paths that only need the robust scale.
    """
    r = np.asarray(signed_residuals, dtype=np.float64)
    if r.size == 0:
        return {"n": 0}
    a = np.abs(r)
    out = {
        "n": int(r.size),
        "mean": float(np.mean(r)),
        "mad_sigma": mad_sigma(r),
        "max_abs": float(np.max(a)),
    }
    if not lite:
        out["std"] = float(np.std(r))
        out["rms"] = float(np.sqrt(np.mean(r * r)))
        out["p95_abs"] = float(np.percentile(a, 95))
    return out


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------


def fit_plane_lsq(
    points: np.ndarray,
    *,
    compute_backend: str = "auto",
) -> Plane:
    """Orthogonal least-squares plane through ``points``.

    Minimises the sum of squared perpendicular distances (total least
    squares). The normal is the eigenvector of the 3x3 scatter matrix
    with the smallest eigenvalue; memory stays O(1) beyond the input.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(points) < 3:
        raise ValueError("need at least 3 points")

    ctx = get_context(compute_backend, n_points=len(points))
    if ctx.name == "numpy":
        n = len(points)
        centroid = points.mean(axis=0)
        cov = points.T @ points
        cov -= n * np.outer(centroid, centroid)
    else:
        xp = ctx.xp
        pts = ctx.to_device(points)
        n = len(pts)
        centroid = pts.mean(axis=0)
        cov = pts.T @ pts
        cov -= n * xp.outer(centroid, centroid)
        cov = ctx.asnumpy(cov)

    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # smallest eigenvalue
    if not np.isfinite(eigvals).all() or eigvals[0] < -1e-9 * max(eigvals[-1], 1.0):
        raise ValueError("degenerate point configuration")
    if ctx.name == "cupy":
        centroid = ctx.asnumpy(centroid)
    d = -float(normal @ centroid)
    return Plane(normal, d)


def ransac_plane(
    points: np.ndarray,
    threshold: float,
    n_iterations: int = 1000,
    seed: int = 0,
    n_hypo_points: int | None = 100_000,
) -> tuple[Plane, np.ndarray]:
    """Pure-numpy plane RANSAC (selector only -- refit afterwards).

    Hypotheses are generated from 3-point samples drawn from a random
    subset of at most ``n_hypo_points`` points (cheap), but inliers are
    counted on the full input (exact). Returns the best hypothesis
    plane and the boolean inlier mask of the *full* input against it.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("need at least 3 points")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    rng = np.random.default_rng(seed)
    if n_hypo_points is not None and len(points) > n_hypo_points:
        hypo_idx = rng.choice(len(points), size=n_hypo_points, replace=False)
        hypo_pts = points[hypo_idx]
    else:
        hypo_pts = points

    # Vectorised hypothesis generation: (n_iterations, 3, 3)
    samples = hypo_pts[rng.integers(0, len(hypo_pts), size=(n_iterations, 3))]
    v1 = samples[:, 1] - samples[:, 0]
    v2 = samples[:, 2] - samples[:, 0]
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-12
    normals[valid] /= norms[valid, None]
    ds = -np.einsum("ij,ij->i", normals, samples[:, 0])

    # Score on a subset for speed, then exact mask for the winner.
    score_pts = hypo_pts
    best_i, best_count = -1, -1
    for i in np.flatnonzero(valid):
        cnt = int(np.count_nonzero(np.abs(score_pts @ normals[i] + ds[i]) <= threshold))
        if cnt > best_count:
            best_count, best_i = cnt, i
    if best_i < 0:
        raise ValueError("RANSAC failed: no valid hypothesis")

    plane = Plane(normals[best_i], float(ds[best_i]))
    inlier_mask = plane.distances(points) <= threshold
    return plane, inlier_mask


def ransac_plane_open3d(
    points: np.ndarray,
    threshold: float,
    n_iterations: int = 1000,
    seed: int = 0,
) -> tuple[Plane, np.ndarray]:
    """Plane RANSAC via Open3D's ``segment_plane`` (selector only).

    Same interface as :func:`ransac_plane`. Note that Open3D refits the
    returned plane on its inliers internally; here the plane is used
    only as a seed for :func:`robust_fit_plane`, so both backends feed
    the identical final estimator. Requires open3d to be installed.
    """
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "RANSAC backend 'open3d' requested but open3d is not installed "
            "(pip install open3d, or use backend='numpy')"
        ) from e

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("need at least 3 points")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    try:
        o3d.utility.random.seed(int(seed))  # open3d >= 0.16
    except AttributeError:
        pass  # older open3d: not seedable, results not reproducible

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    model, inliers = pcd.segment_plane(
        distance_threshold=float(threshold),
        ransac_n=3,
        num_iterations=int(n_iterations),
    )
    plane = Plane.from_array(np.asarray(model, dtype=np.float64))
    mask = np.zeros(len(points), dtype=bool)
    mask[np.asarray(inliers, dtype=np.int64)] = True
    return plane, mask


RANSAC_BACKENDS = ("numpy", "open3d")


def run_ransac(
    points: np.ndarray,
    threshold: float,
    n_iterations: int = 1000,
    seed: int = 0,
    backend: str = "numpy",
    n_hypo_points: int | None = 100_000,
) -> tuple[Plane, np.ndarray]:
    """Dispatch plane RANSAC to the selected backend.

    Both backends are selectors only; the final plane always comes from
    :func:`robust_fit_plane` (orthogonal least squares), so the backend
    choice affects only which points seed the refit.
    """
    if backend == "numpy":
        return ransac_plane(points, threshold, n_iterations, seed, n_hypo_points)
    if backend == "open3d":
        return ransac_plane_open3d(points, threshold, n_iterations, seed)
    raise ValueError(f"unknown RANSAC backend {backend!r} (choose from {RANSAC_BACKENDS})")


@dataclass
class FitResult:
    plane: Plane
    inlier_mask: np.ndarray  # bool, shape (N,) on the input points
    n_iterations: int
    converged: bool
    threshold: float
    stats_inliers: dict  # truncated statistics (inliers only)
    stats_all: dict  # untruncated statistics (all input points)

    @property
    def n_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))


def robust_fit_plane(
    points: np.ndarray,
    threshold: float | None = None,
    sigma_factor: float = 3.0,
    max_iterations: int = 50,
    init: Plane | None = None,
    min_inlier_fraction: float = 0.1,
    *,
    compute_backend: str = "auto",
) -> FitResult:
    """Iterative reweighted plane fit: LSQ -> reselect -> refit.

    Starting from ``init`` (or an LSQ fit of all points), points within
    the threshold are selected and refit until the inlier set is stable.
    If ``threshold`` is None it is chosen adaptively per iteration as
    ``sigma_factor * mad_sigma`` of the current residuals, which is
    robust against truncation bias.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        raise ValueError("need at least 3 points")

    ctx = get_context(compute_backend, n_points=n)
    if ctx.name == "numpy":
        return _robust_fit_plane_cpu(
            points,
            threshold,
            sigma_factor,
            max_iterations,
            init,
            min_inlier_fraction,
        )
    return _robust_fit_plane_cupy(
        points,
        threshold,
        sigma_factor,
        max_iterations,
        init,
        min_inlier_fraction,
        ctx,
    )


def _robust_fit_plane_cpu(
    points: np.ndarray,
    threshold: float | None,
    sigma_factor: float,
    max_iterations: int,
    init: Plane | None,
    min_inlier_fraction: float,
) -> FitResult:
    n = len(points)
    plane = init if init is not None else fit_plane_lsq(points, compute_backend="numpy")
    mask = np.ones(n, dtype=bool)
    converged = False
    thr = threshold if threshold is not None else float("nan")
    r = plane.signed_distances(points)

    for it in range(1, max_iterations + 1):
        if threshold is None:
            thr = sigma_factor * mad_sigma(r[mask])
            if not np.isfinite(thr) or thr <= 0:
                raise ValueError("adaptive threshold collapsed (degenerate residuals)")
        new_mask = np.abs(r) <= thr
        if np.count_nonzero(new_mask) < max(3, int(min_inlier_fraction * n)):
            raise ValueError(
                f"too few inliers ({int(np.count_nonzero(new_mask))}) "
                f"at threshold {thr:.6g}"
            )
        if np.array_equal(new_mask, mask) and it > 1:
            converged = True
            break
        mask = new_mask
        plane = fit_plane_lsq(points[mask], compute_backend="numpy")
        r = plane.signed_distances(points)

    return FitResult(
        plane=plane,
        inlier_mask=mask,
        n_iterations=it,
        converged=converged,
        threshold=float(thr),
        stats_inliers=residual_stats(r[mask]),
        stats_all=residual_stats(r),
    )


def _robust_fit_plane_cupy(
    points: np.ndarray,
    threshold: float | None,
    sigma_factor: float,
    max_iterations: int,
    init: Plane | None,
    min_inlier_fraction: float,
    ctx: ArrayContext,
) -> FitResult:
    xp = ctx.xp
    n = len(points)
    pts = ctx.to_device(points)
    plane = init if init is not None else fit_plane_lsq(points, compute_backend="cupy")
    normal = ctx.to_device(plane.normal)
    d = float(plane.d)
    mask = xp.ones(n, dtype=bool)
    converged = False
    thr = threshold if threshold is not None else float("nan")
    r = pts @ normal + d

    for it in range(1, max_iterations + 1):
        if threshold is None:
            thr = sigma_factor * mad_sigma(r[mask], ctx=ctx)
            if not np.isfinite(thr) or thr <= 0:
                raise ValueError("adaptive threshold collapsed (degenerate residuals)")
        new_mask = xp.abs(r) <= thr
        n_in = int(xp.count_nonzero(new_mask))
        if n_in < max(3, int(min_inlier_fraction * n)):
            raise ValueError(
                f"too few inliers ({n_in}) at threshold {thr:.6g}"
            )
        if xp.array_equal(new_mask, mask) and it > 1:
            converged = True
            break
        mask = new_mask
        inlier_pts = ctx.asnumpy(pts[mask])
        plane = fit_plane_lsq(inlier_pts, compute_backend="cupy")
        normal = ctx.to_device(plane.normal)
        d = float(plane.d)
        r = pts @ normal + d

    mask_np = ctx.asbool(mask)
    r_np = ctx.asnumpy(r)
    return FitResult(
        plane=plane,
        inlier_mask=mask_np,
        n_iterations=it,
        converged=converged,
        threshold=float(thr),
        stats_inliers=residual_stats(r_np[mask_np]),
        stats_all=residual_stats(r_np),
    )
