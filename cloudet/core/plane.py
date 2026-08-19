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

from cloudet.array_backend import ArrayContext, DevicePoints, cupy_available, get_context

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
    "RANSAC_BACKENDS",
    "normalize_ransac_backend",
]

_LSQ_MAX_POINTS = 300_000
_MAD_MAX_SAMPLES = 100_000


def _sample_indices(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``size`` distinct indices from ``0..n-1`` without huge ``choice(n, ...)``."""
    size = int(min(size, n))
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if size >= n:
        return np.arange(n, dtype=np.int64)
    if size * 8 < n:
        buf = np.empty(0, dtype=np.int64)
        while len(buf) < size:
            need = size - len(buf)
            draw = rng.integers(0, n, size=max(need * 2, need + 1024), dtype=np.int64)
            buf = np.unique(np.concatenate([buf, draw]))
        rng.shuffle(buf)
        return buf[:size]
    return rng.choice(n, size=size, replace=False).astype(np.int64, copy=False)


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
        import cupy as cp

        if isinstance(signed_residuals, cp.ndarray):
            r = signed_residuals.ravel()
        else:
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
        return _plane_from_scatter(points)
    xp = ctx.xp
    pts = ctx.to_device(points)
    return fit_plane_lsq_device(pts, ctx)


def _plane_from_scatter(points: np.ndarray) -> Plane:
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    centroid = points.mean(axis=0)
    cov = points.T @ points
    cov -= n * np.outer(centroid, centroid)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    if not np.isfinite(eigvals).all() or eigvals[0] < -1e-9 * max(eigvals[-1], 1.0):
        raise ValueError("degenerate point configuration")
    d = -float(normal @ centroid)
    return Plane(normal, d)


def fit_plane_lsq_device(pts_dev, ctx: ArrayContext, mask=None) -> Plane:
    """LSQ plane from points already on device (optional boolean ``mask``)."""
    xp = ctx.xp
    sub = pts_dev if mask is None else pts_dev[mask]
    n = int(sub.shape[0])
    if n < 3:
        raise ValueError("need at least 3 points")
    centroid = sub.mean(axis=0)
    cov = sub.T @ sub - n * xp.outer(centroid, centroid)
    cov_np = ctx.asnumpy(cov)
    cen_np = ctx.asnumpy(centroid)
    eigvals, eigvecs = np.linalg.eigh(cov_np)
    normal = eigvecs[:, 0]
    if not np.isfinite(eigvals).all() or eigvals[0] < -1e-9 * max(eigvals[-1], 1.0):
        raise ValueError("degenerate point configuration")
    d = -float(normal @ cen_np)
    return Plane(normal, d)


def ransac_plane(
    points: np.ndarray,
    threshold: float,
    n_iterations: int = 1000,
    seed: int = 0,
    n_hypo_points: int | None = 100_000,
    *,
    return_inlier_mask: bool = True,
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
    if not return_inlier_mask:
        return plane, np.empty(0, dtype=bool)
    inlier_mask = plane.distances(points) <= threshold
    return plane, inlier_mask


def ransac_plane_cupy(
    points: np.ndarray,
    threshold: float,
    n_iterations: int,
    seed: int,
    n_hypo_points: int | None,
    ctx: ArrayContext,
    *,
    pts_dev=None,
    return_inlier_mask: bool = True,
) -> tuple[Plane, np.ndarray]:
    """GPU-batched inlier scoring for plane RANSAC (selector only)."""
    xp = ctx.xp
    points = np.asarray(points, dtype=np.float64)
    pts = pts_dev if pts_dev is not None else ctx.to_device(points)
    rng = np.random.default_rng(seed)
    n = len(points)
    if n_hypo_points is not None and n > n_hypo_points:
        hypo_idx = rng.choice(n, size=n_hypo_points, replace=False)
        hypo_pts = pts[hypo_idx]
        hypo_np = points[hypo_idx]
    else:
        hypo_pts = pts
        hypo_np = points

    samples = hypo_np[rng.integers(0, len(hypo_np), size=(n_iterations, 3))]
    v1 = samples[:, 1] - samples[:, 0]
    v2 = samples[:, 2] - samples[:, 0]
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-12
    if not np.any(valid):
        raise ValueError("RANSAC failed: no valid hypothesis")
    normals[valid] /= norms[valid, None]
    ds = -np.einsum("ij,ij->i", normals, samples[:, 0])

    valid_idx = np.flatnonzero(valid)
    normals_g = xp.asarray(normals[valid], dtype=xp.float64)
    ds_g = xp.asarray(ds[valid], dtype=xp.float64)
    dists = xp.abs(hypo_pts @ normals_g.T + ds_g)
    counts = (dists <= threshold).sum(axis=0)
    best_j = int(xp.argmax(counts).item())
    best_i = int(valid_idx[best_j])

    plane = Plane(normals[best_i], float(ds[best_i]))
    if not return_inlier_mask:
        return plane, np.empty(0, dtype=bool)
    n_g = xp.asarray(plane.normal, dtype=xp.float64)
    inlier_mask = xp.abs(pts @ n_g + plane.d) <= threshold
    return plane, ctx.asbool(inlier_mask)


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


RANSAC_BACKENDS = ("seeded", "seeded_cpu", "open3d")


def normalize_ransac_backend(backend: str | None) -> str:
    """Map UI / settings values to a canonical RANSAC backend name.

    Legacy ``numpy`` is treated as ``seeded`` (GPU preferred when available).
    """
    name = (backend or "seeded").lower()
    if name == "numpy":
        return "seeded"
    if name in RANSAC_BACKENDS:
        return name
    raise ValueError(
        f"unknown RANSAC backend {backend!r} (choose from {RANSAC_BACKENDS})"
    )


def run_ransac(
    points: np.ndarray,
    threshold: float,
    n_iterations: int = 1000,
    seed: int = 0,
    backend: str = "seeded",
    n_hypo_points: int | None = 100_000,
    *,
    compute_backend: str = "auto",
    device_points: DevicePoints | None = None,
    return_inlier_mask: bool = True,
) -> tuple[Plane, np.ndarray]:
    """Dispatch plane RANSAC to the selected backend.

    Both backends are selectors only; the final plane always comes from
    :func:`robust_fit_plane` (orthogonal least squares), so the backend
    choice affects only which points seed the refit.

    ``seeded`` prefers GPU scoring when CuPy is available (independent of
    the caller's ``compute_backend`` when a device buffer is absent).
    ``seeded_cpu`` always uses NumPy. ``open3d`` uses Open3D on CPU.
    Legacy ``numpy`` is an alias for ``seeded``.
    """
    backend = normalize_ransac_backend(backend)

    if backend == "seeded_cpu":
        return ransac_plane(
            points,
            threshold,
            n_iterations,
            seed,
            n_hypo_points,
            return_inlier_mask=return_inlier_mask,
        )

    if backend == "open3d":
        return ransac_plane_open3d(points, threshold, n_iterations, seed)

    # seeded: prefer GPU
    if device_points is not None and device_points.ctx.name == "cupy":
        return ransac_plane_cupy(
            points,
            threshold,
            n_iterations,
            seed,
            n_hypo_points,
            device_points.ctx,
            pts_dev=device_points.pts,
            return_inlier_mask=return_inlier_mask,
        )

    # Prefer cupy when available; fall back to caller compute / numpy.
    if cupy_available():
        try:
            ctx = get_context("cupy", n_points=len(points))
            return ransac_plane_cupy(
                points,
                threshold,
                n_iterations,
                seed,
                n_hypo_points,
                ctx,
                return_inlier_mask=return_inlier_mask,
            )
        except Exception:
            pass

    ctx = get_context(compute_backend, n_points=len(points))
    if ctx.name == "cupy":
        return ransac_plane_cupy(
            points,
            threshold,
            n_iterations,
            seed,
            n_hypo_points,
            ctx,
            return_inlier_mask=return_inlier_mask,
        )
    return ransac_plane(
        points,
        threshold,
        n_iterations,
        seed,
        n_hypo_points,
        return_inlier_mask=return_inlier_mask,
    )


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
    device_points: DevicePoints | None = None,
    fit_mask: np.ndarray | None = None,
    max_fit_points: int | None = _LSQ_MAX_POINTS,
    seed: int = 0,
) -> FitResult:
    """Iterative reweighted plane fit: LSQ -> reselect -> refit.

    Starting from ``init`` (or an LSQ fit of all points), points within
    the threshold are selected and refit until the inlier set is stable.
    If ``threshold`` is None it is chosen adaptively per iteration as
    ``sigma_factor * mad_sigma`` of the current residuals, which is
    robust against truncation bias.

    When the masked (or full) input exceeds ``max_fit_points``, the
    iterative refit runs on a seeded subsample and applies the final
    threshold once on the full input for the returned inlier mask.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        raise ValueError("need at least 3 points")
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape != (n,):
            raise ValueError("fit_mask must have shape (N,)")
        if np.count_nonzero(fit_mask) < 3:
            raise ValueError("need at least 3 points in fit_mask")

    if device_points is not None:
        ctx = device_points.ctx
        if ctx.name == "numpy":
            return _robust_fit_plane_cpu(
                points,
                threshold,
                sigma_factor,
                max_iterations,
                init,
                min_inlier_fraction,
                fit_mask=fit_mask,
                max_fit_points=max_fit_points,
                seed=seed,
            )
        return _robust_fit_plane_cupy(
            points,
            threshold,
            sigma_factor,
            max_iterations,
            init,
            min_inlier_fraction,
            ctx,
            pts_dev=device_points.pts,
            fit_mask=fit_mask,
            max_fit_points=max_fit_points,
            seed=seed,
        )

    ctx = get_context(compute_backend, n_points=n)
    if ctx.name == "numpy":
        return _robust_fit_plane_cpu(
            points,
            threshold,
            sigma_factor,
            max_iterations,
            init,
            min_inlier_fraction,
            fit_mask=fit_mask,
            max_fit_points=max_fit_points,
            seed=seed,
        )
    return _robust_fit_plane_cupy(
        points,
        threshold,
        sigma_factor,
        max_iterations,
        init,
        min_inlier_fraction,
        ctx,
        fit_mask=fit_mask,
        max_fit_points=max_fit_points,
        seed=seed,
    )


def _robust_fit_iterate_cpu(
    host: np.ndarray,
    *,
    threshold: float | None,
    sigma_factor: float,
    max_iterations: int,
    init: Plane | None,
    min_inlier_fraction: float,
    seed: int,
    lsq_max_points: int | None,
) -> tuple[Plane, float, bool, int]:
    n_fit = len(host)
    plane = init if init is not None else fit_plane_lsq(host, compute_backend="numpy")
    r = plane.signed_distances(host)
    mask = np.ones(n_fit, dtype=bool)
    rng = np.random.default_rng(int(seed))
    converged = False
    thr = threshold if threshold is not None else float("nan")

    for it in range(1, max_iterations + 1):
        if threshold is None:
            thr = sigma_factor * mad_sigma(
                r[mask], max_samples=_MAD_MAX_SAMPLES, seed=seed
            )
            if not np.isfinite(thr) or thr <= 0:
                raise ValueError("adaptive threshold collapsed (degenerate residuals)")
        new_mask = np.abs(r) <= thr
        if np.count_nonzero(new_mask) < max(3, int(min_inlier_fraction * n_fit)):
            raise ValueError(
                f"too few inliers ({int(np.count_nonzero(new_mask))}) "
                f"at threshold {thr:.6g}"
            )
        if np.array_equal(new_mask, mask) and it > 1:
            converged = True
            break
        mask = new_mask
        in_idx = np.flatnonzero(mask)
        if lsq_max_points is not None and len(in_idx) > lsq_max_points:
            in_idx = in_idx[_sample_indices(len(in_idx), lsq_max_points, rng)]
        plane = fit_plane_lsq(host[in_idx], compute_backend="numpy")
        r = plane.signed_distances(host)
    return plane, float(thr), converged, it


def _robust_fit_plane_cpu(
    points: np.ndarray,
    threshold: float | None,
    sigma_factor: float,
    max_iterations: int,
    init: Plane | None,
    min_inlier_fraction: float,
    fit_mask: np.ndarray | None = None,
    max_fit_points: int | None = _LSQ_MAX_POINTS,
    seed: int = 0,
) -> FitResult:
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if fit_mask is not None:
        fit_idx = np.flatnonzero(np.asarray(fit_mask, dtype=bool))
        host = points[fit_idx]
        n_fit = len(host)
    else:
        fit_idx = None
        host = points
        n_fit = n

    work = host
    if max_fit_points is not None and n_fit > int(max_fit_points):
        rng = np.random.default_rng(int(seed))
        work = host[_sample_indices(n_fit, int(max_fit_points), rng)]

    plane, thr, converged, it = _robust_fit_iterate_cpu(
        work,
        threshold=threshold,
        sigma_factor=sigma_factor,
        max_iterations=max_iterations,
        init=init,
        min_inlier_fraction=min_inlier_fraction,
        seed=seed,
        lsq_max_points=max_fit_points,
    )

    eval_r = plane.signed_distances(points)
    inlier = np.abs(eval_r) <= thr
    if fit_mask is not None:
        inlier &= np.asarray(fit_mask, dtype=bool)
        stats_all = residual_stats(eval_r[fit_mask])
    else:
        stats_all = residual_stats(eval_r)
    return FitResult(
        plane=plane,
        inlier_mask=inlier,
        n_iterations=it,
        converged=converged,
        threshold=float(thr),
        stats_inliers=residual_stats(eval_r[inlier]),
        stats_all=stats_all,
    )


def _robust_fit_iterate_cupy(
    work_pts,
    ctx: ArrayContext,
    *,
    threshold: float | None,
    sigma_factor: float,
    max_iterations: int,
    init: Plane | None,
    min_inlier_fraction: float,
    seed: int,
    lsq_max_points: int | None,
) -> tuple[Plane, float, bool, int]:
    xp = ctx.xp
    n_fit = int(work_pts.shape[0])
    if init is not None:
        plane = init
    else:
        plane = fit_plane_lsq_device(work_pts, ctx)
    normal = ctx.to_device(plane.normal)
    d = float(plane.d)
    mask = xp.ones(n_fit, dtype=bool)
    rng = np.random.default_rng(int(seed))
    converged = False
    thr = threshold if threshold is not None else float("nan")
    r = work_pts @ normal + d

    for it in range(1, max_iterations + 1):
        if threshold is None:
            thr = sigma_factor * mad_sigma(r[mask], max_samples=_MAD_MAX_SAMPLES, seed=seed, ctx=ctx)
            if not np.isfinite(thr) or thr <= 0:
                raise ValueError("adaptive threshold collapsed (degenerate residuals)")
        new_mask = xp.abs(r) <= thr
        n_in = int(xp.count_nonzero(new_mask).item())
        if n_in < max(3, int(min_inlier_fraction * n_fit)):
            raise ValueError(
                f"too few inliers ({n_in}) at threshold {thr:.6g}"
            )
        if xp.array_equal(new_mask, mask) and it > 1:
            converged = True
            break
        mask = new_mask
        n_lsq = int(xp.count_nonzero(mask).item())
        if lsq_max_points is not None and n_lsq > int(lsq_max_points):
            idx = xp.flatnonzero(mask)
            sel = xp.asarray(
                _sample_indices(n_lsq, int(lsq_max_points), rng), dtype=xp.int64
            )
            plane = fit_plane_lsq_device(work_pts[idx[sel]], ctx)
        else:
            plane = fit_plane_lsq_device(work_pts, ctx, mask=mask)
        normal = ctx.to_device(plane.normal)
        d = float(plane.d)
        r = work_pts @ normal + d
    return plane, float(thr), converged, it


def _robust_fit_plane_cupy(
    points: np.ndarray,
    threshold: float | None,
    sigma_factor: float,
    max_iterations: int,
    init: Plane | None,
    min_inlier_fraction: float,
    ctx: ArrayContext,
    *,
    pts_dev=None,
    fit_mask: np.ndarray | None = None,
    max_fit_points: int | None = _LSQ_MAX_POINTS,
    seed: int = 0,
) -> FitResult:
    xp = ctx.xp
    n = len(points)
    pts = pts_dev if pts_dev is not None else ctx.to_device(points)
    fit_mask_g = ctx.to_device_bool(fit_mask) if fit_mask is not None else None
    n_fit = int(xp.count_nonzero(fit_mask_g).item()) if fit_mask_g is not None else n

    if fit_mask_g is not None:
        fit_idx = xp.flatnonzero(fit_mask_g)
    else:
        fit_idx = None

    if max_fit_points is not None and n_fit > int(max_fit_points):
        rng = np.random.default_rng(int(seed))
        if fit_idx is not None:
            sel = xp.asarray(
                _sample_indices(n_fit, int(max_fit_points), rng), dtype=xp.int64
            )
            work_pts = pts[fit_idx[sel]]
        else:
            sel = xp.asarray(
                _sample_indices(n, int(max_fit_points), rng), dtype=xp.int64
            )
            work_pts = pts[sel]
    elif fit_idx is not None:
        work_pts = pts[fit_idx]
    else:
        work_pts = pts

    plane, thr, converged, it = _robust_fit_iterate_cupy(
        work_pts,
        ctx,
        threshold=threshold,
        sigma_factor=sigma_factor,
        max_iterations=max_iterations,
        init=init,
        min_inlier_fraction=min_inlier_fraction,
        seed=seed,
        lsq_max_points=max_fit_points,
    )

    normal = ctx.to_device(plane.normal)
    r = pts @ normal + plane.d
    inlier = xp.abs(r) <= thr
    if fit_mask_g is not None:
        inlier = inlier & fit_mask_g
    inlier_np = ctx.asbool(inlier)
    r_np = ctx.asnumpy(r)
    if fit_mask is not None:
        fit_mask_np = np.asarray(fit_mask, dtype=bool)
        stats_all = residual_stats(r_np[fit_mask_np])
    else:
        stats_all = residual_stats(r_np)
    return FitResult(
        plane=plane,
        inlier_mask=inlier_np,
        n_iterations=it,
        converged=converged,
        threshold=float(thr),
        stats_inliers=residual_stats(r_np[inlier_np]),
        stats_all=stats_all,
    )
