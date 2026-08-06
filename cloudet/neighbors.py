"""Neighbour search and display decimation.

Replaces Open3D's KDTreeFlann in the GUI so that the Qt picker has no
hard Open3D dependency for picking. Display downsampling may optionally
use Open3D's C++ ``voxel_down_sample`` when installed (much faster on
tens of millions of points); rendering itself stays in Qt/PyVista.
"""

from __future__ import annotations

import numpy as np

from cloudet.array_backend import cupy_available, get_context

__all__ = [
    "VoxelHashGrid",
    "voxel_downsample_indices",
    "display_indices",
    "display_xyz",
    "resolve_display_backend",
    "depth_layers_along_ray",
]


class VoxelHashGrid:
    """Uniform-grid spatial index for fixed-radius neighbour queries.

    ``cell_size`` should be of the order of the query radius: queries
    scan the 27 (or more) cells overlapping the query ball.
    """

    def __init__(self, points: np.ndarray, cell_size: float):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.points = points
        self.cell_size = float(cell_size)
        self.origin = points.min(axis=0)

        ijk = np.floor((points - self.origin) / self.cell_size).astype(np.int64)
        self.dims = ijk.max(axis=0) + 1
        keys = (ijk[:, 0] * self.dims[1] + ijk[:, 1]) * self.dims[2] + ijk[:, 2]

        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        # unique cells with start offsets into `order`
        self._cell_keys, self._cell_starts = np.unique(sorted_keys, return_index=True)
        self._order = order

    def _cell_indices(self, key: int) -> np.ndarray:
        pos = np.searchsorted(self._cell_keys, key)
        if pos == len(self._cell_keys) or self._cell_keys[pos] != key:
            return np.empty(0, dtype=np.int64)
        start = self._cell_starts[pos]
        end = (
            self._cell_starts[pos + 1]
            if pos + 1 < len(self._cell_starts)
            else len(self._order)
        )
        return self._order[start:end]

    def radius_indices(self, center, radius: float) -> np.ndarray:
        """Indices of all points within ``radius`` of ``center``."""
        center = np.asarray(center, dtype=np.float64)
        lo = np.floor((center - radius - self.origin) / self.cell_size).astype(np.int64)
        hi = np.floor((center + radius - self.origin) / self.cell_size).astype(np.int64)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, self.dims - 1)
        if np.any(hi < lo):
            return np.empty(0, dtype=np.int64)

        chunks = []
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                base = (i * self.dims[1] + j) * self.dims[2]
                for k in range(lo[2], hi[2] + 1):
                    idx = self._cell_indices(base + k)
                    if len(idx):
                        chunks.append(idx)
        if not chunks:
            return np.empty(0, dtype=np.int64)
        cand = np.concatenate(chunks)
        d2 = np.einsum(
            "ij,ij->i", self.points[cand] - center, self.points[cand] - center
        )
        return cand[d2 <= radius * radius]


def _voxel_downsample_indices_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    ijk = np.floor((points - points.min(axis=0)) / voxel_size).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    keys = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, first = np.unique(keys, return_index=True)
    return np.sort(first)


def voxel_downsample_indices(
    points: np.ndarray,
    voxel_size: float,
    *,
    compute_backend: str = "auto",
) -> np.ndarray:
    """Indices of one representative point per occupied voxel."""
    points = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0:
        return np.arange(len(points))
    ctx = get_context(compute_backend, n_points=len(points))
    if ctx.name == "cupy":
        try:
            return _voxel_downsample_indices_cupy(points, voxel_size, ctx)
        except RuntimeError:
            return _voxel_downsample_indices_numpy(points, voxel_size)
    return _voxel_downsample_indices_numpy(points, voxel_size)


def _voxel_downsample_indices_cupy(
    points: np.ndarray, voxel_size: float, ctx
) -> np.ndarray:
    xp = ctx.xp
    pts = ctx.to_device(points)
    origin = pts.min(axis=0)
    ijk = xp.floor((pts - origin) / voxel_size).astype(xp.int64)
    dims = ijk.max(axis=0) + 1
    keys = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    keys_np = ctx.asnumpy(keys)
    _, first = np.unique(keys_np, return_index=True)
    return np.sort(first)


def display_indices(
    points: np.ndarray,
    voxel_size: float,
    max_points: int,
    seed: int = 0,
    *,
    compute_backend: str = "auto",
) -> np.ndarray:
    """Indices for display: optional voxel filter then a hard random cap."""
    idx = voxel_downsample_indices(
        points, voxel_size, compute_backend=compute_backend
    )
    if len(idx) > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    return idx


def resolve_display_backend(backend: str = "auto") -> str:
    """Return ``'cupy'``, ``'open3d'``, or ``'numpy'``."""
    backend = (backend or "auto").lower()
    if backend == "numpy":
        return "numpy"
    if backend == "cupy":
        if not cupy_available():
            raise ImportError(
                "display downsample backend 'cupy' requested but CuPy/CUDA "
                "is not available (pip install cupy-cuda12x, or use "
                "backend='auto'/'numpy'/'open3d')"
            )
        return "cupy"
    if backend == "open3d":
        try:
            import open3d  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "display downsample backend 'open3d' requested but open3d "
                "is not installed (pip install open3d, or use backend='numpy'/'auto')"
            ) from e
        return "open3d"
    if backend == "auto":
        if cupy_available():
            return "cupy"
        try:
            import open3d  # noqa: F401

            return "open3d"
        except ImportError:
            return "numpy"
    raise ValueError(
        f"unknown display downsample backend {backend!r} "
        "(choose from auto, numpy, open3d, cupy)"
    )


def _cap_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[keep]


def _display_xyz_open3d(
    points: np.ndarray, voxel_size: float, max_points: int, seed: int
) -> np.ndarray:
    import open3d as o3d

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points.reshape(0, 3)

    # Open3D voxel_down_sample is C++ and typically much faster than the
    # numpy unique path on tens of millions of points. It does not retain
    # original indices; that is fine for display-only geometry.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    out = np.asarray(pcd.points, dtype=np.float64)
    return _cap_points(out, int(max_points), seed)


def display_xyz(
    points: np.ndarray,
    voxel_size: float,
    max_points: int,
    seed: int = 0,
    backend: str = "auto",
) -> np.ndarray:
    """Return ``(M, 3)`` points for display (voxel + hard cap).

    ``backend``: ``auto`` (CuPy if available, else Open3D if installed,
    else numpy), ``numpy``, ``open3d``, or ``cupy``.
    """
    points = np.asarray(points, dtype=np.float64)
    resolved = resolve_display_backend(backend)
    if resolved == "open3d":
        return _display_xyz_open3d(points, voxel_size, max_points, seed)
    compute = "cupy" if resolved == "cupy" else "numpy"
    idx = display_indices(
        points,
        voxel_size,
        max_points,
        seed=seed,
        compute_backend=compute,
    )
    return points[idx]


def depth_layers_along_ray(
    points: np.ndarray,
    origin,
    direction,
    *,
    cylinder_radius_mm: float = 3.0,
    gap_mm: float | None = None,
    min_gap_mm: float = 8.0,
    min_points: int = 15,
) -> list[dict]:
    """Cluster points near a camera ray into depth layers (front → back).

    Used when several surfaces overlap in the 2D view: the VTK picker only
    returns the frontmost hit, so the GUI lets the user cycle these layers.

    ``gap_mm`` is the depth void that separates two layers. Left at ``None``
    it adapts to the cloud's own along-ray spacing, which grows with
    decimation and with how obliquely the surface is seen — a fixed
    threshold splits one such surface into several phantom layers.

    Returns a list of dicts with keys ``seed`` (representative point, front
    of the layer), ``depth_mm`` (mean distance from ``origin`` along the
    ray), and ``n_points``.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return []
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0 or not np.isfinite(norm):
        return []
    direction = direction / norm

    rel = points - origin
    t = rel @ direction
    radial = np.linalg.norm(rel - t[:, None] * direction, axis=1)
    keep = (t > 0.0) & (radial <= float(cylinder_radius_mm))
    if not np.any(keep):
        return []

    tt = t[keep]
    pts = points[keep]
    order = np.argsort(tt)
    tt = tt[order]
    pts = pts[order]

    if gap_mm is None:
        steps = np.diff(tt)
        # The median step describes sampling within a surface; a void between
        # surfaces is an order of magnitude larger and survives the median.
        typical = float(np.median(steps)) if len(steps) else 0.0
        threshold = max(float(min_gap_mm), 10.0 * typical)
    else:
        threshold = float(gap_mm)

    layers: list[dict] = []
    start = 0
    for i in range(1, len(tt) + 1):
        at_end = i == len(tt)
        gap = True if at_end else (tt[i] - tt[i - 1]) > threshold
        if not gap:
            continue
        chunk_t = tt[start:i]
        chunk_p = pts[start:i]
        start = i
        if len(chunk_t) < int(min_points):
            continue
        # Seed = point nearest the front of this layer (stable under occlusion).
        seed = chunk_p[0]
        layers.append(
            {
                "seed": np.asarray(seed, dtype=np.float64),
                "depth_mm": float(chunk_t.mean()),
                "n_points": int(len(chunk_t)),
            }
        )
    return layers
