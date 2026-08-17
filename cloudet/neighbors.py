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

_INDEX_CHUNK = 1_000_000
# Above this many grid cells the Python cell walk loses to one vectorized pass.
_MAX_QUERY_CELLS = 8_000
# Display voxel/Open3D must not see the full survey cloud: Vector3dVector /
# (N,3) int64 keys would duplicate tens of millions of points in RAM.
_DISPLAY_WORK_CAP = 8_000_000


def _as_xyz(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points


def _compact_ids(n_points: int, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Use int32 indices when N fits, to cut RAM/disk of the spatial index."""
    if n_points <= np.iinfo(np.int32).max:
        return tuple(np.asarray(a, dtype=np.int32) for a in arrays)
    return tuple(np.asarray(a, dtype=np.int64) for a in arrays)


def _precap_points(points: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Subset so display downsample never walks the full cloud.

    Uses a strided sample rather than ``choice(..., replace=False)``, which
    can permute all N indices in RAM.
    """
    n = len(points)
    cap = int(cap)
    if n <= cap:
        return points
    idx = np.linspace(0, n - 1, cap, dtype=np.int64)
    return points[idx]


def _keys_chunked(points: np.ndarray, origin: np.ndarray, cell_size: float):
    """Linear voxel keys without an (N, 3) int64 index (that alone is 8 bytes × 3N)."""
    n = len(points)
    inv = 1.0 / float(cell_size)
    mx = np.zeros(3, dtype=np.int64)
    for i in range(0, n, _INDEX_CHUNK):
        sl = points[i : i + _INDEX_CHUNK]
        ijk = np.floor((sl - origin) * inv).astype(np.int64)
        np.maximum(mx, ijk.max(axis=0), out=mx)
    dims = mx + 1
    keys = np.empty(n, dtype=np.int64)
    for i in range(0, n, _INDEX_CHUNK):
        sl = points[i : i + _INDEX_CHUNK]
        ijk = np.floor((sl - origin) * inv).astype(np.int64)
        keys[i : i + len(sl)] = (
            (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
        )
    return keys, dims


def _unique_runs(keys: np.ndarray, order: np.ndarray):
    """Cell keys / start offsets from argsort, without copying all keys."""
    n = len(order)
    if n == 0:
        empty = np.empty(0, dtype=keys.dtype)
        return empty, np.empty(0, dtype=np.int64)
    mask = np.empty(n, dtype=np.bool_)
    mask[0] = True
    for i in range(0, n - 1, _INDEX_CHUNK):
        j = min(i + _INDEX_CHUNK, n - 1)
        mask[i + 1 : j + 1] = keys[order[i + 1 : j + 1]] != keys[order[i:j]]
    starts = np.flatnonzero(mask)
    return keys[order[starts]], starts


def _build_voxel_index(points: np.ndarray, cell_size: float):
    origin = points.min(axis=0)
    keys, dims = _keys_chunked(points, origin, cell_size)
    order = np.argsort(keys, kind="quicksort")
    cell_keys, cell_starts = _unique_runs(keys, order)
    del keys
    order, cell_starts = _compact_ids(len(points), order, cell_starts)
    return origin, dims, cell_keys, cell_starts, order


class VoxelHashGrid:
    """Uniform-grid spatial index for fixed-radius neighbour queries.

    ``cell_size`` should be of the order of the query radius: queries
    scan the 27 (or more) cells overlapping the query ball.
    """

    def __init__(self, points: np.ndarray, cell_size: float):
        points = _as_xyz(points)
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.points = points
        self.cell_size = float(cell_size)
        (
            self.origin,
            self.dims,
            self._cell_keys,
            self._cell_starts,
            self._order,
        ) = _build_voxel_index(points, self.cell_size)

    @classmethod
    def from_arrays(
        cls,
        points: np.ndarray,
        *,
        cell_size: float,
        origin: np.ndarray,
        dims: np.ndarray,
        cell_keys: np.ndarray,
        cell_starts: np.ndarray,
        order: np.ndarray,
        validate_range: bool = True,
    ) -> "VoxelHashGrid":
        """Rebuild from arrays produced by :meth:`index_arrays` (no re-index).

        ``validate_range`` scans every index; skip it when loading a trusted
        cache so a 60 M-point memmap is not read twice.
        """
        points = _as_xyz(points)
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        order = np.asarray(order).reshape(-1)
        if order.dtype.kind not in "iu":
            raise ValueError("order must be integer")
        if len(order) != len(points):
            raise ValueError("order length must match number of points")
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        dims = np.asarray(dims, dtype=np.int64).reshape(3)
        cell_keys = np.asarray(cell_keys).reshape(-1)
        cell_starts = np.asarray(cell_starts).reshape(-1)
        if len(cell_keys) != len(cell_starts):
            raise ValueError("cell_keys and cell_starts length mismatch")
        if validate_range and order.size:
            if int(order.min()) < 0 or int(order.max()) >= len(points):
                raise ValueError("order contains out-of-range indices")
            if not np.allclose(origin, points.min(axis=0)):
                raise ValueError("cached origin does not match points")
        obj = cls.__new__(cls)
        obj.points = points
        obj.cell_size = float(cell_size)
        obj.origin = origin
        obj.dims = dims
        obj._cell_keys = cell_keys
        obj._cell_starts = cell_starts
        obj._order = order
        return obj

    def index_arrays(self) -> dict:
        """Arrays needed to persist this index (points themselves are omitted)."""
        return {
            "cell_size": float(self.cell_size),
            "origin": np.asarray(self.origin, dtype=np.float64),
            "dims": np.asarray(self.dims, dtype=np.int64),
            "cell_keys": np.asarray(self._cell_keys),
            "cell_starts": np.asarray(self._cell_starts),
            "order": np.asarray(self._order),
        }

    @classmethod
    def cell_size_for_radius(cls, radius_mm: float) -> float:
        """Grid resolution matched to the neighbour query radius."""
        return max(float(radius_mm), 1.0)

    def _bruteforce_radius_indices(self, center: np.ndarray, radius: float) -> np.ndarray:
        d2 = np.einsum(
            "ij,ij->i", self.points - center, self.points - center, optimize=True
        )
        return np.flatnonzero(d2 <= radius * radius)

    def _collect_cell_candidates(
        self, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray:
        ii = np.arange(lo[0], hi[0] + 1, dtype=np.int64)
        jj = np.arange(lo[1], hi[1] + 1, dtype=np.int64)
        kk = np.arange(lo[2], hi[2] + 1, dtype=np.int64)
        I, J, K = np.meshgrid(ii, jj, kk, indexing="ij")
        keys = ((I * self.dims[1] + J) * self.dims[2] + K).ravel()
        pos = np.searchsorted(self._cell_keys, keys)
        in_range = pos < len(self._cell_keys)
        pos = pos[in_range]
        keys = keys[in_range]
        match = self._cell_keys[pos] == keys
        pos = pos[match]
        if len(pos) == 0:
            return np.empty(0, dtype=np.int64)
        chunks = [
            self._order[self._cell_starts[p] : (
                self._cell_starts[p + 1]
                if p + 1 < len(self._cell_starts)
                else len(self._order)
            )]
            for p in pos
        ]
        return np.concatenate(chunks)

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
        radius = float(radius)
        lo = np.floor((center - radius - self.origin) / self.cell_size).astype(np.int64)
        hi = np.floor((center + radius - self.origin) / self.cell_size).astype(np.int64)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, self.dims - 1)
        if np.any(hi < lo):
            return np.empty(0, dtype=np.int64)

        n_cells = int(np.prod(hi - lo + 1))
        if n_cells > _MAX_QUERY_CELLS:
            return self._bruteforce_radius_indices(center, radius)

        cand = self._collect_cell_candidates(lo, hi)
        if len(cand) == 0:
            return cand
        d2 = np.einsum(
            "ij,ij->i",
            self.points[cand] - center,
            self.points[cand] - center,
            optimize=True,
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
    work_cap = min(_DISPLAY_WORK_CAP, max(int(max_points) * 4, int(max_points)))
    if len(points) > work_cap:
        points = _precap_points(points, work_cap, seed)
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
