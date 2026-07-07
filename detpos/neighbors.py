"""Neighbour search and display decimation, numpy only.

Replaces Open3D's KDTreeFlann / voxel_down_sample in the GUI so that
the Qt picker has no Open3D dependency. A voxel hash grid gives O(N)
build time (seconds for ~6e7 points) and fast radius queries, which is
all the picker needs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["VoxelHashGrid", "voxel_downsample_indices", "display_indices"]


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


def voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Indices of one representative point per occupied voxel."""
    points = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0:
        return np.arange(len(points))
    ijk = np.floor((points - points.min(axis=0)) / voxel_size).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    keys = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, first = np.unique(keys, return_index=True)
    return np.sort(first)


def display_indices(
    points: np.ndarray,
    voxel_size: float,
    max_points: int,
    seed: int = 0,
) -> np.ndarray:
    """Indices for display: optional voxel filter then a hard random cap."""
    idx = voxel_downsample_indices(points, voxel_size)
    if len(idx) > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    return idx
