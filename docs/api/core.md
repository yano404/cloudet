# cloudet.core

Shared types, I/O, and spatial indexing used across all subsystems.

## Overview

`cloudet.core` provides the foundational building blocks:

- **`Plane`** — immutable plane in Hesse normal form (`n · x + d = 0`, `|n| = 1`).
  This is the primary data type flowing through the entire pipeline.
- **Plane fitting** — from a simple least-squares fit (`fit_plane_lsq`) to
  a full iterative robust pipeline (`robust_fit_plane`) that alternates
  fitting and strict inlier reselection until convergence.
- **RANSAC** — used only for initial point selection; the final plane is always
  an orthogonal least-squares fit. Multiple backends are supported
  (NumPy, CuPy, Open3D).
- **PLY I/O** — read/write double-precision XYZ point clouds without Open3D.
- **`VoxelHashGrid`** — spatial indexing for neighbor queries and display
  downsampling. Supports both CPU and optional CuPy GPU paths.
- **`ArrayContext`** — transparent NumPy ↔ CuPy switching with automatic
  fallback when CuPy is unavailable.

## Quick example

```python
import numpy as np
from cloudet.core import Plane, fit_plane_lsq, robust_fit_plane, read_ply_xyz

# Load a point cloud
points = read_ply_xyz("scan.ply")

# Simple least-squares fit
plane = fit_plane_lsq(points)
print(plane.normal, plane.d)

# Robust iterative fit (RANSAC seed → strict reselection → converge)
result = robust_fit_plane(points, threshold=0.5)
print(f"converged={result.converged}, inliers={result.inlier_mask.sum()}")

# Signed distances from the plane (mm)
residuals = result.plane.signed_distances(points)

# Residual statistics
from cloudet.core.plane import residual_stats
stats = residual_stats(residuals)
print(f"mean={stats['mean_mm']:.4f}, mad_sigma={stats['mad_sigma_mm']:.4f}")
```

---

## Plane

The `Plane` dataclass represents a plane in Hesse normal form.
All planes in cloudet use the convention `n · x + d = 0` with `|n| = 1`.

::: cloudet.core.plane.Plane
    options:
      members:
        - normal
        - d
        - signed_distances
        - as_array
        - from_array

## FitResult

Returned by `robust_fit_plane`, bundles the fitted plane with convergence
information and the inlier mask.

::: cloudet.core.plane.FitResult
    options:
      members:
        - plane
        - inlier_mask
        - n_iterations
        - converged
        - threshold

## Plane fitting functions

### fit_plane_lsq

Orthogonal least-squares fit (SVD-based). Use when all points are known
to belong to the plane (no outliers).

::: cloudet.core.plane.fit_plane_lsq

### robust_fit_plane

The primary fitting entry point. Alternates between fitting and strict
inlier reselection until the plane equation converges. Starts from a
RANSAC seed or a user-supplied initial plane.

::: cloudet.core.plane.robust_fit_plane

### ransac_plane

RANSAC plane selection. Returns the best-fit plane and an inlier boolean mask.
Normally called internally by `robust_fit_plane`, but available for direct use.

::: cloudet.core.plane.ransac_plane

## Residual statistics

### residual_stats

Computes summary statistics (mean, std, median, MAD, percentiles) for
signed residuals. Reports both full-set and inlier-only numbers.

::: cloudet.core.plane.residual_stats

### mad_sigma

Median absolute deviation scaled to Gaussian σ equivalent.
Robust to outliers; used as the default quality metric.

::: cloudet.core.plane.mad_sigma

---

## PLY I/O

Read and write PLY point clouds in double precision without Open3D dependency.

```python
from cloudet.core import read_ply_xyz, write_ply_xyz

points = read_ply_xyz("scan.ply")       # → ndarray (N, 3) float64
write_ply_xyz("output.ply", points)
```

::: cloudet.core.plyio.read_ply_xyz

::: cloudet.core.plyio.write_ply_xyz

---

## Spatial indexing

### VoxelHashGrid

Hash-based voxel grid for fast spatial queries. Used for neighbor lookup
during picking and for display downsampling.

```python
from cloudet.core import VoxelHashGrid

grid = VoxelHashGrid(points, cell_size=5.0)
neighbors = grid.query_ball(center, radius=10.0)
```

::: cloudet.core.neighbors.VoxelHashGrid

### display_xyz

Downsample a point cloud for display, selecting one representative point
per voxel. Prefers CuPy when available.

::: cloudet.core.neighbors.display_xyz

---

## GPU backend

Transparent NumPy ↔ CuPy switching. When CuPy is not installed or the GPU
is unavailable, all operations fall back to NumPy automatically.

```python
from cloudet.core import get_context, cupy_available

print(cupy_available())  # True if CuPy + CUDA are working

ctx = get_context(backend="auto")
print(ctx.xp.__name__)  # "cupy" or "numpy"
```

::: cloudet.core.array_backend.ArrayContext

::: cloudet.core.array_backend.get_context

::: cloudet.core.array_backend.cupy_available
