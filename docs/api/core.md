# cloudet.core

Shared types, I/O, and spatial indexing used across all subsystems.

- [`plane`](core-plane.md) — Hesse-form `Plane`, least-squares / RANSAC / robust fitting
- [`plyio`](core-plyio.md) — double-precision PLY read/write
- [`neighbors`](core-neighbors.md) — voxel hash grid and display downsampling
- [`array_backend`](core-array-backend.md) — NumPy / CuPy switching

## Quick example

```python
from cloudet.core import fit_plane_lsq, read_ply_xyz, robust_fit_plane

points = read_ply_xyz("scan.ply")
plane = fit_plane_lsq(points)
result = robust_fit_plane(points, threshold=0.5)
residuals = result.plane.signed_distances(points)
```
