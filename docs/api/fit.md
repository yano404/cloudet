# cloudet.fit

Click-to-fit plane extraction. All of these modules are GUI-independent.

1. [`picking`](fit-picking.md) — grow a connected planar region from a click
2. [`mainplane`](fit-mainplane.md) — refine the region into one high-quality plane
3. [`multiplane`](fit-multiplane.md) — split nearby parallel faces when needed
4. [`pipeline`](fit-pipeline.md) — residual u–v maps for QC

## Typical workflow

```python
import numpy as np
from cloudet.core import VoxelHashGrid, read_ply_xyz
from cloudet.fit import MainPlaneParams, PickParams, extract_main_plane, pick_plane_region

points = read_ply_xyz("scan.ply")
grid = VoxelHashGrid(points, cell_size=5.0)
clicked = np.array([100.0, 200.0, 50.0])
neighbors = grid.radius_indices(clicked, radius=10.0)
indices, initial_plane = pick_plane_region(
    points, clicked, neighbors,
    params=PickParams(local_radius_mm=10.0),
)
result = extract_main_plane(points[indices], MainPlaneParams(), initial_plane=initial_plane)
```
