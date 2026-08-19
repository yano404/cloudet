# cloudet.fit

Face extraction and fitting from 3D point clouds.

## Overview

The `fit` package implements the click-to-plane extraction pipeline:

1. **Picking** (`picking.py`) — the user clicks a point on a detector surface;
   `pick_plane_region` grows a connected planar region from that seed using
   local RANSAC + accumulation, then returns the region mask and an initial
   plane fit.
2. **Main plane extraction** (`mainplane.py`) — refines the picked region by
   decomposing it into connected components, discarding outlier clusters, and
   computing final statistics (in-plane u–v coordinates, quality metrics).
3. **Multi-plane separation** (`multiplane.py`) — when a single pick spans
   two nearby parallel surfaces, `extract_planes` separates them using
   bimodality detection on signed residuals.
4. **Residual pipeline** (`pipeline.py`) — computes u–v residual maps for
   visual QC in the GUI.

All functions are GUI-independent and testable with synthetic point clouds.

## Typical workflow

```python
import numpy as np
from cloudet.core import read_ply_xyz, VoxelHashGrid
from cloudet.fit.picking import PickParams, pick_plane_region
from cloudet.fit.mainplane import MainPlaneParams, extract_main_plane

# Load point cloud and build spatial index
points = read_ply_xyz("scan.ply")
grid = VoxelHashGrid(points, cell_size=5.0)
neighbor_idx = grid.neighbor_indices()

# User clicks at a point on the surface
clicked = np.array([100.0, 200.0, 50.0])

# Extract the planar region around the click
indices, initial_plane = pick_plane_region(
    points, clicked, neighbor_idx,
    params=PickParams(local_radius_mm=10.0),
)

# Refine: connected components, QC, final plane
result = extract_main_plane(
    points[indices],
    MainPlaneParams(),
    initial_plane=initial_plane,
)
print(f"plane: {result.plane.normal}, inliers: {result.n_inliers}")
```

---

## Picking

### PickParams

Controls the seed expansion and accumulation behaviour.
All distances are in mm.

::: cloudet.fit.picking.PickParams
    options:
      members:
        - local_radius_mm
        - local_distance_threshold_mm
        - local_ransac_iterations
        - min_neighbor_points
        - min_local_inliers
        - accumulate_threshold_mm
        - connect
        - cell_size_mm

### pick_plane_region

The main entry point for interactive plane extraction.
Given a clicked 3D point and a pre-built neighbor index, it returns the
extracted region indices and an initial plane fit.

::: cloudet.fit.picking.pick_plane_region

---

## Main plane extraction

### MainPlaneParams

Parameters for the main-plane refinement step.

::: cloudet.fit.mainplane.MainPlaneParams

### MainPlaneResult

Result of `extract_main_plane`, including the fitted plane, in-plane
coordinates, component labels, and quality metrics.

::: cloudet.fit.mainplane.MainPlaneResult

### extract_main_plane

Refines a picked region into a single high-quality plane with connected
component analysis and iterative robust fitting.

::: cloudet.fit.mainplane.extract_main_plane

---

## Multi-plane separation

### extract_planes

When the signed-residual distribution is bimodal (two parallel surfaces
in one pick), this function separates them into individual planes.

::: cloudet.fit.multiplane.extract_planes

---

## Residual pipeline

### residual_uv_map

Computes the in-plane u–v residual map used by the GUI residual QC view.

::: cloudet.fit.pipeline.residual_uv_map
