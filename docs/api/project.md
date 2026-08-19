# cloudet.project

Project persistence: manifest, settings, group save/load, and spatial caching.

## Overview

A cloudet project is a directory containing:

```text
<project>/
  manifest.json          # cloud source, hash, creation metadata
  settings.json          # picker and display settings
  groups/
    group_000.ply        # extracted point subset
    group_000.json       # fit results, labels, metadata
    group_000_indices.npy  # indices into the original cloud
    group_000_p0_indices.npy  # inlier indices for the primary fit
    ...
  vtk.log               # VTK warnings (GUI mode)
```

The `project` package handles all file I/O for this layout:

- **`store.py`** — save/load manifest, settings, and individual groups.
  `PickerSettings` holds the full picker configuration.
- **`groups.py`** — load all groups from a project directory, with SHA-256
  integrity checking of PLY files.
- **`spatial_cache.py`** — on-disk caching of `VoxelHashGrid` and display
  downsampled points. Avoids re-computing the spatial index on every launch.
- **`settings_apply.py`** — classifies which settings changes require
  re-detection vs. only a display refresh.

## Typical usage

```python
from pathlib import Path
from cloudet.project.store import (
    PickerSettings, save_settings, load_settings,
    write_manifest, save_group,
)
from cloudet.project.groups import load_groups

project = Path("~/surveys/proj1").expanduser()

# Save settings
settings = PickerSettings()
save_settings(project, settings)

# Load all saved groups
groups = load_groups(project)
for g in groups:
    print(f"Group {g.group_id}: {g.label}, {g.n_points} points")
```

---

## Store

### PickerSettings

Full picker and display configuration. Serialized to `settings.json`.

::: cloudet.project.store.PickerSettings

### save_settings / load_settings

::: cloudet.project.store.save_settings

::: cloudet.project.store.load_settings

### Manifest

The manifest records the source cloud path, its SHA-256 hash, and
creation metadata.

::: cloudet.project.store.write_manifest

::: cloudet.project.store.read_manifest

### Group save/load

::: cloudet.project.store.save_group

::: cloudet.project.store.FittedPlane

::: cloudet.project.store.load_group_doc

::: cloudet.project.store.load_group_docs

::: cloudet.project.store.load_fitted_plane

---

## Groups

### GroupInfo

Lightweight summary of a saved group (id, label, point count, PLY path).

::: cloudet.project.groups.GroupInfo

### load_groups

Load all groups from a project directory. Returns a list of `GroupInfo`
sorted by group id.

::: cloudet.project.groups.load_groups

---

## Spatial cache

On-disk caching avoids recomputing the voxel grid and display-resolution
points on every application launch.

### Voxel grid cache

::: cloudet.project.spatial_cache.load_voxel_grid

::: cloudet.project.spatial_cache.save_voxel_grid

### Display points cache

::: cloudet.project.spatial_cache.load_display_xyz

::: cloudet.project.spatial_cache.save_display_xyz

---

## Settings classification

Determines whether a settings change requires re-running detection
(expensive) or only updating the display (cheap).

::: cloudet.project.settings_apply.classify_settings_apply
