# cloudet.project

On-disk project layout: manifest, settings, groups, and spatial caches.

```text
<project>/
  manifest.json
  settings.json
  groups/group_000.ply / .json / _indices.npy
```

- [`store`](project-store.md) — settings, manifest, group save/load
- [`schema`](project-schema.md) — plane JSON + recipe/geometry migration
- [`groups`](project-groups.md) — load saved groups
- [`spatial_cache`](project-spatial-cache.md) — voxel / display caches
- [`settings_apply`](project-settings-apply.md) — detect vs display setting changes
