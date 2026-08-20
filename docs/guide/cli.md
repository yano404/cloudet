# CLI Reference

## Commands

### Launch GUI

```bash
cloudet [project_dir] --cloud /path/to/scan.ply
```

| Argument | Description |
|----------|-------------|
| `project_dir` | Optional project directory (default: current directory) |
| `--cloud PATH` | Path to PLY point cloud file |

### Batch reduction

```bash
cloudet reduce <project_dir> --recipe recipe.json [-o geometry.json]
```

| Argument | Description |
|----------|-------------|
| `project_dir` | Project directory containing saved groups |
| `--recipe PATH` | Recipe JSON file |
| `-o PATH` | Output geometry.json path (default: `geometry.json` in project dir) |

### Migrate on-disk JSON

```bash
cloudet migrate <project_dir> [--recipe FILE] [--geometry FILE] [--dry-run]
```

Rewrites legacy schema keys to the current form:

- group `fit.planes[]`: `abcd` → `normal` + `d`
- recipe construct/measures: v1 operand keys → v2 (`plane`, `plane_a`, …)
- geometry entity records: `abcd` / provenance `of` → `normal`/`d` / `parents`

Always scans `<project>/groups/group_*.json`. Without `--recipe` / `--geometry`,
also migrates `geometry_recipe.json` / `recipe.json` / `geometry.json` under the
project when present. `--dry-run` lists files that would change without writing.

| Argument | Description |
|----------|-------------|
| `project_dir` | Project directory |
| `--recipe PATH` | Optional recipe JSON path |
| `--geometry PATH` | Optional geometry.json path |
| `--dry-run` | Report only |

### Version

```bash
cloudet version
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLOUDET_COMPUTE_BACKEND` | `auto` | Force compute backend: `auto`, `numpy`, or `cupy` |
| `CLOUDET_VTK_LOG` | `<project>/vtk.log` | VTK log path; set to `0` to use PyVista default |
