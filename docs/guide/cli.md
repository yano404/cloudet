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

### Version

```bash
cloudet version
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLOUDET_COMPUTE_BACKEND` | `auto` | Force compute backend: `auto`, `numpy`, or `cupy` |
| `CLOUDET_VTK_LOG` | `<project>/vtk.log` | VTK log path; set to `0` to use PyVista default |
