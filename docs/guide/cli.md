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

Also writes a sibling `geometry_summary.json` (slim names + coordinates;
aligned frame when available).

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

### Convert text point clouds to PLY

```bash
cloudet convert <input> [<input> ...] [-o OUT] [--delimiter D] [--chunk-rows N] [--force]
```

Converts text point clouds (`x,y,z` per line, e.g. FARO / PolyWorks exports) to
double-precision `binary_little_endian` PLY — the same layout as Open3D-written
files, so coordinates are preserved exactly (float32 would only resolve ~0.1 mm
at ~2 m from the origin). Input is streamed in chunks, so file size is not
limited by memory. Output is written to `<name>.ply.part` and renamed on
success; an interrupted run never leaves a file that looks valid. The tool also
checks free disk space first (double-precision PLY is 24 bytes per point).

Blank lines and `#` comments are skipped, CRLF/LF both work, and columns beyond
the first three are ignored. Non-numeric or non-finite coordinates are an error.
Existing outputs are skipped unless `--force` is given; a failure on one file
does not stop the others (exit status 1 if any failed).

| Argument | Description |
|----------|-------------|
| `INPUT` | One or more text point cloud files |
| `-o PATH` | Output `.ply` file (single input) or output directory (created if missing). Default: next to each input, with a `.ply` suffix |
| `--delimiter` | `auto` (default; sniffed from the first data line), `comma`, `semicolon`, `tab`, `space` |
| `--chunk-rows N` | Rows parsed per chunk (default 4000000) |
| `--force` | Overwrite existing outputs |

```bash
cloudet convert scan_1.txt scan_2.txt -o ply/
```

### Version

```bash
cloudet version
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLOUDET_COMPUTE_BACKEND` | `auto` | Force compute backend: `auto`, `numpy`, or `cupy` |
| `CLOUDET_VTK_LOG` | `<project>/vtk.log` | VTK log path; set to `0` to use PyVista default |
