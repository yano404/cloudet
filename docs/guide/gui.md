# GUI Workflow

## Launching

```bash
cloudet --cloud /path/to/scan.ply
cloudet ~/surveys/proj1 --cloud /path/to/scan.ply
```

## Project layout

```text
<project>/
  manifest.json
  settings.json
  groups/
    group_000.ply / .json / _indices.npy
    group_000_p0_indices.npy   # inliers used to fit p0 (optional)
    group_000_cyl0_indices.npy # cylinder inliers (optional)
    group_000_cir0_indices.npy # circle inliers (optional)
    ...
  vtk.log          # VTK warnings (when using the GUI)
```

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `P` | Pick (start region extraction) |
| `>` / `<` | Overlap only: farther / nearer |
| `M` | Append toggle |
| `F` | Fit active group |
| `V` | Show only active group |
| `Ctrl+S` | Save groups |
| `Cmd/Ctrl+drag` | Rectangle selection |

## Workflow

1. Set the output folder under **PROJECT**
2. Load the point cloud under **SOURCE**
3. **Pick** faces on the 3D view — each click seeds a connected planar region
4. **Fit** to compute the plane equation
5. Inspect the residual u–v map and histogram in the **Residuals** tab
6. **Save** groups to the project folder
7. Proceed to **Reduction** to derive geometry (see [Geometry Reduction](reduction.md))

## Residual QC

After Fit, the right dock shows:

- **u–v residual map** (pyqtgraph) with signed residuals in µm
- **Histogram** of signed residuals

Use `Cmd/Ctrl+drag` for rectangle selection to refit a subset.
Refit adds extra planes (p1, p2, …) on the same group.
Import them into Reduction as `G6_p1`. Cylinder axes and circle centers
import the same way (`G6_cyl0`, `G6_cir0`).

## VTK logging

VTK errors go to `<project_dir>/vtk.log` instead of the terminal.

- `CLOUDET_VTK_LOG=0` restores PyVista's default
- Any other value is used as the log path
