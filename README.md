# cloudet

日本語版は [README.ja.md](README.ja.md) を参照。

Tool for reducing detector positions and relative geometry from 3D point clouds (e.g. FARO Quantum-S), aimed at nuclear-physics detector surveys.

## Design

- **Extraction contract: one click = one connected physical face = one group = one plane equation.**
  Growth starts at the click seed and expands in-plane; extent is set by connectivity (not by a radius cutoff).
  The seed normal is assumed noisy: accumulate → refit → re-accumulate until convergence
  (full-face recovery from a ~20° tilted seed has been verified).
  Separating nearby parallel faces is an exception mode (GUI “Extract multiple planes (p0, p1, …)”).
- The compute core (`cloudet/`) depends only on NumPy, is fully decoupled from the GUI, and is covered by unit tests.
- RANSAC is used only to select points; the final plane is always an orthogonal least-squares fit (`robust_fit_plane`: fit → strict reselection → iterate to convergence).
- Statistics are reported for both inliers (truncated) and all points, together with a truncation-robust `mad_sigma`.
- Randomness is seeded for reproducibility.
- Units are mm; planes use Hesse normal form `n·x + d = 0` (`|n|=1`, with a defined sign convention).

## Layout

```
cloudet/
  plane.py      Plane fit core (LSQ / RANSAC / robust iteration / residual stats)
  mainplane.py  Main plane component extraction (connected components + QC gates)
  picking.py    Click-driven region extraction (GUI-independent)
  plyio.py      PLY I/O (double precision, Open3D-free)
  groups.py     Group loading
  project.py    Project directory I/O (manifest / settings / group save)
  array_backend.py  Optional CuPy GPU backend (auto fallback to NumPy)
  geometry.py   Constructive ops (offset plane, intersections)
  frame.py      Display-only Align Z pose (axis → +Z, optional line → XY)
  reduce.py     Recipe-driven reduction → geometry.json for analysis
  pipeline.py   Residual u–v maps (for GUI QC)
  app_window.py Preferred Qt app window entrypoint
  picker_qt.py  Legacy re-exports (use app_window or ui.main_window)
  ui/
    main_window.py    CloudetAppWindow + run_picker_qt
    groups_mixin.py   Groups / Settings dock, pick, fit, tree
    reduction_mixin.py  Reduction + Measure docks
    uv_mixin.py       Residual u–v map dock
    render_mixin.py   3D actor rendering
    frame_mixin.py    Align Z view frame
    widgets.py        Shared Qt styling and helpers
  reduction_ops.py  Shared reduction op metadata (GUI ↔ recipe)
  cli.py        cloudet [project] [--cloud ...] | reduce | version
tests/          Synthetic validation (FARO-like σ ≈ 0.03 mm)
```

## Usage

```bash
pip install -e ".[dev]"       # full app (includes Qt UI)
pip install -e ".[dev,open3d]"  # optional: Open3D display decimation and RANSAC backend
pip install -e ".[dev,gpu]"   # optional: CuPy GPU for Fit / residual QC / display voxel
pytest

# Launch the app (pick / Fit / residual QC / save are all in the GUI)
cloudet --cloud /path/to/scan.ply
cloudet ~/surveys/proj1 --cloud /path/to/scan.ply

# Constructive geometry reduction (saved fits + recipe → analysis parameters)
cloudet reduce ~/surveys/proj1 --recipe recipe.json -o geometry.json
```

On Linux/WSL, the `[gpu]` extra installs `cupy-cuda12x[ctk]` (CUDA headers for kernel compile). If CuPy is installed without headers, cloudet falls back to NumPy automatically in `auto` mode.

### GPU (optional, NVIDIA + CUDA 12.x)

3D **rendering** already uses the GPU via VTK/OpenGL. Optional **CuPy** accelerates Fit, residual u–v maps, and display voxel downsampling on large clouds.

```bat
pip install -e ".[dev,gpu]"
pip install "cupy-cuda12x[ctk]"   # if GPU probe fails: missing CUDA headers (common on WSL)
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
cloudet --cloud C:\path\to\scan.ply
```

In Settings → **Compute backend**: `auto` (CuPy when available), `numpy`, or `cupy`.  
**Display downsampling method** `auto` also prefers CuPy over Open3D when installed.

CuPy is not required: Mac and CPU-only machines keep using NumPy. Clouds under ~50k points stay on CPU even in `auto` mode. Set `CLOUDET_COMPUTE_BACKEND=numpy` to force CPU.

Project layout:

```text
<project>/
  manifest.json
  settings.json
  groups/
    group_000.ply / .json / _indices.npy
    group_000_p0_indices.npy   # inliers used to fit p0 (optional)
    ...
  vtk.log          # when using the GUI
```

Qt UI: set the output folder under PROJECT / Load the cloud under SOURCE /
`P` pick / overlap only `>` farther and `<` nearer /
`M` append toggle / `F` fit active / `V` show only active / `Ctrl+S` save groups /
rename in the tree, toggle visibility, and see per-plane quality in the tree.
After Fit, the right dock shows a pyqtgraph residual u–v map and a signed-residual histogram (µm).
Cmd/Ctrl+drag for rectangle selection (handles for adjustment). Zoom / pan supported.
Refit selection fits an extra plane on those points and adds it as p1, p2, … on the same group (original plane kept). Import that plane into Reduction as G6_p1.
Clear refit removes only the extra fit. Selecting a plane switches the display.
The Groups tab mirrors depth controls with navigator buttons.
VTK’s own errors and warnings go to `<project_dir>/vtk.log` instead of the terminal
(`CLOUDET_VTK_LOG=0` restores PyVista’s default; any other value is used as the log path).

## Geometry reduction

After Fit + save, faces live in `groups/group_*.json` (`fit.planes[].abcd`).
Analysis parameters (virtual axes, beam-on-target points, drawing offsets)
are derived with a declarative recipe — no point cloud required at this stage.

```bash
cloudet reduce <project> --recipe recipe.json -o geometry.json
```

Offset sign convention: positive `distance_mm` moves the plane along its
Hesse unit normal (same sign convention as Fit). Use a negative distance
for the opposite side (e.g. “inward” relative to an outward-facing normal).

Example recipe (tracker walls → beam axis ∩ target):

```json
{
  "version": 1,
  "units": "mm",
  "faces": {
    "tracker_left":  { "from": "group", "name": "G0" },
    "tracker_front": { "from": "group", "name": "G1" },
    "target":        { "from": "group", "name": "G2" }
  },
  "construct": [
    { "id": "left_in",  "op": "offset", "of": "tracker_left",  "distance_mm": 12.0 },
    { "id": "front_in", "op": "offset", "of": "tracker_front", "distance_mm": 12.0 },
    { "id": "beam_axis", "op": "intersect_planes", "a": "left_in", "b": "front_in" },
    { "id": "beam_on_target", "op": "intersect_line_plane", "line": "beam_axis", "plane": "target" }
  ],
  "export": ["beam_axis", "beam_on_target"],
  "frame": { "axis": "beam_axis", "origin": "beam_on_target", "flip_z": false }
}
```

Supported construct ops: `offset`, `intersect_planes`, `intersect_three_planes`,
`intersect_line_plane`, `intersect_normal_plane` (source-plane normal ∩ dest plane), `line_from_point_normal`
(axis through a point along a plane normal), `line_from_two_points`,
`midpoint_line_planes` (midpoint of the segment cut by two planes),
`plane_from_plane_point`, `plane_from_line_point`, `plane_from_two_lines`,
`rotate_plane_about_line` (rigid rotation about any axis; angle in degrees).

#### `geometry.json` (export output)

`geometry.json` is the **executed** recipe plus computed entity records (not the
point cloud). Top-level `planes` / `lines` / `points` are always in **survey**
coordinates. Each record includes provenance (`scanned` | `offset` | `intersection`
| `constructed`) and parameters (`abcd`, `point`/`direction`, `xyz`, …).

| Key | Contents |
|-----|----------|
| `recipe` | `{ "sha256", "echo" }` — full recipe copy for reproducibility |
| `export` | ids marked for analysis (metadata; all entities are still listed) |
| `frame` | optional Align Z pose (`axis`, `origin`, `flip_z`, optional `yaw_*`) |
| `aligned` | optional `{ planes, lines, points }` in the aligned frame |
| `measures` | optional pinned measurements with recomputed `value` / `unit` |

**Aligned export:** `cloudet reduce` adds `aligned` + `frame` when the recipe
has `frame`. In the GUI, check **Also write aligned-frame coordinates** and
set FRAME **axis** and **origin** (Align Z is not required for export).
The GUI also writes a sibling `geometry_recipe.json` for replay.

#### Aligned axis operands (`aligned.x` / `aligned.y` / `aligned.z`)

When `recipe.frame` sets `axis` and `origin`, construct steps may reference
virtual line ids `aligned.x`, `aligned.y`, `aligned.z` as operands. They are
lines through the frame origin along the view triad (+X / +Y / +Z in survey
space after Align Z). They appear in GUI **line** combos once FRAME axis and
origin are chosen; they are **not** stored as separate entities and do not
appear in `geometry.json` as their own rows. Example:

```json
{
  "frame": { "axis": "beam_axis", "origin": "beam_on_target", "flip_z": false },
  "construct": [
    {
      "id": "tilted",
      "op": "rotate_plane_about_line",
      "plane": "target",
      "line": "aligned.x",
      "angle_deg": 90.0
    }
  ]
}
```

Output `geometry.json` lists planes / lines / points with provenance.

Optional top-level `frame` is Align Z metadata only (`axis` line id, `origin`
point id, `flip_z`, and optionally `yaw_line` or `yaw_plane` with `yaw_to`
(`x`, `-x`, `y`, or `-y`). The line direction or plane normal is projected
into the XY plane after the axis maps to +Z. It is not a construct step and
does not change survey coordinates. `cloudet reduce` still writes survey numbers at the top level;
when `frame` is present it also writes an `aligned` copy plus the pose.
The GUI restores those picks on Load recipe / Load All; it does **not**
apply Align Z until you press the button.

### Interactive reduction (GUI)

In the app, the right-hand docks are tabbed (**Residuals** / **Reduction** / **Measure**).

Open **Reduction** to construct geometry:

1. Choose an **operation** — only that step’s inputs appear (plane / axis pickers, offset slider, …)
2. For **Offset**: pick a plane, drag the distance slider for a green live preview, then Apply
3. For intersections: pick the required planes/axes, then Apply.
   **Normal ∩ plane → point** shoots the source overlay's normal at another plane.
4. Toggle visibility in Entities; **Save recipe…** / **Export geometry…** for analysis
5. **FRAME** (display only): pick Axis (line) and Origin (point), then **Align Z**.
   The view uses the smallest rotation that maps the axis to `(0, 0, 1)` with
   the origin at `(0, 0, 0)`. Optional **XY**: map a **line** or **plane
   normal** (horizontal component only) onto ±X or ±Y. Omit XY for the smallest
   rotation only. **Survey** returns the view to survey coordinates. Groups, recipe constructs, and Fit stay in survey;
   picking still uses the original cloud. Once axis and origin are set, **aligned X/Y/Z**
   appear in line operand combos (rotate, line ∩ plane, …) and as thin
   **aligned X/Y/Z** rows at the bottom of Entities (visibility only;
   rename/delete stay off).
6. With **Also write aligned-frame coordinates** checked and FRAME **axis** and
   **origin** set, **Export geometry…** adds an `aligned` copy plus `frame` under
   survey numbers (Align Z is not required for export). `cloudet reduce` uses the
   same rule when the recipe has `frame`. Load recipe restores FRAME combos from
   `recipe.frame`; press Align Z when you want the 3D view aligned.

Open **Measure** to read distances and angles from those entities:

- Kinds: Distance (point - point / point - plane / point - line) and
  Angle (plane - plane / line - line / line - plane).
- Distances are unsigned (mm).
- Line–plane angle is 0° if the line is parallel to the plane.
- **Add measurement** pins the row into `recipe.measures` and `geometry.json`
  (values are recomputed). Distance measures draw a teal segment in the 3D view.

## Roadmap

1. [done] Core
2. [done] GUI picker (Fit / residual QC / save)
3. [done] Constructive geometry reduction (recipe → geometry.json; CLI + GUI)
4. [todo] Richer recipe editor; detector rigid-body pose helpers
