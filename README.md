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
  reduce.py     Recipe-driven reduction → geometry.json for analysis
  pipeline.py   Residual u–v maps (for GUI QC)
  picker_qt.py  Interactive app (PySide6 + PyVista)
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
  "export": ["beam_axis", "beam_on_target"]
}
```

Supported construct ops: `offset`, `intersect_planes`, `intersect_three_planes`,
`intersect_line_plane`, `intersect_normal_plane`, `line_from_point_normal`
(axis through a point along a plane normal), `line_from_two_points`,
`midpoint_line_planes` (midpoint of the segment cut by two planes).
Output `geometry.json` lists
planes / lines / points with provenance (`scanned` | `offset` | `intersection`).

### Interactive reduction (GUI)

In the app, open the **Reduction** dock (tabified with Residuals):

1. Choose an **operation** — PARAMETERS shows only that step’s inputs (plane / axis pickers, offset slider, …)
2. For **Offset**: pick a plane in PARAMETERS, drag the distance slider for a green live preview, then Apply
3. For intersections: pick the required planes/axes in PARAMETERS, then Apply
4. Toggle visibility in Entities; **Save recipe…** / **Export geometry…** for analysis

## Roadmap

1. [done] Core
2. [done] GUI picker (Fit / residual QC / save)
3. [done] Constructive geometry reduction (recipe → geometry.json; CLI + GUI)
4. [todo] Richer recipe editor; detector rigid-body pose helpers
