# cloudet

日本語版は [README.ja.md](README.ja.md) を参照。

Tool for reducing detector positions and relative geometry from 3D point clouds (e.g. FARO Quantum-S), aimed at nuclear-physics detector surveys.

## Design

- **Extraction contract: one click = one connected physical face = one group = one plane equation.**
  Growth starts at the click seed and expands in-plane; extent is set by connectivity (not by a radius cutoff).
  The seed normal is assumed noisy: accumulate → refit → re-accumulate until convergence
  (full-face recovery from a ~20° tilted seed has been verified).
  Separating nearby parallel faces is an exception mode (GUI “Split into parallel planes”).
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
  pipeline.py   Residual u–v maps (for GUI QC)
  picker_qt.py  Interactive app (PySide6 + PyVista)
  cli.py        cloudet [project] [--cloud ...] (default: launch the app)
tests/          Synthetic validation (FARO-like σ ≈ 0.03 mm)
```

## Usage

```bash
pip install -e ".[dev]"       # full app (includes Qt UI)
pip install -e ".[dev,fast]"  # optional: faster display decimation via Open3D
pytest

# Launch the app (pick / Fit / residual QC / save are all in the GUI)
cloudet --cloud /path/to/scan.ply
cloudet ~/surveys/proj1 --cloud /path/to/scan.ply
```

Project layout:

```text
<project>/
  manifest.json
  settings.json
  groups/
    group_000.ply / .json / _indices.npy
    ...
  vtk.log          # when using the GUI
```

Qt UI: set the output folder under PROJECT / Load the cloud under SOURCE /
`P` pick / overlap only `>` farther and `<` nearer /
`M` append toggle / `F` fit active / `V` show only active / `Ctrl+S` save groups /
rename in the tree, toggle visibility, and see per-plane quality in the tree.
After Fit, the right dock shows a pyqtgraph residual u–v map and a signed-residual histogram (µm).
Cmd/Ctrl+drag for rectangle selection (handles for adjustment). Zoom / pan supported.
Refit selection fits an extra plane on those points only (original fit and rectangle remain).
Clear refit removes only the extra fit. Selecting a plane switches the display.
The Groups tab mirrors depth controls with navigator buttons.
VTK’s own errors and warnings go to `<project_dir>/vtk.log` instead of the terminal
(`CLOUDET_VTK_LOG=0` restores PyVista’s default; any other value is used as the log path).

## Roadmap

1. [done] Core
2. [done] GUI picker (Fit / residual QC / save)
3. [todo] Relative-geometry reduction (inter-face distances, angles, intersections, corners)
