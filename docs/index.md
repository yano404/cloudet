# cloudet

Tool for reducing detector positions and relative geometry from 3D point clouds
(e.g. FARO Quantum-S), aimed at nuclear-physics detector surveys.

## Features

- **One-click face extraction** — seed-driven in-plane growth with iterative robust fitting
- **Constructive geometry reduction** — declarative recipe turns fitted planes into axes, intersection points, and offsets
- **Aligned coordinate frame** — virtual origin, axes, and coordinate planes derived from the rigid frame
- **GPU acceleration** — optional CuPy backend for large point clouds
- **CLI and GUI** — Qt-based interactive workflow or headless batch processing

## Quick start

```bash
pip install -e ".[dev]"
pytest

# Launch the GUI
cloudet --cloud /path/to/scan.ply

# Batch reduction
cloudet reduce ~/surveys/proj1 --recipe recipe.json -o geometry.json
```

See [Getting Started](getting-started.md) for detailed installation instructions.

## Package layout

```
cloudet/
  core/        Plane fitting, PLY I/O, spatial indexing, GPU backend
  fit/         Click-driven face extraction and QC
  project/     Project persistence (manifest, groups, settings)
  reduction/   Constructive geometry (session, ops, geometry, frame)
  ui/          Qt GUI (main window, docks, mixins)
```
