# cloudet

<p align="center">
  <img src="assets/cloudet-wordmark.svg" alt="cloudet" width="360">
</p>

Tool for reducing detector positions and relative geometry from 3D point clouds
(e.g. FARO Quantum-S), aimed at nuclear-physics detector surveys.

## Features

- **Click-to-fit extraction** — planes (default), cylinders, and planar circles from UV selection
- **Constructive geometry reduction** — declarative recipe turns fitted faces into axes, intersection points, and offsets (sample recipes: `examples/recipes/` in the repo)
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
