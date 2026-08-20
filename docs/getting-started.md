# Getting Started

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.24
- PySide6 ≥ 6.5 (for the GUI)

## Installation

```bash
# Full app with Qt UI
pip install -e ".[dev]"

# Optional: Open3D display decimation and RANSAC backend
pip install -e ".[dev,open3d]"

# Optional: CuPy GPU acceleration (NVIDIA + CUDA 12.x)
pip install -e ".[dev,gpu]"
```

## Verify installation

```bash
pytest                              # run the test suite
cloudet --version                   # print version
cloudet --cloud /path/to/scan.ply   # launch the GUI
```

## GPU setup (optional)

3D rendering already uses the GPU via VTK/OpenGL.
Optional **CuPy** accelerates fitting, residual u–v maps, and display voxel downsampling.

```bash
pip install -e ".[dev,gpu]"
pip install "cupy-cuda12x[ctk]"  # if GPU probe fails (missing CUDA headers on WSL)
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
```

In Settings → **Compute backend**: `auto` (CuPy when available), `numpy`, or `cupy`.

CuPy is not required — Mac and CPU-only machines use NumPy.
Clouds under ~50k points stay on CPU even in `auto` mode.
Set `CLOUDET_COMPUTE_BACKEND=numpy` to force CPU.
