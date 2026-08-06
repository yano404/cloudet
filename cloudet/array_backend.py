"""Optional CuPy-backed array operations with CPU fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

__all__ = [
    "COMPUTE_BACKENDS",
    "DISPLAY_BACKENDS",
    "GPU_MIN_POINTS",
    "ArrayContext",
    "cupy_available",
    "device_name",
    "get_context",
    "resolve_compute_backend",
    "set_default_backend",
]

COMPUTE_BACKENDS = ("auto", "numpy", "cupy")
DISPLAY_BACKENDS = ("auto", "numpy", "open3d", "cupy")
GPU_MIN_POINTS = 50_000

_default_backend: str | None = None


def cupy_available() -> bool:
    try:
        import cupy as cp

        return bool(cp.cuda.is_available())
    except ImportError:
        return False


def device_name() -> str | None:
    if not cupy_available():
        return None
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def _resolve_name(name: str, *, n_points: int | None, required: str) -> str:
    name = (name or "auto").lower()
    if name == "auto":
        env = os.environ.get("CLOUDET_COMPUTE_BACKEND")
        if env:
            name = env.lower()
    if name == "numpy":
        return "numpy"
    if name == "cupy":
        if not cupy_available():
            raise ImportError(
                f"{required} backend 'cupy' requested but CuPy/CUDA is not available "
                "(pip install cupy-cuda12x, or use backend='auto'/'numpy')"
            )
        if n_points is not None and n_points < GPU_MIN_POINTS:
            return "numpy"
        return "cupy"
    if name == "auto":
        if n_points is not None and n_points < GPU_MIN_POINTS:
            return "numpy"
        if cupy_available():
            return "cupy"
        return "numpy"
    raise ValueError(
        f"unknown compute backend {name!r} (choose from {', '.join(COMPUTE_BACKENDS)})"
    )


def resolve_compute_backend(name: str = "auto", *, n_points: int | None = None) -> str:
    """Return ``'numpy'`` or ``'cupy'``."""
    global _default_backend
    if name == "auto" and _default_backend is not None:
        name = _default_backend
    return _resolve_name(name, n_points=n_points, required="compute")


def set_default_backend(name: str | None) -> None:
    """Set session default for ``compute_backend='auto'`` (GUI startup)."""
    global _default_backend
    _default_backend = name


@dataclass(frozen=True)
class ArrayContext:
    """NumPy or CuPy execution context."""

    name: str  # "numpy" | "cupy"

    @property
    def xp(self):
        if self.name == "cupy":
            import cupy as cp

            return cp
        return np

    def to_device(self, x: np.ndarray):
        xp = self.xp
        arr = np.asarray(x, dtype=np.float64)
        if self.name == "numpy":
            return arr
        return xp.asarray(arr)

    def asnumpy(self, x) -> np.ndarray:
        if self.name == "numpy":
            return np.asarray(x, dtype=np.float64)
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return np.asarray(x, dtype=np.float64)

    def asbool(self, x) -> np.ndarray:
        return self.asnumpy(x).astype(bool, copy=False)


def get_context(backend: str = "auto", *, n_points: int | None = None) -> ArrayContext:
    return ArrayContext(resolve_compute_backend(backend, n_points=n_points))
