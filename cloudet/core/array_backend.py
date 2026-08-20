"""Optional CuPy-backed array operations with CPU fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

__all__ = [
    "COMPUTE_BACKENDS",
    "DISPLAY_BACKENDS",
    "DevicePoints",
    "GPU_MIN_POINTS",
    "ArrayContext",
    "cupy_available",
    "cupy_unavailable_reason",
    "device_name",
    "get_context",
    "resolve_compute_backend",
    "set_default_backend",
]

COMPUTE_BACKENDS = ("auto", "numpy", "cupy")
DISPLAY_BACKENDS = ("auto", "numpy", "open3d", "cupy")
GPU_MIN_POINTS = 10_000  # auto skips GPU below this (transfer overhead)

_default_backend: str | None = None
_cupy_probe_ok: bool | None = None
_cupy_probe_reason: str | None = None


def cupy_unavailable_reason() -> str | None:
    """Human-readable reason when [`cupy_available`][cloudet.core.array_backend.cupy_available] is False."""
    if _cupy_probe_ok is None:
        cupy_available()
    return _cupy_probe_reason


def _probe_cupy() -> bool:
    """True only if CuPy can compile and run a tiny reduction kernel."""
    global _cupy_probe_reason
    try:
        import cupy as cp
    except ImportError:
        _cupy_probe_reason = "cupy is not installed (pip install cupy-cuda12x[ctk])"
        return False
    if not cp.cuda.is_available():
        _cupy_probe_reason = "CUDA device not available to CuPy"
        return False
    try:
        # Runtime kernel compile needs CUDA headers on Linux/WSL unless [ctk] is installed.
        probe = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float64)
        float(probe.min())
        _ = float((probe @ probe).sum())
    except RuntimeError as e:
        _cupy_probe_reason = str(e)
        return False
    except Exception as e:
        _cupy_probe_reason = f"{type(e).__name__}: {e}"
        return False
    _cupy_probe_reason = None
    return True


def cupy_available() -> bool:
    global _cupy_probe_ok
    if _cupy_probe_ok is None:
        _cupy_probe_ok = _probe_cupy()
    return _cupy_probe_ok


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
            reason = cupy_unavailable_reason() or "CuPy/CUDA not usable"
            raise ImportError(
                f"{required} backend 'cupy' requested but CuPy/CUDA is not available "
                f"({reason}). Try: pip install 'cupy-cuda12x[ctk]' or set CUDA_PATH; "
                "otherwise use backend='auto'/'numpy'."
            )
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

    def to_device(self, x):
        xp = self.xp
        if self.name == "cupy":
            import cupy as cp

            if isinstance(x, cp.ndarray):
                return x if x.dtype == xp.float64 else x.astype(xp.float64)
            arr = np.asarray(x, dtype=np.float64)
            return xp.asarray(arr)
        return np.asarray(x, dtype=np.float64)

    def to_device_bool(self, x):
        xp = self.xp
        if self.name == "cupy":
            import cupy as cp

            if isinstance(x, cp.ndarray):
                return x if x.dtype == bool else x.astype(bool)
            return xp.asarray(np.asarray(x, dtype=bool))
        return np.asarray(x, dtype=bool)

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


@dataclass(frozen=True)
class DevicePoints:
    """Host points uploaded once; reused across multi-stage GPU fits."""

    ctx: ArrayContext
    pts: object  # numpy.ndarray | cupy.ndarray

    @classmethod
    def create(
        cls, points: np.ndarray, compute_backend: str = "auto"
    ) -> "DevicePoints | None":
        """Return a GPU buffer, or ``None`` when compute resolves to numpy."""
        points = np.asarray(points, dtype=np.float64)
        ctx = get_context(compute_backend, n_points=len(points))
        if ctx.name == "numpy":
            return None
        return cls(ctx=ctx, pts=ctx.to_device(points))
