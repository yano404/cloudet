"""cloudet: detector positioning from 3D point clouds.

Units are millimetres throughout (FARO export convention).
"""

from cloudet.plane import (
    Plane,
    FitResult,
    fit_plane_lsq,
    ransac_plane,
    robust_fit_plane,
    residual_stats,
    mad_sigma,
)
from cloudet.plyio import read_ply_xyz, write_ply_xyz

__all__ = [
    "Plane",
    "FitResult",
    "fit_plane_lsq",
    "ransac_plane",
    "robust_fit_plane",
    "residual_stats",
    "mad_sigma",
    "read_ply_xyz",
    "write_ply_xyz",
]

__version__ = "0.1.0"
