"""Core types, I/O, and spatial indexing shared across all subsystems."""

from cloudet.core.array_backend import ArrayContext, DevicePoints, cupy_available, get_context
from cloudet.core.circle import Circle, CircleFitResult, robust_fit_circle
from cloudet.core.cylinder import (
    Cylinder,
    CylinderFitResult,
    cylinder_from_three_points,
    refine_cylinder_geometric,
    robust_fit_cylinder,
)
from cloudet.core.neighbors import VoxelHashGrid, display_xyz
from cloudet.core.plane import (
    FitResult,
    Plane,
    fit_plane_lsq,
    mad_sigma,
    ransac_plane,
    residual_stats,
    robust_fit_plane,
)
from cloudet.core.plyio import read_ply_xyz, write_ply_xyz

__all__ = [
    "ArrayContext",
    "Circle",
    "CircleFitResult",
    "Cylinder",
    "CylinderFitResult",
    "DevicePoints",
    "FitResult",
    "Plane",
    "VoxelHashGrid",
    "cupy_available",
    "cylinder_from_three_points",
    "display_xyz",
    "fit_plane_lsq",
    "get_context",
    "mad_sigma",
    "ransac_plane",
    "read_ply_xyz",
    "refine_cylinder_geometric",
    "residual_stats",
    "robust_fit_circle",
    "robust_fit_cylinder",
    "robust_fit_plane",
    "write_ply_xyz",
]
