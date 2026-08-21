"""Face / cylinder extraction and fitting (pick, main component, multi-plane)."""

from cloudet.fit.mainplane import (
    MainPlaneParams,
    MainPlaneResult,
    extract_main_plane,
    inplane_basis,
    label_components,
)
from cloudet.fit.multiplane import MultiPlaneParams, bimodality_flag, extract_planes
from cloudet.fit.picking import (
    PickParams,
    pick_ball_region,
    pick_cylinder_region,
    pick_cylinder_region_from_cylinder,
    pick_plane_region,
    resolve_cylinder_shell_mm,
)
from cloudet.fit.pipeline import residual_cylinder_map, residual_uv_map

__all__ = [
    "MainPlaneParams",
    "MainPlaneResult",
    "MultiPlaneParams",
    "PickParams",
    "bimodality_flag",
    "extract_main_plane",
    "extract_planes",
    "inplane_basis",
    "label_components",
    "pick_ball_region",
    "pick_cylinder_region",
    "pick_cylinder_region_from_cylinder",
    "pick_plane_region",
    "residual_cylinder_map",
    "residual_uv_map",
    "resolve_cylinder_shell_mm",
]
