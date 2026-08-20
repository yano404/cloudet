"""Face extraction and plane fitting (pick, main component, multi-plane)."""

from cloudet.fit.mainplane import (
    MainPlaneParams,
    MainPlaneResult,
    extract_main_plane,
    inplane_basis,
    label_components,
)
from cloudet.fit.multiplane import MultiPlaneParams, bimodality_flag, extract_planes
from cloudet.fit.picking import PickParams, pick_plane_region
from cloudet.fit.pipeline import residual_uv_map

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
    "pick_plane_region",
    "residual_uv_map",
]
