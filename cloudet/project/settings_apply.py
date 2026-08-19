"""Classify picker settings Apply side-effects (GUI-independent).

Used by the Qt picker so that changing detection parameters does not
rebuild the display point cloud, and so the decision is unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

from cloudet.fit.picking import PickParams
from cloudet.project import ViewSettings

__all__ = ["ApplySideEffects", "classify_settings_apply", "values_equal"]


@dataclass(frozen=True)
class ApplySideEffects:
    """What the GUI should do after applying new settings."""

    detection_changed: bool
    view_changed: bool
    invalidate_grid: bool  # local_radius changed -> rebuild VoxelHashGrid on next pick
    refresh_display: bool  # voxel / max_points changed -> recompute display indices
    update_point_sizes: bool  # point-size fields changed, display sampling unchanged


_VIEW_DISPLAY_FIELDS = frozenset(
    {"display_voxel_size_mm", "display_max_points", "display_downsample_backend"}
)
_VIEW_POINT_SIZE_FIELDS = frozenset(
    {"base_point_size", "active_point_size", "inactive_point_size"}
)


def values_equal(a: Any, b: Any, *, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """Equality for settings fields; floats use a small tolerance.

    QDoubleSpinBox round-trips can introduce tiny float noise; exact ``!=``
    would falsely mark View as changed and trigger a multi-million-point
    actor rebuild on Detection-only Apply.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b


def _changed_fields(old, new) -> set[str]:
    return {
        f.name
        for f in fields(old)
        if not values_equal(getattr(old, f.name), getattr(new, f.name))
    }


def classify_settings_apply(
    old_detection: PickParams,
    new_detection: PickParams,
    old_view: ViewSettings,
    new_view: ViewSettings,
) -> ApplySideEffects:
    """Decide Apply side-effects from old vs new settings."""
    det_changed = _changed_fields(old_detection, new_detection)
    view_changed = _changed_fields(old_view, new_view)

    invalidate_grid = "local_radius_mm" in det_changed
    refresh_display = bool(view_changed & _VIEW_DISPLAY_FIELDS)
    sizes_changed = bool(view_changed & _VIEW_POINT_SIZE_FIELDS)
    # Full display rebuild already applies new sizes; only need a light
    # update when sampling is unchanged.
    update_point_sizes = sizes_changed and not refresh_display

    return ApplySideEffects(
        detection_changed=bool(det_changed),
        view_changed=bool(view_changed),
        invalidate_grid=invalidate_grid,
        refresh_display=refresh_display,
        update_point_sizes=update_point_sizes,
    )
