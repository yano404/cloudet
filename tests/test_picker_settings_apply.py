"""Unit tests for settings Apply side-effect classification."""

from dataclasses import replace

from cloudet.fit.picking import PickParams
from cloudet.project import ViewSettings
from cloudet.project.settings_apply import classify_settings_apply


def test_float_noise_does_not_count_as_view_change():
    """QDoubleSpinBox round-trip must not trigger a display rebuild."""
    det0 = PickParams()
    det1 = replace(det0, accumulate_threshold_mm=2.5)
    view0 = ViewSettings()
    view1 = replace(
        view0,
        base_point_size=view0.base_point_size + 1e-12,
        display_voxel_size_mm=view0.display_voxel_size_mm + 1e-12,
    )
    fx = classify_settings_apply(det0, det1, view0, view1)
    assert fx.detection_changed
    assert not fx.view_changed
    assert not fx.refresh_display
    assert not fx.update_point_sizes


def test_detection_only_no_display_refresh():
    det0 = PickParams()
    det1 = replace(det0, accumulate_threshold_mm=2.5)
    view = ViewSettings()
    fx = classify_settings_apply(det0, det1, view, view)
    assert fx.detection_changed
    assert not fx.view_changed
    assert not fx.invalidate_grid
    assert not fx.refresh_display
    assert not fx.update_point_sizes


def test_local_radius_invalidates_grid_only():
    det0 = PickParams()
    det1 = replace(det0, local_radius_mm=15.0)
    view = ViewSettings()
    fx = classify_settings_apply(det0, det1, view, view)
    assert fx.detection_changed
    assert fx.invalidate_grid
    assert not fx.refresh_display
    assert not fx.update_point_sizes


def test_display_sampling_triggers_refresh():
    det = PickParams()
    view0 = ViewSettings()
    view1 = replace(view0, display_voxel_size_mm=1.0)
    fx = classify_settings_apply(det, det, view0, view1)
    assert fx.view_changed
    assert fx.refresh_display
    assert not fx.update_point_sizes  # rebuild covers sizes


def test_max_points_triggers_refresh():
    det = PickParams()
    view0 = ViewSettings()
    view1 = replace(view0, display_max_points=1_000_000)
    fx = classify_settings_apply(det, det, view0, view1)
    assert fx.refresh_display


def test_point_size_only_light_update():
    det = PickParams()
    view0 = ViewSettings()
    view1 = replace(view0, base_point_size=2.0, active_point_size=8.0)
    fx = classify_settings_apply(det, det, view0, view1)
    assert fx.view_changed
    assert not fx.refresh_display
    assert fx.update_point_sizes


def test_unchanged_settings():
    det = PickParams()
    view = ViewSettings()
    fx = classify_settings_apply(det, det, view, view)
    assert not fx.detection_changed
    assert not fx.view_changed
    assert not fx.invalidate_grid
    assert not fx.refresh_display
    assert not fx.update_point_sizes


def test_axis_only_no_heavy_work():
    det = PickParams()
    view0 = ViewSettings()
    view1 = replace(view0, axis_size_mm=50.0)
    fx = classify_settings_apply(det, det, view0, view1)
    assert fx.view_changed
    assert not fx.refresh_display
    assert not fx.update_point_sizes


def test_combined_detection_and_display():
    det0 = PickParams()
    det1 = replace(det0, accumulate_threshold_mm=2.0, local_radius_mm=12.0)
    view0 = ViewSettings()
    view1 = replace(view0, display_max_points=500_000)
    fx = classify_settings_apply(det0, det1, view0, view1)
    assert fx.detection_changed
    assert fx.invalidate_grid
    assert fx.refresh_display


def test_compute_backend_is_detection_only():
    det0 = PickParams()
    det1 = replace(det0, compute_backend="cupy")
    view = ViewSettings()
    fx = classify_settings_apply(det0, det1, view, view)
    assert fx.detection_changed
    assert not fx.view_changed
    assert not fx.refresh_display
    assert not fx.invalidate_grid
