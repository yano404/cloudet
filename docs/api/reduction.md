# cloudet.reduction

Constructive geometry reduction: recipe-driven session that turns fitted
planes into derived geometry (axes, intersection points, offsets, rotations).

## Overview

The reduction subsystem derives analysis parameters from fitted planes
using a declarative recipe — no point cloud required at this stage.

- **`ReductionSession`** — the central stateful object. Scanned planes are
  bound, then construct steps (offset, intersect, rotate, …) build derived
  entities. The session can export a recipe JSON and a geometry JSON.
- **`ops.py`** — metadata for every construct operation. This drives both
  the CLI dispatch and the GUI form auto-generation.
- **`geometry.py`** — pure-math functions: plane offsets, line–plane
  intersections, rotations (Rodrigues), projections, distances, and angles.
- **`frame.py`** — the Align Z rigid frame that maps a chosen axis to +Z
  and an origin to (0,0,0), plus virtual aligned entities.

## Typical workflow (programmatic)

```python
import numpy as np
from cloudet.core import Plane
from cloudet.reduction import ReductionSession

# Create a session and bind scanned planes
sess = ReductionSession()
sess.bind_scanned("left",  Plane(np.array([1,0,0.]), 50.0),
                  group_name="G0", group_id=0)
sess.bind_scanned("front", Plane(np.array([0,1,0.]), 20.0),
                  group_name="G1", group_id=1)
sess.bind_scanned("target", Plane(np.array([0,0,1.]), -100.0),
                  group_name="G2", group_id=2)

# Construct derived geometry step-by-step
sess.offset("left_in", "left", 12.0)
sess.offset("front_in", "front", 12.0)
sess.intersect_planes("beam_axis", "left_in", "front_in")
sess.intersect_line_plane("beam_on_target", "beam_axis", "target")

# Query results
print(sess.kind_of("beam_axis"))        # "line"
print(sess.point("beam_on_target"))     # ndarray (3,)

# Export
recipe = sess.to_recipe(export=["beam_axis", "beam_on_target"])
result = sess.to_result(source_project="/path/to/project")
```

## Rotation example

```python
# Rotate a point 90° about an axis (right-hand rule)
sess.rotate_point_about_line("rotated_pt", "beam_on_target", "beam_axis", 90.0)

# Rotate a line 45° about another axis
sess.rotate_line_about_line("tilted_axis", "beam_axis", "aligned.z", 45.0)

# Rotate a plane
sess.rotate_plane_about_line("tilted_plane", "target", "beam_axis", 30.0)
```

---

## Session

### ReductionSession

The main interactive session. Tracks scanned faces, construct history,
visibility, and anchors. Exports recipe and result JSON.

::: cloudet.reduction.session.ReductionSession
    options:
      members:
        - bind_scanned
        - apply_step
        - offset
        - intersect_planes
        - intersect_three_planes
        - intersect_line_plane
        - intersect_normal_plane
        - line_from_point_normal
        - line_from_two_points
        - midpoint_line_planes
        - plane_from_plane_point
        - plane_from_line_point
        - plane_from_two_lines
        - rotate_plane_about_line
        - rotate_point_about_line
        - rotate_line_about_line
        - kind_of
        - plane
        - line
        - point
        - to_recipe
        - to_result
        - unique_id

### ReductionResult

The output of a completed reduction: entity records in survey coordinates,
optional aligned-frame copy, measures, and the originating recipe.

::: cloudet.reduction.session.ReductionResult

### ConstructPreview

Lightweight preview of a construct step before applying it (used for
live-preview overlays in the GUI).

::: cloudet.reduction.session.ConstructPreview

### run_reduction

Batch entry point — runs a full recipe against a project directory
and returns a `ReductionResult`.

::: cloudet.reduction.session.run_reduction

### load_recipe

Load and validate a recipe JSON file.

::: cloudet.reduction.session.load_recipe

### export_reduction_result

Serialize a `ReductionResult` to a geometry JSON file.

::: cloudet.reduction.session.export_reduction_result

---

## Operation metadata

Operation definitions drive both CLI dispatch and GUI form generation.
Each `ReductionOpDef` declares operand types, scalar parameters, labels,
and hints.

### REDUCTION_OPS

Tuple of all supported construct operations, in GUI display order.

::: cloudet.reduction.ops.REDUCTION_OPS

### ReductionOpDef

::: cloudet.reduction.ops.ReductionOpDef
    options:
      members:
        - key
        - op
        - label
        - button_label
        - input_kind
        - output_kind
        - operands
        - scalars
        - missing_msg
        - operands_must_differ
        - hint

### OperandField

Describes one operand input (plane, line, or point) for a construct step.

::: cloudet.reduction.ops.OperandField

### ScalarField

Describes one scalar parameter (e.g. angle, distance) for a construct step.

::: cloudet.reduction.ops.ScalarField

### build_construct_step

Build a construct-step dict from GUI form values.

::: cloudet.reduction.ops.build_construct_step

---

## Geometry primitives

Pure-math functions for constructive geometry. All operate on `Plane`
and `Line` objects; angles are in degrees, distances in mm.

### Line

Parametric line `x(t) = point + t · direction` with `|direction| = 1`.
Direction is sign-normalized so the largest component is positive.

::: cloudet.reduction.geometry.Line
    options:
      members:
        - point
        - direction
        - from_point_direction

### Intersections

::: cloudet.reduction.geometry.intersect_planes

::: cloudet.reduction.geometry.intersect_three_planes

::: cloudet.reduction.geometry.intersect_line_plane

::: cloudet.reduction.geometry.intersect_normal_plane

### Plane construction

::: cloudet.reduction.geometry.offset_plane

::: cloudet.reduction.geometry.plane_from_line_point

::: cloudet.reduction.geometry.plane_from_plane_point

::: cloudet.reduction.geometry.plane_from_two_lines

### Rotations

All rotations use Rodrigues' formula. Positive angle follows the right-hand
rule around the axis direction.

::: cloudet.reduction.geometry.rotate_plane_about_line

::: cloudet.reduction.geometry.rotate_point_about_line

::: cloudet.reduction.geometry.rotate_line_about_line

### Projections and distances

::: cloudet.reduction.geometry.project_point_to_plane

::: cloudet.reduction.geometry.project_point_to_line

::: cloudet.reduction.geometry.distance_points

::: cloudet.reduction.geometry.distance_point_plane

::: cloudet.reduction.geometry.distance_point_line

### Angles

::: cloudet.reduction.geometry.angle_planes_deg

::: cloudet.reduction.geometry.angle_lines_deg

::: cloudet.reduction.geometry.angle_line_plane_deg

### Display helpers

::: cloudet.reduction.geometry.plane_patch_corners

::: cloudet.reduction.geometry.line_segment_points

::: cloudet.reduction.geometry.axis_arrow_points

---

## Aligned frame

When a FRAME axis and origin are set, the session exposes virtual aligned
entities (`aligned.origin`, `aligned.x/y/z`, `aligned.yz/zx/xy`) that
can be used as operands in construct steps.

### RigidFrame

The Align Z rigid transform: maps a chosen axis to +Z and origin to (0,0,0),
with optional yaw control.

::: cloudet.reduction.frame.RigidFrame

### Virtual entity constructors

::: cloudet.reduction.frame.aligned_axis_line

::: cloudet.reduction.frame.aligned_origin_point

::: cloudet.reduction.frame.aligned_plane

### Frame coordinate transform

::: cloudet.reduction.frame.result_in_frame
