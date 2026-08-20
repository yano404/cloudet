# Geometry Reduction

After Fit + save, faces live in `groups/group_*.json`
(`fit.planes[].normal` + `d`). Analysis parameters — virtual axes,
beam-on-target points, drawing offsets — are derived with a declarative
recipe (no point cloud required).

## CLI usage

```bash
cloudet reduce <project> --recipe recipe.json -o geometry.json
cloudet migrate <project> [--recipe FILE] [--geometry FILE] [--dry-run]
```

`cloudet migrate` rewrites legacy keys (`abcd`, recipe v1 operands) to the
current schema. Load paths also accept the old form and normalize in memory;
new saves write v2 only.

## Recipe format

```json
{
  "version": 2,
  "units": "mm",
  "faces": {
    "tracker_left":  { "from": "group", "name": "G0" },
    "tracker_front": { "from": "group", "name": "G1" },
    "target":        { "from": "group", "name": "G2" }
  },
  "construct": [
    { "id": "left_in",  "op": "offset", "plane": "tracker_left",  "distance_mm": 12.0 },
    { "id": "front_in", "op": "offset", "plane": "tracker_front", "distance_mm": 12.0 },
    { "id": "beam_axis", "op": "intersect_planes", "plane_a": "left_in", "plane_b": "front_in" },
    { "id": "beam_on_target", "op": "intersect_line_plane", "line": "beam_axis", "plane": "target" }
  ],
  "export": ["beam_axis", "beam_on_target"],
  "frame": { "axis": "beam_axis", "origin": "beam_on_target", "flip_z": false }
}
```

Legacy v1 recipes (`of`, `a`/`b`/`c`, `src`/`dst`) still load; they are
migrated to the keys above.

## Recipe operand keys (version 2)

Construct steps use explicit JSON keys aligned with the Python API.
Meaning by key:

| Key | Meaning |
|-----|---------|
| `plane` | Plane operand (`offset`, intersections, …) |
| `plane_a` / `plane_b` / `plane_c` | Ordered plane operands |
| `point` / `point_a` / `point_b` | Point operands |
| `line` / `line_a` / `line_b` | Line / axis operands |
| `axis` | Rotation axis for `rotate_line_about_line` |
| `source_plane` / `destination_plane` | Normal source / hit plane for `intersect_normal_plane` |
| `distance_mm` | Signed offset along the Hesse normal (mm) |
| `angle_deg` | Rotation angle, right-hand rule (degrees) |

The same glossary lives in code as `cloudet.reduction.ops.RECIPE_OPERAND_KEYS`.

## Supported construct operations

| Operation | Inputs | Output | Description |
|-----------|--------|--------|-------------|
| `offset` | plane, distance_mm | plane | Move plane along its Hesse normal |
| `intersect_planes` | plane a, plane b | line | Intersection line of two planes |
| `intersect_three_planes` | plane a, b, c | point | Intersection point of three planes |
| `intersect_line_plane` | line, plane | point | Line–plane intersection |
| `intersect_normal_plane` | source plane, dest plane | point | Source normal ∩ dest plane |
| `line_from_point_normal` | point, plane | line | Axis through point along plane normal |
| `line_from_two_points` | point a, point b | line | Axis through two points |
| `midpoint_line_planes` | line, plane a, plane b | point | Midpoint of segment cut by two planes |
| `plane_from_plane_point` | plane, point | plane | Parallel plane through point |
| `plane_from_line_point` | line, point | plane | Plane containing line and point |
| `plane_from_two_lines` | line a, line b | plane | Plane containing both coplanar lines |
| `rotate_plane_about_line` | plane, axis, angle_deg | plane | Rigid rotation of a plane |
| `rotate_point_about_line` | point, axis, angle_deg | point | Rigid rotation of a point |
| `rotate_line_about_line` | line, axis, angle_deg | line | Rigid rotation of a line |

All rotations follow the right-hand rule. Positive angle rotates
counter-clockwise when looking along the axis direction.

## Offset sign convention

Positive `distance_mm` moves the plane along its Hesse unit normal
(same sign convention as Fit). Use a negative distance for the opposite side.

## Aligned triad operands

When `recipe.frame` sets `axis` and `origin`, construct steps may reference
virtual ids that follow the aligned coordinate frame:

| id | kind | geometry |
|----|------|----------|
| `aligned.origin` | point | FRAME origin |
| `aligned.x` / `aligned.y` / `aligned.z` | line | Through origin along +X / +Y / +Z |
| `aligned.yz` / `aligned.zx` / `aligned.xy` | plane | Through origin, normals +X / +Y / +Z |

These are virtual entities — not stored in the entity store and not exported
as their own rows in `geometry.json`.

!!! warning
    Aligned entities must **not** be used as FRAME axis, origin, or yaw reference.

## geometry.json output

The output contains the executed recipe plus computed entity records
in **survey** coordinates:

| Key | Contents |
|-----|----------|
| `recipe` | `{ "sha256", "echo" }` — full recipe for reproducibility |
| `export` | ids marked for analysis |
| `frame` | optional Align Z pose |
| `aligned` | optional `{ planes, lines, points }` in the aligned frame |
| `measures` | optional pinned measurements with recomputed values |

When the recipe has `frame`, the output includes `aligned` coordinates
alongside the survey-frame data.

## Interactive reduction (GUI)

The right-hand docks are tabbed: **Residuals** / **Reduction** / **Measure**.

1. Choose an **operation** — only that step's inputs appear
2. Pick operands from the combo boxes and set scalar parameters
3. **Apply** to create the entity
4. Toggle visibility in **Entities**; **Save recipe…** / **Export geometry…**

### FRAME (display only)

Pick **Axis** (line) and **Origin** (point), then **Align Z** to align the view.
Optional **XY** maps a line or plane normal onto ±X / ±Y.
Once set, aligned operands appear in the matching combo boxes.

### Measure

Read distances and angles from constructed entities:

- Distance: point–point, point–plane, point–line (unsigned, mm)
- Angle: plane–plane, line–line, line–plane (degrees)
- **Add measurement** pins the result to the recipe and geometry.json
