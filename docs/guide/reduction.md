# Geometry Reduction

After Fit + save, faces live in `groups/group_*.json`
(`fit.planes[].normal` + `d`). Analysis parameters — virtual axes,
beam-on-target points, drawing offsets — are derived with a declarative
recipe (no point cloud required).

Groups may also store **cylinders** and **circles** (`fit.cylinders[]` /
`fit.circles[]`) with drawing diameter `diameter_mm` and optional
`diameter_fixed`.

**Circles on a face (photogrammetry markers):**

1. Fit kind = **plane** (pick / Fit) — supporting plane + UV residual map  
2. Residuals → select rim → optional **Fix Φ** / **Φ (mm)** beside the button →
   **Fit circle on selection** — appends `cir0`, `cir1`, …  
   The supporting Groups plane is **locked**; only the in-plane center
   (and free Φ) are estimated, so the center lies on that face.

There is no separate Fit kind=`circle`; marker holes have no circumference
points, so circles are always added from the UV selection. Recipe still binds
with `"kind": "circle"` and optional `"circle_index"`.
**Cylinder (ducts / pipes):** Fit kind = `cylinder`. Pick is always
**3-point seed**: click three non-collinear points on the circumference (same
cross-section works best). That builds the axis from the circumcircle, then
expands a radial shell and refines. **Esc** clears an in-progress seed.
Set **Fix diameter Φ** when the drawing diameter is known.

**Cylinder QC:** Residuals dock shows an unrolled **s–z map** (arc length
`s = r·θ` × axial `z`) colored by signed radial residual `ρ − Φ/2`, plus a
histogram. Raise **range ±** if the colorbar saturates (duct residuals are
often larger than plane µm-scale).

Fit polish uses a **geometric** nonlinear refine (minimize `ρ − r`), after an
algebraic seed — better on partial arcs than algebraic circle alone. Prefer
**Fix Φ** when the drawing diameter is known.

**Cylinder shell (Settings → Detection):** under **3 CYLINDER SHELL**:
- **Cylinder shell ± (radial)** — half-thickness of `|ρ − r|` band (`0` = auto
  `max(6 mm, 0.15·r)`)
- **Cylinder axial half-length** — half-length along the axis from the pick /
  seed (`0` = auto `max(40 mm, 1.0·r)`)

Apply settings before picking; shell size does not rebuild the spatial index.

In the recipe, bind them as faces with `"kind": "cylinder"`
(→ axis **line**) or `"kind": "circle"` (→ center **point**):

```json
"bore": {
  "from": "group",
  "name": "G3",
  "kind": "cylinder",
  "diameter_mm": 80.0,
  "diameter_fixed": true
},
"marker_a": {
  "from": "group",
  "name": "G3",
  "kind": "circle",
  "circle_index": 0,
  "diameter_mm": 50.0,
  "diameter_fixed": true
}
```

In the GUI **Reduction** dock, **Import** lists fitted planes, cylinders, and
circles from Groups. Cylinder axes become line entities (`G3_cyl0`); circle
centers become point entities (`G3_cir0`). The same kinds load from
`recipe.json` / `cloudet reduce`.

## Sample recipes

Ready-to-copy JSON in the repository under `examples/recipes/`:

| File | Use case |
|------|----------|
| `tracker_planes.json` | Tracker walls → beam axis ∩ target |
| `marker_baseline.json` | Marker circles → chord / baseline (+ distances) |
| `duct_on_wall.json` | Cylinder axis ∩ wall → hit point (+ angle) |

Adjust `faces.*.name` / indices to your Groups, then
`cloudet reduce <project> --recipe …` or **Load recipe…** in the GUI.
See also `examples/recipes/README.md`.

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

A sibling **`geometry_summary.json`** is also written (CLI and GUI): names and
coordinates only, preferring the aligned frame when present, otherwise survey.
If recipe `export` is non-empty, only those ids are listed.

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
