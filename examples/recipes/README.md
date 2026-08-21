# Sample reduction recipes

Copy into a project as `recipe.json`, adjust `faces.*.name` / indices to match
your Groups, then:

```bash
cloudet reduce <project> --recipe recipe.json -o geometry.json
```

Or **Load recipe…** in the GUI Reduction dock (**Import from Groups** can bind
the same entities interactively).

| File | Use case |
|------|----------|
| [`tracker_planes.json`](tracker_planes.json) | Two tracker walls + target → beam axis ∩ target |
| [`marker_baseline.json`](marker_baseline.json) | Two marker circles → chord / baseline between centers |
| [`duct_on_wall.json`](duct_on_wall.json) | Cylinder axis ∩ wall plane → hit point |

Circle / cylinder faces need `kind` (`circle` / `cylinder`) and optional
`circle_index` / `cylinder_index`, `diameter_mm`, `diameter_fixed`. See
[Geometry reduction](../../docs/guide/reduction.md).
