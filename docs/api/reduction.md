# cloudet.reduction

Constructive geometry: bind fitted planes, cylinders (axis lines), and
circles (centers), apply recipe steps, export `geometry.json` and a slim
sibling `geometry_summary.json`.

- [`session`](reduction-session.md) — `ReductionSession`, recipe load/run/export
- [`ops`](reduction-ops.md) — operation metadata used by CLI and GUI forms
- [`geometry`](reduction-geometry.md) — intersections, offsets, rotations
- [`frame`](reduction-frame.md) — Align Z pose and virtual aligned entities

## Typical workflow

```python
import numpy as np
from cloudet.core import Plane
from cloudet.reduction import ReductionSession

sess = ReductionSession()
sess.bind_scanned("left", Plane(np.array([1.0, 0.0, 0.0]), 50.0), group_name="G0", group_id=0)
sess.bind_scanned("front", Plane(np.array([0.0, 1.0, 0.0]), 20.0), group_name="G1", group_id=1)
sess.offset("left_in", "left", 12.0)
sess.offset("front_in", "front", 12.0)
sess.intersect_planes("beam_axis", "left_in", "front_in")
```
