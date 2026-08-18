"""UI constants and color tokens for cloudet desktop app."""

from __future__ import annotations

from cloudet.reduction_ops import GUI_TO_RECIPE_OP, RECIPE_TO_GUI_OP

GROUP_COLORS = [
    (0.90, 0.25, 0.25), (0.25, 0.55, 0.95), (0.20, 0.75, 0.35),
    (0.95, 0.75, 0.20), (0.75, 0.35, 0.85), (0.20, 0.80, 0.80),
    (0.95, 0.45, 0.15), (0.60, 0.60, 0.60),
]

# Reduction overlay colours (3D + tree hints).
RD_PLANE_SCANNED = "#4a90d9"
RD_PLANE_OFFSET = "#e07b39"
RD_AXIS = "#c0392b"
RD_POINT = "#f1c40f"
RD_SELECTED_RING = "#ffffff"
RD_NORMAL = "#2ecc71"
RD_MEASURE = "#16a085"
RD_KIND_LABEL = {"plane": "plane", "line": "line", "point": "point"}
RD_GUI_TO_RECIPE_OP = dict(GUI_TO_RECIPE_OP)
RD_RECIPE_TO_GUI_OP = dict(RECIPE_TO_GUI_OP)

DEPTH_TIP = (
    "Depth candidates along the last picked view ray (near to far).\n"
    "The ray does not stop at the visible face, so the back side can appear.\n"
    "When only one candidate exists, < and > are disabled.\n"
    "Press P first, then use < > only when overlap exists."
)

SETTINGS_HELP_DEFAULT = """
<div style='white-space: normal;'>
<b>SETTINGS WORKFLOW</b>
<p>
1. Press <b>P</b> to test a pick with the current values.<br>
2. Change Detection only when face extraction needs adjustment.<br>
3. Change Display to improve view density or interaction speed.<br>
4. Press <b>Apply</b> to use changes; <b>Save as Default</b> keeps them.
</p>
<span style='color: #777;'>Hover a setting name or value to see what it controls.</span>
</div>
"""

# GUI fit ceiling (mm). Adaptive robust fit never exceeds this.
# Residual plot half-range (±) is controlled separately in the DISPLAY card.
FIT_MAX_THRESHOLD_MM = 0.5
