"""Shared Qt widgets and styling for cloudet."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QToolButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from cloudet.ui.constants import GROUP_COLORS

UI_STYLE = """
QFrame#card {
    background: palette(base);
    border: 1px solid palette(mid);
    border-radius: 8px;
}
QLabel#sectionTitle,
QToolButton#sectionTitle {
    color: palette(mid);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
QToolButton#sectionTitle {
    border: none;
    background: transparent;
    padding: 0px;
}
QToolButton#sectionTitle:hover {
    color: palette(text);
}
QLabel#muted {
    color: palette(mid);
    font-size: 11px;
}
/* A stylesheet on a QPushButton drops the native macOS look entirely, so
   every role has to draw its own background, border and states — without
   them a button reads as disabled. */
QPushButton#primaryBtn {
    font-weight: 700;
    border-radius: 6px;
    padding: 5px 10px;
    border: 1px solid palette(highlight);
    background: palette(highlight);
    color: palette(highlighted-text);
}
QPushButton#primaryBtn:hover:!disabled {
    background: palette(highlight);
    border-color: palette(text);
}
QPushButton#primaryBtn:pressed {
    background: palette(dark);
    border-color: palette(dark);
}
QPushButton#secondaryBtn {
    border-radius: 6px;
    padding: 5px 10px;
    border: 1px solid palette(mid);
    background: palette(button);
    color: palette(button-text);
}
QPushButton#secondaryBtn:hover:!disabled {
    border-color: palette(highlight);
}
QPushButton#secondaryBtn:pressed {
    background: palette(midlight);
}
QPushButton#dangerBtn {
    border-radius: 6px;
    padding: 5px 10px;
    border: 1px solid #a44;
    background: palette(button);
    color: #a44;
}
QPushButton#dangerBtn:hover:!disabled {
    background: #a44;
    color: palette(highlighted-text);
}
QPushButton#dangerBtn:pressed {
    background: #833;
    border-color: #833;
    color: palette(highlighted-text);
}
QPushButton#primaryBtn:disabled,
QPushButton#secondaryBtn:disabled,
QPushButton#dangerBtn:disabled {
    background: palette(window);
    border-color: palette(midlight);
    color: palette(mid);
}
QLabel#badgeOk {
    color: #1f7a1f;
    font-weight: 700;
}
QLabel#badgeWarn {
    color: #a05a00;
    font-weight: 700;
}
QTreeWidget {
    border: 1px solid palette(mid);
    border-radius: 6px;
    alternate-background-color: palette(alternate-base);
}
QTreeWidget::item:selected {
    background: palette(highlight);
    color: palette(highlighted-text);
}
QToolTip {
    padding: 8px;
    border: 1px solid palette(mid);
    border-radius: 5px;
    background: palette(base);
    color: palette(text);
}
"""


def _make_collapsible_card(
    heading: str,
    *,
    expanded: bool = True,
    header_extra: QWidget | None = None,
) -> tuple[QFrame, QVBoxLayout]:
    """Card with a clickable title that shows or hides the body."""
    card = QFrame()
    card.setObjectName("card")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(10, 8, 10, 8)
    outer.setSpacing(6)

    toggle = QToolButton()
    toggle.setObjectName("sectionTitle")
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
    toggle.setText(heading)
    toggle.setCheckable(True)
    toggle.setChecked(expanded)
    toggle.setAutoRaise(True)
    toggle.setCursor(Qt.PointingHandCursor)
    toggle.setToolTip("Click to expand or collapse this section.")

    body = QWidget()
    body_lay = QVBoxLayout(body)
    body_lay.setContentsMargins(0, 0, 0, 0)
    body_lay.setSpacing(6)
    body.setVisible(expanded)

    def _on_toggled(checked: bool, *, t=toggle, b=body, c=card) -> None:
        b.setVisible(checked)
        t.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        if checked:
            c.setMaximumHeight(16777215)
        else:
            c.setMaximumHeight(t.sizeHint().height() + 20)

    toggle.toggled.connect(_on_toggled)
    if not expanded:
        card.setMaximumHeight(toggle.sizeHint().height() + 20)
    if header_extra is None:
        outer.addWidget(toggle)
    else:
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(6)
        hdr.addWidget(toggle, stretch=1)
        hdr.addWidget(header_extra)
        outer.addLayout(hdr)
    outer.addWidget(body)
    return card, body_lay


def group_color(gid: int) -> np.ndarray:
    return np.asarray(GROUP_COLORS[gid % len(GROUP_COLORS)], dtype=np.float64)


def _reset_tree_widget(tree: QTreeWidget) -> None:
    """Empty a tree without leaving macOS accessibility on a deleted row.

    ``QTreeWidget.clear()`` can make Qt log
    ``qt.accessibility.table: Cell requested for row N is out of bounds
    for table with 0 rows`` (and occasionally crash) because Cocoa still
    holds a ``QAccessibleTableCell`` for the old current index.
    """
    tree.clearSelection()
    tree.setCurrentItem(None)
    while tree.topLevelItemCount():
        tree.takeTopLevelItem(0)


def _reset_combo(combo: QComboBox) -> None:
    """Clear a combo after dropping the current index (same a11y issue)."""
    combo.setCurrentIndex(-1)
    combo.clear()


def _line_tube_mesh(p0, p1, diameter_mm: float) -> pv.PolyData:
    """Finite tube along p0→p1. Diameter is in the same mm units as the cloud."""
    radius = max(0.5 * float(diameter_mm), 0.05)
    return pv.Line(p0, p1).tube(radius=radius, n_sides=10)
