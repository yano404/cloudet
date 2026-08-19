"""PyQtGraph helpers for residual u–v map."""

from __future__ import annotations

import numpy as np

def _rdbu_r_colormap():
    """Diverging blue–white–red map matching the former matplotlib RdBu_r."""
    import pyqtgraph as pg

    pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    colors = np.array(
        [
            [5, 48, 97, 255],
            [67, 147, 195, 255],
            [247, 247, 247, 255],
            [214, 96, 77, 255],
            [103, 0, 31, 255],
        ],
        dtype=np.ubyte,
    )
    return pg.ColorMap(pos, colors)


class _UVSelectViewBox:
    """Factory: Cmd/Ctrl+drag selects a rectangle; otherwise normal pan/zoom.

    On macOS the Command key is Qt.MetaModifier; Control is still accepted.
    """

    @staticmethod
    def create(on_rect):
        import pyqtgraph as pg
        from PySide6.QtCore import Qt

        class VB(pg.ViewBox):
            def __init__(self):
                super().__init__(enableMenu=False)
                self._on_rect = on_rect
                self._sel_start = None
                self._rubber = None
                self.setAspectLocked(True)

            def mouseDragEvent(self, ev, axis=None):
                # Cmd (Meta) on macOS; Ctrl on Windows/Linux (and macOS Control).
                select_mod = ev.modifiers() & (
                    Qt.ControlModifier | Qt.MetaModifier
                )
                if ev.button() != Qt.LeftButton or not select_mod:
                    return super().mouseDragEvent(ev, axis)
                ev.accept()
                pt = self.mapSceneToView(ev.scenePos())
                if ev.isStart():
                    self._sel_start = pt
                    if self._rubber is not None:
                        try:
                            self.removeItem(self._rubber)
                        except Exception:
                            pass
                    self._rubber = pg.RectROI(
                        [pt.x(), pt.y()],
                        [1e-6, 1e-6],
                        pen=pg.mkPen("#1a4a9a", width=1.5),
                        movable=False,
                        resizable=False,
                    )
                    self.addItem(self._rubber)
                elif self._sel_start is None:
                    return
                elif ev.isFinish():
                    u0, u1 = sorted((self._sel_start.x(), pt.x()))
                    v0, v1 = sorted((self._sel_start.y(), pt.y()))
                    if self._rubber is not None:
                        try:
                            self.removeItem(self._rubber)
                        except Exception:
                            pass
                        self._rubber = None
                    self._sel_start = None
                    if (u1 - u0) > 0 and (v1 - v0) > 0 and self._on_rect is not None:
                        self._on_rect(u0, u1, v0, v1)
                else:
                    u0, u1 = sorted((self._sel_start.x(), pt.x()))
                    v0, v1 = sorted((self._sel_start.y(), pt.y()))
                    if self._rubber is not None:
                        self._rubber.setPos([u0, v0], update=False)
                        self._rubber.setSize(
                            [max(u1 - u0, 1e-6), max(v1 - v0, 1e-6)], update=True
                        )

        return VB()
