"""Qt/VTK startup helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pyvista as pv
from PySide6.QtCore import qInstallMessageHandler

_QT_MSG_PREV = None
_QT_MSG_FILTER_INSTALLED = False


def _qt_message_filter(mode, context, message: str) -> None:
    if "out of bounds for table with" in message:
        return
    if _QT_MSG_PREV is not None:
        _QT_MSG_PREV(mode, context, message)
        return
    sys.stderr.write(message + "\n")


def install_qt_message_filter() -> None:
    """Hide the known-harmless macOS QTreeWidget accessibility warning."""
    global _QT_MSG_PREV, _QT_MSG_FILTER_INSTALLED
    if _QT_MSG_FILTER_INSTALLED:
        return
    _QT_MSG_PREV = qInstallMessageHandler(_qt_message_filter)
    _QT_MSG_FILTER_INSTALLED = True


# vtkOutputWindow only borrows the Python wrappers, so they must outlive the call.
_VTK_LOG_KEEPALIVE: list = []


def route_vtk_messages_to_file(path: Path) -> Path | None:
    """Send VTK's own errors and warnings to *path* instead of the console.

    ``import pyvista`` unconditionally calls ``send_errors_to_logging()``, which
    hands every VTK message to the unconfigured root logger. Long OpenGL
    messages re-enter that handler and bury the terminal under repeated
    ``PyVista error in handling VTK error message``. Writing straight to a file
    removes the logging round-trip and keeps the original text readable.

    ``CLOUDET_VTK_LOG=0`` keeps PyVista's default; any other value is used as the
    log path.
    """
    setting = os.environ.get("CLOUDET_VTK_LOG", "")
    if setting == "0":
        return None
    if setting not in ("", "1"):
        path = Path(setting).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _VTK_LOG_KEEPALIVE.extend(pv.set_error_output_file(path))
    except Exception:
        return None
    return path
