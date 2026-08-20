"""Application icons shipped with the UI package.

Uses the mark-only asset (no wordmark text). Wordmark lives under ``docs/assets/``
for README / documentation branding.
"""

from __future__ import annotations

from importlib import resources

from PySide6.QtGui import QIcon


def app_icon() -> QIcon:
    """Return the cloudet window / Dock icon (mark only; SVG preferred, PNG fallback)."""
    try:
        root = resources.files("cloudet.ui.assets")
    except (ModuleNotFoundError, TypeError, AttributeError):
        return QIcon()

    for name in ("cloudet-icon.svg", "cloudet-icon.png"):
        ref = root.joinpath(name)
        try:
            if not ref.is_file():
                continue
        except (OSError, TypeError, AttributeError):
            continue
        try:
            with resources.as_file(ref) as path:
                icon = QIcon(str(path))
        except (FileNotFoundError, OSError, TypeError):
            continue
        if not icon.isNull():
            return icon
    return QIcon()
