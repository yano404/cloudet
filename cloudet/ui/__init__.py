"""cloudet desktop UI package."""

from __future__ import annotations

__all__ = ["CloudetAppWindow", "run_cloudet_qt"]


def __getattr__(name: str):
    if name in __all__:
        from cloudet.ui.main_window import CloudetAppWindow, run_cloudet_qt

        return {
            "CloudetAppWindow": CloudetAppWindow,
            "run_cloudet_qt": run_cloudet_qt,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
