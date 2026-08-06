"""Command line interface.

Primary entry (GUI-first)::

    cloudet [project_dir] [--cloud <file>]
    cloudet version

Interactive pick / Fit / residual QC / save all live in the app.

Contract: one pick = one connected physical face = one group = one plane.
GUI ``Split into parallel planes`` peels parallel faces when needed.
"""

from __future__ import annotations

import argparse
import sys

import cloudet


def build_app_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cloudet",
        description=(
            "Detector survey tool: open a project and extract planar faces "
            "from a point cloud (pick, fit, residual QC, save)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  cloudet --cloud scan.ply\n"
            "  cloudet ~/surveys/run1 --cloud scan.ply\n"
            "  cloudet version\n"
        ),
    )
    p.add_argument(
        "project_dir",
        nargs="?",
        default="cloudet_result",
        help="project directory (created if missing; default: ./cloudet_result; "
             "also choosable in the app)",
    )
    p.add_argument(
        "--cloud",
        "--pcd",
        dest="cloud",
        default=None,
        metavar="FILE",
        help="point cloud to load at startup (--pcd is an alias)",
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    """App parser (default entry). Kept for callers/tests."""
    return build_app_parser()


def _run_app(args: argparse.Namespace) -> int:
    try:
        from cloudet.picker_qt import run_picker_qt
    except ImportError as e:
        print(
            f"error: the Qt UI requires PySide6 + pyvista + pyvistaqt ({e})\n"
            'install with: pip install -e .',
            file=sys.stderr,
        )
        return 1
    run_picker_qt(args.project_dir, args.cloud)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("-h", "--help"):
        build_app_parser().parse_args(argv)
        return 0

    if argv and argv[0] == "version":
        print(cloudet.__version__)
        return 0

    args = build_app_parser().parse_args(argv)
    return _run_app(args)


if __name__ == "__main__":
    raise SystemExit(main())
