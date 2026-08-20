"""Command line interface.

Primary entry (GUI-first)::

    cloudet [project_dir] [--cloud <file>]
    cloudet reduce <project> --recipe recipe.json [-o geometry.json]
    cloudet version

Interactive pick / Fit / residual QC / save all live in the app.
``reduce`` derives analysis geometry from saved fits + a recipe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
            "  cloudet reduce ~/surveys/run1 --recipe recipe.json -o geometry.json\n"
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


def build_reduce_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cloudet reduce",
        description=(
            "Constructive geometry reduction: bind scanned faces from a saved "
            "project, apply offset / intersection steps from a recipe, and "
            "write analysis parameters to geometry.json. If the recipe has "
            "frame metadata, survey coordinates stay at the top level and an "
            "aligned copy is added."
        ),
    )
    p.add_argument("project_dir", help="project directory containing groups/")
    p.add_argument(
        "--recipe",
        required=True,
        metavar="FILE",
        help="reduction recipe JSON (faces + construct + export; optional frame)",
    )
    p.add_argument(
        "-o",
        "--output",
        default="geometry.json",
        metavar="FILE",
        help="output path (default: ./geometry.json)",
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    """App parser (default entry). Kept for callers/tests."""
    return build_app_parser()


def _run_app(args: argparse.Namespace) -> int:
    try:
        from cloudet.app_window import run_cloudet_qt
    except ImportError as e:
        print(
            f"error: the Qt UI requires PySide6 + pyvista + pyvistaqt ({e})\n"
            'install with: pip install -e .',
            file=sys.stderr,
        )
        return 1
    run_cloudet_qt(args.project_dir, cloud_path=args.cloud)
    return 0


def _run_reduce(argv: list[str]) -> int:
    from cloudet.reduction import load_recipe, run_reduction, write_geometry_json

    args = build_reduce_parser().parse_args(argv)
    project = Path(args.project_dir)
    try:
        recipe = load_recipe(args.recipe)
        result = run_reduction(project, recipe)
        out = write_geometry_json(args.output, result)
    except (OSError, ValueError, KeyError, TypeError, NotADirectoryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    n_p = len(result.planes)
    n_l = len(result.lines)
    n_x = len(result.points)
    frame_note = ", survey + aligned" if result.aligned else ", frame: survey"
    print(
        f"wrote {out}  ({n_p} plane(s), {n_l} line(s), {n_x} point(s); "
        f"export={result.exported}{frame_note})"
    )
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("-h", "--help"):
        build_app_parser().parse_args(argv)
        return 0

    if argv and argv[0] == "version":
        print(cloudet.__version__)
        return 0

    if argv and argv[0] == "reduce":
        return _run_reduce(argv[1:])

    args = build_app_parser().parse_args(argv)
    return _run_app(args)


if __name__ == "__main__":
    raise SystemExit(main())
