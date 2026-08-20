"""Command line interface.

Primary entry (GUI-first)::

    cloudet [project_dir] [--cloud <file>]
    cloudet reduce <project> --recipe recipe.json [-o geometry.json]
    cloudet migrate <project> [--recipe FILE] [--geometry FILE] [--dry-run]
    cloudet version

Interactive pick / Fit / residual QC / save all live in the app.
``reduce`` derives analysis geometry from saved fits + a recipe.
``migrate`` rewrites legacy recipe / geometry / group JSON keys in place.
"""

from __future__ import annotations

import argparse
import json
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
            "  cloudet migrate ~/surveys/run1 --dry-run\n"
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


def build_migrate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cloudet migrate",
        description=(
            "Rewrite legacy on-disk JSON to the current schema: recipe operand "
            "keys (v2), plane records (normal+d), and entity parents. Always "
            "scans <project>/groups/group_*.json; optionally migrates recipe "
            "and geometry files."
        ),
    )
    p.add_argument("project_dir", help="project directory containing groups/")
    p.add_argument(
        "--recipe",
        default=None,
        metavar="FILE",
        help="recipe JSON to migrate (default: geometry_recipe.json / recipe.json under project if present)",
    )
    p.add_argument(
        "--geometry",
        default=None,
        metavar="FILE",
        help="geometry.json to migrate (default: geometry.json under project if present)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report files that would change without writing",
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
    from cloudet.reduction import (
        geometry_summary_path,
        load_recipe,
        run_reduction,
        write_geometry_json,
        write_geometry_summary_json,
    )

    args = build_reduce_parser().parse_args(argv)
    project = Path(args.project_dir)
    try:
        recipe = load_recipe(args.recipe)
        result = run_reduction(project, recipe)
        out = write_geometry_json(args.output, result)
        summary_out = write_geometry_summary_json(
            geometry_summary_path(out), result
        )
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
    print(f"wrote {summary_out}  (slim names + coordinates)")
    return 0


def _run_migrate(argv: list[str]) -> int:
    from cloudet.project.schema import migrate_project

    args = build_migrate_parser().parse_args(argv)
    project = Path(args.project_dir)
    if not project.is_dir():
        print(f"error: project directory not found: {project}", file=sys.stderr)
        return 1
    try:
        changed = migrate_project(
            project,
            recipe_path=args.recipe,
            geometry_path=args.geometry,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not changed:
        print("nothing to migrate")
        return 0
    verb = "would update" if args.dry_run else "updated"
    print(f"{verb} {len(changed)} file(s):")
    for path in changed:
        print(f"  {path}")
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

    if argv and argv[0] == "migrate":
        return _run_migrate(argv[1:])

    args = build_app_parser().parse_args(argv)
    return _run_app(args)


if __name__ == "__main__":
    raise SystemExit(main())
