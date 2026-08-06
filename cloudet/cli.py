"""Command line interface.

Primary entry (GUI-first)::

    cloudet [project_dir] [--cloud <file>]
    cloudet version

Interactive pick / Fit / residual QC / save all live in the app.
Batch ``cloudet fit`` remains available but is deprecated; prefer Fit in the GUI
(including u–v selection refit). Additional subcommands can be added later for
headless export if needed.

Contract: one pick = one connected physical face = one group = one plane.
GUI ``Split into parallel planes`` / ``fit --multi-plane`` peel parallel faces.
"""

from __future__ import annotations

import argparse
import sys

import cloudet
from cloudet.mainplane import MainPlaneParams
from cloudet.multiplane import MultiPlaneParams
from cloudet.pipeline import FitParams, fit_groupset

# First-token dispatch. A project directory literally named one of these must
# be passed as ``./fit`` (or an absolute path).
_DISPATCH = frozenset({"fit", "pick", "version"})


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
            "\n"
            "Deprecated: cloudet fit <project> -o <out>  (use Fit in the app)\n"
            "Compat:     cloudet pick ...               (same as cloudet ...)\n"
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
    p.add_argument(
        "--ui",
        choices=["qt", "open3d"],
        default="qt",
        help="GUI backend (default: qt = PySide6 + PyVista)",
    )
    return p


def build_fit_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cloudet fit",
        description=(
            "Deprecated batch plane fitting. Prefer Fit inside the app "
            "(u–v map, selection refit, save to the project)."
        ),
    )
    p.add_argument("groups_dir", help="project dir, groups dir, or legacy groups_out")
    p.add_argument("-o", "--out", required=True, help="output directory for fits")
    p.add_argument("--ransac-threshold", type=float, default=0.5,
                   help="RANSAC selection threshold in mm (default: 0.5)")
    p.add_argument("--ransac-iterations", type=int, default=1000)
    p.add_argument("--strict-threshold", type=float, default=None,
                   help="strict refit threshold in mm (default: adaptive 3*mad_sigma)")
    p.add_argument("--sigma-factor", type=float, default=3.0,
                   help="adaptive threshold = factor * mad_sigma (default: 3.0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ransac-backend", choices=["numpy", "open3d"], default="numpy",
                   help="RANSAC selector backend (default: numpy = seeded, "
                        "reproducible; open3d = segment_plane). The final plane "
                        "is always the orthogonal least-squares refit.")
    p.add_argument("--no-uv-maps", action="store_true",
                   help="skip residual u-v map PNGs")
    p.add_argument("--uv-bins", type=int, default=200)
    p.add_argument("--simple", action="store_true",
                   help="plain RANSAC+refit without main-component extraction")
    p.add_argument("--multi-plane", action="store_true",
                   help="peel each group into several planes (default: one "
                        "dominant plane per group, matching one pick = one face)")
    p.add_argument("--single-plane", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--max-planes", type=int, default=5,
                   help="max planes per group in multi-plane mode (default: 5)")
    p.add_argument("--max-threshold", type=float, default=None,
                   help="ceiling for the adaptive threshold in mm "
                        "(default: 0.15 multi / 0.3 single)")
    p.add_argument("--cell-size", type=float, default=5.0,
                   help="connectivity grid cell size in mm (default: 5.0)")
    return p


def build_parser() -> argparse.ArgumentParser:
    """App parser (default entry). Kept for callers/tests."""
    return build_app_parser()


def _run_app(args: argparse.Namespace) -> int:
    if args.ui == "qt":
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
    try:
        from cloudet.picker_gui import run_picker
    except ImportError as e:
        print(f"error: the open3d UI requires open3d ({e})", file=sys.stderr)
        return 1
    run_picker(args.project_dir, args.cloud)
    return 0


def _run_fit(args: argparse.Namespace) -> int:
    if args.simple:
        params = FitParams(
            ransac_threshold_mm=args.ransac_threshold,
            ransac_iterations=args.ransac_iterations,
            ransac_backend=args.ransac_backend,
            seed=args.seed,
            strict_threshold_mm=args.strict_threshold,
            sigma_factor=args.sigma_factor,
        )
    elif args.multi_plane:
        max_thr = args.max_threshold if args.max_threshold is not None else 0.15
        params = MultiPlaneParams(
            plane=MainPlaneParams(
                ransac_threshold_mm=min(0.1, max_thr),
                ransac_iterations=args.ransac_iterations,
                ransac_backend=args.ransac_backend,
                seed=args.seed,
                sigma_factor=args.sigma_factor,
                max_threshold_mm=max_thr,
                cell_size_mm=args.cell_size,
            ),
            max_planes=args.max_planes,
        )
    else:
        max_thr = args.max_threshold if args.max_threshold is not None else 0.3
        params = MainPlaneParams(
            ransac_threshold_mm=min(args.ransac_threshold, max_thr),
            ransac_iterations=args.ransac_iterations,
            ransac_backend=args.ransac_backend,
            seed=args.seed,
            sigma_factor=args.sigma_factor,
            max_threshold_mm=max_thr,
            cell_size_mm=args.cell_size,
        )
    try:
        fit_groupset(
            args.groups_dir,
            args.out,
            params=params,
            uv_maps=not args.no_uv_maps,
            uv_bins=args.uv_bins,
        )
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("-h", "--help"):
        build_app_parser().parse_args(argv)
        return 0

    if argv and argv[0] == "version":
        print(cloudet.__version__)
        return 0

    if argv and argv[0] == "fit":
        print(
            "warning: 'cloudet fit' is deprecated; use Fit in the app "
            "(including u–v selection refit). "
            "This batch entry may be removed in a future release.",
            file=sys.stderr,
        )
        return _run_fit(build_fit_parser().parse_args(argv[1:]))

    if argv and argv[0] == "pick":
        # Compat alias for the old subcommand name.
        argv = argv[1:]

    args = build_app_parser().parse_args(argv)
    return _run_app(args)


if __name__ == "__main__":
    raise SystemExit(main())
