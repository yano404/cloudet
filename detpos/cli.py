"""Command line interface.

- ``detpos pick <project_dir> [--pcd <cloud>]`` : interactive picker GUI
- ``detpos fit <groups_dir> -o <out_dir>``      : batch plane fitting
"""

from __future__ import annotations

import argparse
import sys

from detpos.mainplane import MainPlaneParams
from detpos.pipeline import FitParams, fit_groupset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="detpos", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser(
        "fit",
        help="fit planes to all groups in a directory (legacy or new format)",
    )
    f.add_argument("groups_dir", help="project dir, groups dir, or legacy groups_out")
    f.add_argument("-o", "--out", required=True, help="output directory for fits")
    f.add_argument("--ransac-threshold", type=float, default=0.5,
                   help="RANSAC selection threshold in mm (default: 0.5)")
    f.add_argument("--ransac-iterations", type=int, default=1000)
    f.add_argument("--strict-threshold", type=float, default=None,
                   help="strict refit threshold in mm (default: adaptive 3*mad_sigma)")
    f.add_argument("--sigma-factor", type=float, default=3.0,
                   help="adaptive threshold = factor * mad_sigma (default: 3.0)")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--no-uv-maps", action="store_true",
                   help="skip residual u-v map PNGs")
    f.add_argument("--uv-bins", type=int, default=200)
    f.add_argument("--simple", action="store_true",
                   help="plain RANSAC+refit without main-component extraction")
    f.add_argument("--max-threshold", type=float, default=0.3,
                   help="ceiling for the adaptive threshold in mm (default: 0.3)")
    f.add_argument("--cell-size", type=float, default=5.0,
                   help="connectivity grid cell size in mm (default: 5.0)")

    pk = sub.add_parser("pick", help="interactive plane picker GUI (requires open3d)")
    pk.add_argument("project_dir", help="project directory (created if missing)")
    pk.add_argument("--pcd", default=None, help="point cloud file to load at startup")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "fit":
        if args.simple:
            params = FitParams(
                ransac_threshold_mm=args.ransac_threshold,
                ransac_iterations=args.ransac_iterations,
                seed=args.seed,
                strict_threshold_mm=args.strict_threshold,
                sigma_factor=args.sigma_factor,
            )
        else:
            params = MainPlaneParams(
                ransac_threshold_mm=min(args.ransac_threshold, args.max_threshold),
                ransac_iterations=args.ransac_iterations,
                seed=args.seed,
                sigma_factor=args.sigma_factor,
                max_threshold_mm=args.max_threshold,
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

    elif args.command == "pick":
        try:
            from detpos.picker_gui import run_picker  # lazy: needs open3d
        except ImportError as e:
            print(f"error: the picker GUI requires open3d ({e})", file=sys.stderr)
            return 1
        run_picker(args.project_dir, args.pcd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
