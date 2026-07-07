"""Command line interface: ``detpos fit <groups_dir> -o <out_dir>``."""

from __future__ import annotations

import argparse
import sys

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
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "fit":
        params = FitParams(
            ransac_threshold_mm=args.ransac_threshold,
            ransac_iterations=args.ransac_iterations,
            seed=args.seed,
            strict_threshold_mm=args.strict_threshold,
            sigma_factor=args.sigma_factor,
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


if __name__ == "__main__":
    raise SystemExit(main())
