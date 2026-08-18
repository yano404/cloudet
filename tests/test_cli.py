"""CLI dispatch: default app launch; version."""

from __future__ import annotations

import cloudet
from cloudet.cli import build_app_parser, main


def test_version():
    assert main(["version"]) == 0


def test_app_parser_cloud_alias():
    args = build_app_parser().parse_args(["proj", "--cloud", "a.ply"])
    assert args.project_dir == "proj"
    assert args.cloud == "a.ply"
    args2 = build_app_parser().parse_args(["--pcd", "b.ply"])
    assert args2.cloud == "b.ply"
    assert args2.project_dir == "cloudet_result"


def test_default_dispatches_to_app():
    import cloudet.cli as cli

    called = {}

    def fake_run(args):
        called["project_dir"] = args.project_dir
        called["cloud"] = args.cloud
        return 0

    orig = cli._run_app
    cli._run_app = fake_run
    try:
        assert main(["myproj", "--cloud", "c.ply"]) == 0
    finally:
        cli._run_app = orig
    assert called == {"project_dir": "myproj", "cloud": "c.ply"}


def test_package_version_matches():
    assert cloudet.__version__
