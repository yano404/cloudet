"""CLI dispatch: default app launch; deprecated fit; version."""

from __future__ import annotations

import io
from contextlib import redirect_stderr

import cloudet
from cloudet.cli import build_app_parser, build_fit_parser, main


def test_version():
    assert main(["version"]) == 0


def test_app_parser_cloud_alias():
    args = build_app_parser().parse_args(["proj", "--cloud", "a.ply"])
    assert args.project_dir == "proj"
    assert args.cloud == "a.ply"
    args2 = build_app_parser().parse_args(["--pcd", "b.ply"])
    assert args2.cloud == "b.ply"
    assert args2.project_dir == "cloudet_result"


def test_pick_alias_dispatches_to_app_parser():
    # Does not launch GUI: patch _run_app.
    import cloudet.cli as cli

    called = {}

    def fake_run(args):
        called["project_dir"] = args.project_dir
        called["cloud"] = args.cloud
        return 0

    orig = cli._run_app
    cli._run_app = fake_run
    try:
        assert main(["pick", "myproj", "--cloud", "c.ply"]) == 0
    finally:
        cli._run_app = orig
    assert called == {"project_dir": "myproj", "cloud": "c.ply"}


def test_fit_deprecated_warns(tmp_path):
    import cloudet.cli as cli

    def fake_fit(args):
        return 0

    orig = cli._run_fit
    cli._run_fit = fake_fit
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            rc = main(["fit", str(tmp_path), "-o", str(tmp_path / "out")])
    finally:
        cli._run_fit = orig
    assert rc == 0
    assert "deprecated" in err.getvalue().lower()


def test_fit_parser_requires_out():
    p = build_fit_parser()
    try:
        p.parse_args(["groups_only"])
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_package_version_matches():
    assert cloudet.__version__
