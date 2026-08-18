"""CLI dispatch: default app launch; reduce; version."""

from __future__ import annotations

import json

import numpy as np

import cloudet
from cloudet.cli import build_app_parser, build_reduce_parser, main
from cloudet.picking import PickParams
from cloudet.plane import Plane
from cloudet.project import SourceInfo, save_group, write_manifest


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


def test_reduce_cli(tmp_path):
    params = PickParams()
    plane = Plane(np.array([0.0, 0.0, 1.0]), -10.0)
    save_group(
        tmp_path,
        0,
        "target",
        np.zeros((3, 3)),
        None,
        plane.as_array(),
        None,
        detection=params,
        fit_summary={
            "planes": [{
                "plane_index": 0,
                "abcd": plane.as_array().tolist(),
                "n_points": 3,
                "status": "ok",
                "reasons": [],
                "bimodal": False,
                "mad_sigma_mm": 0.04,
                "threshold_mm": 0.15,
            }]
        },
    )
    write_manifest(
        tmp_path, SourceInfo(path="s.ply", n_points=3), params, n_groups=1
    )
    recipe = {
        "version": 1,
        "units": "mm",
        "faces": {"target": {"from": "group", "name": "target"}},
        "construct": [],
        "export": ["target"],
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    out = tmp_path / "out" / "geometry.json"
    assert main([
        "reduce",
        str(tmp_path),
        "--recipe",
        str(recipe_path),
        "-o",
        str(out),
    ]) == 0
    doc = json.loads(out.read_text())
    assert "target" in doc["planes"]
    assert doc["planes"]["target"]["provenance"] == "scanned"


def test_reduce_parser_requires_recipe():
    p = build_reduce_parser()
    args = p.parse_args(["proj", "--recipe", "r.json", "-o", "g.json"])
    assert args.project_dir == "proj"
    assert args.recipe == "r.json"
    assert args.output == "g.json"
