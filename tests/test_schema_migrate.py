"""Schema migration: recipe v1→v2, plane abcd→normal/d, migrate CLI."""

from __future__ import annotations

import json

import numpy as np

from cloudet.cli import build_migrate_parser, main
from cloudet.core.plane import Plane
from cloudet.fit.picking import PickParams
from cloudet.project import SourceInfo, save_group, save_manifest
from cloudet.project.schema import (
    RECIPE_VERSION,
    migrate_geometry,
    migrate_group_doc,
    migrate_project,
    migrate_recipe,
    plane_from_json,
    plane_to_json,
)
from cloudet.reduction import load_recipe, run_reduction, write_geometry_json, write_recipe_json


def test_plane_json_roundtrip():
    plane = Plane(np.array([0.0, 0.0, 1.0]), -12.5)
    rec = plane_to_json(plane)
    assert "normal" in rec and "d" in rec
    assert "abcd" not in rec
    back = plane_from_json(rec)
    np.testing.assert_allclose(back.as_array(), plane.as_array())


def test_plane_from_legacy_abcd():
    plane = plane_from_json({"abcd": [0.0, 1.0, 0.0, -3.0]})
    np.testing.assert_allclose(plane.as_array(), [0.0, 1.0, 0.0, -3.0])


def test_migrate_recipe_renames_construct_and_measures():
    doc = {
        "version": 1,
        "units": "mm",
        "faces": {"wall": {"from": "group", "name": "G0"}},
        "construct": [
            {"id": "in", "op": "offset", "of": "wall", "distance_mm": 1.0},
            {"id": "axis", "op": "intersect_planes", "a": "in", "b": "wall"},
            {
                "id": "hit",
                "op": "intersect_normal_plane",
                "src": "wall",
                "dst": "in",
            },
        ],
        "measures": [
            {"id": "d", "op": "distance_points", "a": "p0", "b": "p1"},
            {"id": "ap", "op": "angle_planes", "a": "wall", "b": "in"},
        ],
    }
    out = migrate_recipe(doc)
    assert out["version"] == RECIPE_VERSION
    assert out["construct"][0]["plane"] == "wall"
    assert "of" not in out["construct"][0]
    assert out["construct"][1]["plane_a"] == "in"
    assert out["construct"][1]["plane_b"] == "wall"
    assert "a" not in out["construct"][1]
    assert out["construct"][2]["source_plane"] == "wall"
    assert out["construct"][2]["destination_plane"] == "in"
    assert out["measures"][0]["point_a"] == "p0"
    assert out["measures"][1]["plane_a"] == "wall"


def test_migrate_plane_record_and_geometry():
    geo = {
        "planes": {
            "wall": {
                "kind": "plane",
                "abcd": [0.0, 0.0, 1.0, 0.0],
                "of": ["src"],
                "provenance": "offset",
            }
        },
        "lines": {},
        "points": {},
        "recipe": {
            "echo": {
                "version": 1,
                "construct": [
                    {"id": "in", "op": "offset", "of": "wall", "distance_mm": 1.0}
                ],
            }
        },
    }
    out = migrate_geometry(geo)
    wall = out["planes"]["wall"]
    assert wall["normal"] == [0.0, 0.0, 1.0]
    assert wall["d"] == 0.0
    assert "abcd" not in wall
    assert wall["parents"] == ["src"]
    assert "of" not in wall
    assert out["recipe"]["echo"]["construct"][0]["plane"] == "wall"


def test_migrate_group_doc():
    doc = {
        "fit": {
            "planes": [
                {
                    "plane_index": 0,
                    "abcd": [1.0, 0.0, 0.0, -2.0],
                    "status": "ok",
                }
            ]
        }
    }
    out = migrate_group_doc(doc)
    p = out["fit"]["planes"][0]
    assert p["normal"] == [1.0, 0.0, 0.0]
    assert p["d"] == -2.0
    assert "abcd" not in p


def test_v1_recipe_loads_and_runs(tmp_path):
    params = PickParams()
    left = Plane(np.array([1.0, 0.0, 0.0]), 50.0)
    front = Plane(np.array([0.0, 1.0, 0.0]), 20.0)
    target = Plane(np.array([0.0, 0.0, 1.0]), -100.0)
    for i, (name, plane) in enumerate(
        [("tracker_left", left), ("tracker_front", front), ("target", target)]
    ):
        save_group(
            tmp_path,
            i,
            name,
            np.zeros((3, 3)),
            None,
            plane.as_array(),
            None,
            detection=params,
            fit_summary={
                "planes": [
                    {
                        "plane_index": 0,
                        "abcd": plane.as_array().tolist(),
                        "n_points": 3,
                        "status": "ok",
                        "reasons": [],
                        "bimodal": False,
                        "mad_sigma_mm": 0.04,
                        "threshold_mm": 0.15,
                    }
                ]
            },
        )
    save_manifest(tmp_path, SourceInfo(path="s.ply", n_points=3), params, n_groups=3)

    recipe_v1 = {
        "version": 1,
        "units": "mm",
        "faces": {
            "tracker_left": {"from": "group", "name": "tracker_left"},
            "tracker_front": {"from": "group", "name": "tracker_front"},
            "target": {"from": "group", "name": "target"},
        },
        "construct": [
            {
                "id": "left_in",
                "op": "offset",
                "of": "tracker_left",
                "distance_mm": 50.0,
            },
            {
                "id": "front_in",
                "op": "offset",
                "of": "tracker_front",
                "distance_mm": 20.0,
            },
            {
                "id": "beam_axis",
                "op": "intersect_planes",
                "a": "left_in",
                "b": "front_in",
            },
            {
                "id": "beam_on_target",
                "op": "intersect_line_plane",
                "line": "beam_axis",
                "plane": "target",
            },
        ],
        "export": ["beam_axis", "beam_on_target"],
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe_v1))
    loaded = load_recipe(recipe_path)
    assert loaded["version"] == RECIPE_VERSION
    assert loaded["construct"][0]["plane"] == "tracker_left"
    assert "of" not in loaded["construct"][0]

    result = run_reduction(tmp_path, loaded)
    assert "beam_axis" in result.lines
    assert "beam_on_target" in result.points

    out = tmp_path / "geometry.json"
    write_geometry_json(out, result)
    doc = json.loads(out.read_text())
    wall = doc["planes"]["tracker_left"]
    assert "normal" in wall and "d" in wall
    assert "abcd" not in wall
    assert doc["recipe"]["echo"]["version"] == RECIPE_VERSION
    assert doc["recipe"]["echo"]["construct"][0]["plane"] == "tracker_left"

    saved = tmp_path / "saved_recipe.json"
    write_recipe_json(saved, loaded)
    on_disk = json.loads(saved.read_text())
    assert on_disk["version"] == RECIPE_VERSION
    assert on_disk["construct"][2]["plane_a"] == "left_in"


def test_migrate_project_and_cli(tmp_path):
    groups = tmp_path / "groups"
    groups.mkdir()
    group_doc = {
        "fit": {
            "planes": [
                {"plane_index": 0, "abcd": [0.0, 0.0, 1.0, -1.0], "status": "ok"}
            ]
        }
    }
    group_path = groups / "group_000.json"
    group_path.write_text(json.dumps(group_doc))

    recipe = {
        "version": 1,
        "faces": {},
        "construct": [
            {"id": "in", "op": "offset", "of": "wall", "distance_mm": 1.0}
        ],
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))

    geometry = {
        "planes": {
            "wall": {"kind": "plane", "abcd": [1.0, 0.0, 0.0, 0.0], "of": ["x"]}
        },
        "lines": {},
        "points": {},
    }
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(geometry))

    dry = migrate_project(tmp_path, dry_run=True)
    assert len(dry) == 3
    assert json.loads(group_path.read_text())["fit"]["planes"][0]["abcd"]

    assert main(["migrate", str(tmp_path), "--dry-run"]) == 0
    assert "abcd" in json.loads(group_path.read_text())["fit"]["planes"][0]

    assert main(["migrate", str(tmp_path)]) == 0
    g = json.loads(group_path.read_text())
    assert g["fit"]["planes"][0]["normal"] == [0.0, 0.0, 1.0]
    assert "abcd" not in g["fit"]["planes"][0]
    r = json.loads(recipe_path.read_text())
    assert r["version"] == RECIPE_VERSION
    assert r["construct"][0]["plane"] == "wall"
    geo = json.loads(geometry_path.read_text())
    assert geo["planes"]["wall"]["parents"] == ["x"]
    assert "abcd" not in geo["planes"]["wall"]

    # Second pass is a no-op.
    assert migrate_project(tmp_path) == []


def test_migrate_parser():
    p = build_migrate_parser()
    args = p.parse_args(["proj", "--recipe", "r.json", "--geometry", "g.json", "--dry-run"])
    assert args.project_dir == "proj"
    assert args.recipe == "r.json"
    assert args.geometry == "g.json"
    assert args.dry_run is True
