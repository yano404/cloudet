"""End-to-end pipeline test on a synthetic legacy-format group directory."""

import json

import numpy as np
import pytest

from cloudet.cli import main as cli_main
from cloudet.groups import load_groups
from cloudet.pipeline import FitParams, fit_groupset, residual_uv_map
from cloudet.plane import Plane
from cloudet.plyio import write_ply_xyz

SIGMA = 0.03


@pytest.fixture
def legacy_dir(tmp_path):
    """Two synthetic groups in the legacy picker layout."""
    rng = np.random.default_rng(0)
    summary = []
    true_planes = {}
    for gid, (normal, offset) in enumerate([((0, 0, 1), 100.0), ((1, 0, 0), -50.0)]):
        n = np.asarray(normal, dtype=np.float64)
        a = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(n, a); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        uv = rng.uniform(-50, 50, size=(20_000, 2))
        noise = rng.normal(0, SIGMA, size=len(uv))
        pts = -offset * n + uv[:, :1] * u + uv[:, 1:] * v + noise[:, None] * n
        # contamination: second surface 0.5 mm off
        ghost = pts[:2000] + 0.5 * n
        pts = np.vstack([pts, ghost])

        write_ply_xyz(tmp_path / f"group_{gid:03d}.ply", pts)
        summary.append({
            "group_id": gid,
            "name": f"G{gid}",
            "num_points": len(pts),
            "clicked": (-offset * n).tolist(),
            "coarse_plane_model": [*n, offset],
            "ply_file": f"group_{gid:03d}.ply",
        })
        true_planes[gid] = Plane(n, offset)

    with open(tmp_path / "groups_summary.json", "w") as f:
        json.dump(summary, f)
    return tmp_path, true_planes


def test_load_groups_legacy(legacy_dir):
    d, _ = legacy_dir
    groups = load_groups(d)
    assert [g.group_id for g in groups] == [0, 1]
    pts = groups[0].load_points()
    assert pts.shape == (22_000, 3)


def test_fit_groupset(legacy_dir, tmp_path):
    d, true_planes = legacy_dir
    out = tmp_path / "fits"
    records = fit_groupset(d, out, FitParams(seed=1), uv_maps=False, log=lambda *_: None)
    assert len(records) == 2

    for rec in records:
        gid = rec["group"]["id"]
        plane = Plane.from_array(rec["plane"]["abcd"])
        true = true_planes[gid]
        assert plane.angle_to(true) < 1e-4
        assert abs(plane.d - true.d) < 3e-3  # ghost surface rejected
        assert rec["quality"]["converged"]
        assert rec["quality"]["stats_inliers"]["mad_sigma"] == pytest.approx(SIGMA, rel=0.15)
        assert len(rec["group"]["ply_sha256"]) == 64

    assert (out / "fit_000.json").exists()
    assert (out / "fits_summary.csv").exists()
    lines = (out / "fits_summary.csv").read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 groups


def test_fit_reproducible(legacy_dir, tmp_path):
    d, _ = legacy_dir
    r1 = fit_groupset(d, tmp_path / "a", FitParams(seed=7), uv_maps=False, log=lambda *_: None)
    r2 = fit_groupset(d, tmp_path / "b", FitParams(seed=7), uv_maps=False, log=lambda *_: None)
    for a, b in zip(r1, r2):
        assert a["plane"]["abcd"] == b["plane"]["abcd"]
        assert a["quality"]["n_inliers"] == b["quality"]["n_inliers"]


def test_residual_uv_map(legacy_dir):
    d, true_planes = legacy_dir
    g = load_groups(d)[0]
    pts = g.load_points()[:20_000]  # clean part only
    uv = residual_uv_map(pts, true_planes[0], bins=50)
    assert uv["mean"].shape == (50, 50)
    filled = uv["counts"] > 0
    # mean residual per bin should be small everywhere on the clean plane
    assert np.nanmax(np.abs(uv["mean"][filled])) < 5 * SIGMA


def test_residual_uv_map_aligns_rotated_rectangle():
    """A 45°-rotated rectangle should become axis-aligned in u–v."""
    rng = np.random.default_rng(0)
    # Axis-aligned 80 x 40 mm patch in xy, then rotate 45° about z.
    xs = rng.uniform(-40, 40, size=8000)
    ys = rng.uniform(-20, 20, size=8000)
    pts0 = np.column_stack([xs, ys, rng.normal(0.0, 0.01, size=8000)])
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = pts0 @ R.T
    plane = Plane.from_array([0.0, 0.0, 1.0, 0.0])
    uv = residual_uv_map(pts, plane, bins=40, return_points=True)
    u_span = float(uv["u"].max() - uv["u"].min())
    v_span = float(uv["v"].max() - uv["v"].min())
    long_s, short_s = max(u_span, v_span), min(u_span, v_span)
    assert long_s > 70.0
    assert short_s < 50.0
    assert long_s / short_s > 1.5


def test_residual_uv_map_aligns_despite_density_bias():
    """Uneven sampling should not leave a rectangular face tilted."""
    rng = np.random.default_rng(1)
    # Dense cluster on one side biases PCA away from the true edges.
    n_edge, n_bulk = 6000, 2000
    xs = np.concatenate([
        rng.uniform(-40, -25, size=n_edge),
        rng.uniform(-40, 40, size=n_bulk),
    ])
    ys = np.concatenate([
        rng.uniform(-20, 20, size=n_edge),
        rng.uniform(-20, 20, size=n_bulk),
    ])
    pts0 = np.column_stack([xs, ys, rng.normal(0.0, 0.01, size=n_edge + n_bulk)])
    ang = np.deg2rad(33.0)
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = pts0 @ R.T
    plane = Plane.from_array([0.0, 0.0, 1.0, 0.0])
    uv = residual_uv_map(pts, plane, return_points=True)
    # Corners of the true rectangle in the aligned frame should sit near
    # the AABB corners (small empty corners ⇒ low tilt).
    u_span = float(uv["u"].max() - uv["u"].min())
    v_span = float(uv["v"].max() - uv["v"].min())
    long_s, short_s = max(u_span, v_span), min(u_span, v_span)
    assert long_s > 70.0
    assert short_s < 50.0
    # Occupancy of the AABB should be high if edges are axis-aligned.
    uu, vv = uv["u"], uv["v"]
    nu = ((uu - uu.min()) / max(u_span, 1e-9) * 19).astype(int).clip(0, 19)
    nv = ((vv - vv.min()) / max(v_span, 1e-9) * 19).astype(int).clip(0, 19)
    occ = np.zeros((20, 20), dtype=bool)
    occ[nu, nv] = True
    assert occ.mean() > 0.55


def test_cli_fit(legacy_dir, tmp_path, capsys):
    d, _ = legacy_dir
    out = tmp_path / "cli_out"
    rc = cli_main(["fit", str(d), "-o", str(out), "--no-uv-maps", "--seed", "3"])
    assert rc == 0
    assert (out / "fits_summary.csv").exists()
    assert "G0" in capsys.readouterr().out


def test_cli_missing_dir(tmp_path):
    rc = cli_main(["fit", str(tmp_path / "nope"), "-o", str(tmp_path / "o")])
    assert rc == 1
