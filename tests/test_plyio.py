import numpy as np
import pytest

from cloudet.plyio import read_ply_xyz, write_ply_xyz


def test_roundtrip_binary(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1000, 1000, size=(1234, 3))
    path = tmp_path / "pts.ply"
    write_ply_xyz(path, pts, binary=True)
    back = read_ply_xyz(path)
    assert back.shape == pts.shape
    assert np.array_equal(back, pts)  # double precision, bit-exact


def test_roundtrip_ascii(tmp_path):
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1000, 1000, size=(57, 3))
    path = tmp_path / "pts_ascii.ply"
    write_ply_xyz(path, pts, binary=False)
    back = read_ply_xyz(path)
    assert np.allclose(back, pts, atol=1e-9)


def test_extra_properties_skipped(tmp_path):
    """Binary PLY with extra float property between coordinates."""
    n = 10
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property float intensity\n"
        "end_header\n"
    )
    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("i", "<f4")])
    data = np.zeros(n, dtype=dtype)
    data["x"] = np.arange(n)
    data["y"] = 2.0 * np.arange(n)
    data["z"] = -1.5
    path = tmp_path / "extra.ply"
    with open(path, "wb") as f:
        f.write(header.encode())
        data.tofile(f)
    pts = read_ply_xyz(path)
    assert np.array_equal(pts[:, 0], np.arange(n))
    assert np.array_equal(pts[:, 1], 2.0 * np.arange(n))
    assert np.all(pts[:, 2] == -1.5)


def test_truncated_file_raises(tmp_path):
    pts = np.zeros((100, 3))
    path = tmp_path / "trunc.ply"
    write_ply_xyz(path, pts)
    raw = path.read_bytes()
    path.write_bytes(raw[:-100])
    with pytest.raises(ValueError, match="truncated"):
        read_ply_xyz(path)
