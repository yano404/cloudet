"""Text point cloud -> PLY conversion (core + CLI)."""

from __future__ import annotations

import numpy as np
import pytest

from cloudet.cli import main
from cloudet.core.convert import text_xyz_to_ply
from cloudet.core.plyio import read_ply_xyz


def _points(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    # ~2 m extents with sub-micron digits: float32 would visibly lose precision
    return rng.uniform(-2000.0, 2000.0, size=(n, 3)).round(6)


def _write_text(path, pts, *, delim=",", eol="\n", trailing_blank=0, extra_cols=0):
    rows = []
    for r in pts:
        cols = [f"{v:.6f}" for v in r] + ["7"] * extra_cols
        rows.append(delim.join(cols))
    text = eol.join(rows) + eol + eol * trailing_blank
    path.write_bytes(text.encode())


def test_roundtrip_bit_exact(tmp_path):
    pts = _points(5000)
    src = tmp_path / "scan.txt"
    _write_text(src, pts)
    res = text_xyz_to_ply(src, chunk_rows=777)  # several chunks, odd boundary
    assert res.dst == tmp_path / "scan.ply"
    assert res.n_points == len(pts)
    back = read_ply_xyz(res.dst)
    ref = np.loadtxt(src, delimiter=",")
    assert back.dtype == np.float64
    assert np.array_equal(back, ref)
    assert np.allclose(res.bbox_min, pts.min(axis=0))
    assert np.allclose(res.bbox_max, pts.max(axis=0))
    assert not (tmp_path / "scan.ply.part").exists()


@pytest.mark.parametrize(
    "delim,name",
    [(",", "auto"), (";", "auto"), ("\t", "auto"), (" ", "auto"), (" ", "space")],
)
def test_delimiters(tmp_path, delim, name):
    pts = _points(50)
    src = tmp_path / "a.txt"
    _write_text(src, pts, delim=delim)
    res = text_xyz_to_ply(src, delimiter=name)
    assert np.array_equal(read_ply_xyz(res.dst), np.loadtxt(src, delimiter=None if delim == " " else delim))


def test_crlf_blank_lines_comments_and_extra_columns(tmp_path):
    pts = _points(40)
    src = tmp_path / "b.txt"
    _write_text(src, pts, eol="\r\n", trailing_blank=3, extra_cols=2)
    text = src.read_bytes()
    src.write_bytes(b"# exported by test\r\n\r\n" + text)
    res = text_xyz_to_ply(src, chunk_rows=7)
    assert res.n_points == 40
    assert np.allclose(read_ply_xyz(res.dst), pts)


def test_no_trailing_newline(tmp_path):
    src = tmp_path / "c.txt"
    src.write_bytes(b"1.5,2.5,3.5\n4.5,5.5,6.5")
    res = text_xyz_to_ply(src)
    assert res.n_points == 2
    assert np.array_equal(read_ply_xyz(res.dst), [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])


def test_header_parses_with_trailing_space_padding(tmp_path):
    src = tmp_path / "d.txt"
    _write_text(src, _points(3))
    res = text_xyz_to_ply(src)
    head = res.dst.read_bytes()[:200].split(b"end_header\n")[0].decode()
    assert "format binary_little_endian 1.0" in head
    assert "property double x" in head
    n_line = [ln for ln in head.splitlines() if ln.startswith("element vertex")][0]
    assert int(n_line.split()[2]) == 3  # padded with spaces, still parses


def test_existing_destination_requires_overwrite(tmp_path):
    src = tmp_path / "e.txt"
    _write_text(src, _points(5))
    dst = tmp_path / "e.ply"
    dst.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        text_xyz_to_ply(src)
    assert dst.read_bytes() == b"keep me"
    text_xyz_to_ply(src, overwrite=True)
    assert read_ply_xyz(dst).shape == (5, 3)


def test_bad_input_leaves_no_output(tmp_path):
    cases = {
        "empty.txt": b"",
        "comments.txt": b"# nothing\n\n",
        "two_cols.txt": b"1,2\n3,4\n",
        "text.txt": b"1,2,3\nx,y,z\n",
        "nan.txt": b"1,2,3\nnan,2,3\n",
    }
    for name, content in cases.items():
        src = tmp_path / name
        src.write_bytes(content)
        with pytest.raises(ValueError):
            text_xyz_to_ply(src)
        assert not src.with_suffix(".ply").exists(), name
        assert not (tmp_path / (src.stem + ".ply.part")).exists(), name


def test_same_file_rejected(tmp_path):
    src = tmp_path / "f.ply"  # misleading name, but dst == src must still be refused
    _write_text(src, _points(3))
    with pytest.raises(ValueError):
        text_xyz_to_ply(src, src, overwrite=True)


def test_low_disk_space_refused(tmp_path):
    src = tmp_path / "g.txt"
    _write_text(src, _points(10))
    with pytest.raises(OSError, match="not enough free space"):
        text_xyz_to_ply(src, free_margin_bytes=1 << 62)
    assert not (tmp_path / "g.ply").exists()


def test_progress_callback(tmp_path):
    src = tmp_path / "h.txt"
    _write_text(src, _points(25))
    seen = []
    text_xyz_to_ply(src, chunk_rows=10, progress=seen.append)
    assert seen == [10, 20, 25]


# --- CLI ----------------------------------------------------------------

def test_cli_default_output_next_to_input(tmp_path, capsys):
    src = tmp_path / "s.txt"
    _write_text(src, _points(20))
    assert main(["convert", str(src)]) == 0
    assert read_ply_xyz(tmp_path / "s.ply").shape == (20, 3)
    assert "converted" in capsys.readouterr().out


def test_cli_output_dir_and_skip_then_force(tmp_path, capsys):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    _write_text(a, _points(5, 1))
    _write_text(b, _points(6, 2))
    out = tmp_path / "ply"
    assert main(["convert", str(a), str(b), "-o", str(out)]) == 0
    assert read_ply_xyz(out / "a.ply").shape == (5, 3)
    assert read_ply_xyz(out / "b.ply").shape == (6, 3)
    capsys.readouterr()
    assert main(["convert", str(a), "-o", str(out)]) == 0  # skipped
    assert "skip" in capsys.readouterr().out
    _write_text(a, _points(9, 3))
    assert main(["convert", str(a), "-o", str(out), "--force"]) == 0
    assert read_ply_xyz(out / "a.ply").shape == (9, 3)


def test_cli_single_input_explicit_file(tmp_path):
    src = tmp_path / "s.txt"
    _write_text(src, _points(4))
    dst = tmp_path / "custom_name.ply"
    assert main(["convert", str(src), "-o", str(dst)]) == 0
    assert read_ply_xyz(dst).shape == (4, 3)


def test_cli_several_inputs_with_ply_output_is_error(tmp_path, capsys):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    _write_text(a, _points(2))
    _write_text(b, _points(2))
    assert main(["convert", str(a), str(b), "-o", str(tmp_path / "x.ply")]) == 1
    assert "directory" in capsys.readouterr().err


def test_cli_failure_does_not_stop_other_files(tmp_path, capsys):
    bad, good = tmp_path / "bad.txt", tmp_path / "good.txt"
    bad.write_bytes(b"1,2\n")
    _write_text(good, _points(3))
    assert main(["convert", str(bad), str(good)]) == 1
    assert (tmp_path / "good.ply").exists()
    assert not (tmp_path / "bad.ply").exists()
    assert "error:" in capsys.readouterr().err


def test_cli_missing_input(tmp_path, capsys):
    assert main(["convert", str(tmp_path / "nope.txt")]) == 1
    assert "error:" in capsys.readouterr().err
