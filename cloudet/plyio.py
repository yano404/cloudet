"""Minimal PLY point reader/writer (numpy only, no Open3D dependency).

Reads x, y, z from ascii or binary_little_endian PLY files, preserving
double precision. Extra per-vertex properties are skipped. Open3D-written
group files (binary, double x/y/z) are the primary target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_ply_xyz", "write_ply_xyz"]

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}

# Prefer mmap over loading full vertex records when N is large.
_MMAP_MIN_VERTICES = 50_000


def _parse_header(f) -> tuple[str, int, list[tuple[str, str]], int]:
    """Return (format, n_vertex, [(name, dtype), ...], header_end_offset)."""
    magic = f.readline().strip()
    if magic != b"ply":
        raise ValueError("not a PLY file")

    fmt = None
    n_vertex = None
    props: list[tuple[str, str]] = []
    in_vertex = False

    while True:
        line = f.readline()
        if not line:
            raise ValueError("unexpected end of PLY header")
        tokens = line.decode("ascii", errors="replace").strip().split()
        if not tokens:
            continue
        kw = tokens[0]
        if kw == "format":
            fmt = tokens[1]
        elif kw == "comment":
            continue
        elif kw == "element":
            in_vertex = tokens[1] == "vertex"
            if in_vertex:
                n_vertex = int(tokens[2])
            elif n_vertex is not None:
                # elements after vertex don't affect vertex reading
                in_vertex = False
        elif kw == "property" and in_vertex:
            if tokens[1] == "list":
                raise ValueError("list properties on vertex element are not supported")
            props.append((tokens[-1], _PLY_TYPES[tokens[1]]))
        elif kw == "end_header":
            break

    if fmt is None or n_vertex is None:
        raise ValueError("invalid PLY header (missing format or vertex element)")
    if fmt == "binary_big_endian":
        raise ValueError("big-endian PLY is not supported")
    for name in ("x", "y", "z"):
        if name not in [p[0] for p in props]:
            raise ValueError(f"vertex property '{name}' not found")
    return fmt, n_vertex, props, f.tell()


def _vertex_dtype(props: list[tuple[str, str]]) -> np.dtype:
    return np.dtype([(name, "<" + t) for name, t in props])


def _check_binary_payload(path: Path, offset: int, n_vertex: int, itemsize: int) -> None:
    need = offset + n_vertex * itemsize
    if path.stat().st_size < need:
        raise ValueError(
            f"binary PLY truncated: file has {path.stat().st_size} bytes, "
            f"need {need} for {n_vertex} vertices"
        )


def _read_binary_xyz_only(
    f, n_vertex: int, *, itemsize: int, count: int, out_dtype: np.dtype
) -> np.ndarray:
    buf = np.fromfile(f, dtype=out_dtype, count=count)
    if buf.size != count:
        raise ValueError(
            f"binary PLY truncated: got {buf.size} values, expected {count}"
        )
    out = buf.reshape(n_vertex, 3)
    if out_dtype != np.float64:
        return out.astype(np.float64, copy=False)
    return out


def _read_binary_xyz(
    path: Path, offset: int, n_vertex: int, props: list[tuple[str, str]]
) -> np.ndarray:
    """Load x/y/z from a binary_little_endian vertex block."""
    dtype = _vertex_dtype(props)

    # Fast path: contiguous xyz-only clouds (common for cloudet groups / Open3D).
    if props == [("x", "f8"), ("y", "f8"), ("z", "f8")]:
        _check_binary_payload(path, offset, n_vertex, 24)
        with open(path, "rb") as f:
            f.seek(offset)
            return _read_binary_xyz_only(
                f, n_vertex, itemsize=24, count=n_vertex * 3, out_dtype=np.float64
            )

    if props == [("x", "f4"), ("y", "f4"), ("z", "f4")]:
        _check_binary_payload(path, offset, n_vertex, 12)
        with open(path, "rb") as f:
            f.seek(offset)
            return _read_binary_xyz_only(
                f, n_vertex, itemsize=12, count=n_vertex * 3, out_dtype=np.float32
            )

    # Fast path: double x/y/z first, extra vertex properties after (FARO-style).
    if (
        len(props) >= 3
        and props[0] == ("x", "f8")
        and props[1] == ("y", "f8")
        and props[2] == ("z", "f8")
    ):
        vertex_size = dtype.itemsize
        _check_binary_payload(path, offset, n_vertex, vertex_size)
        if n_vertex >= _MMAP_MIN_VERTICES:
            mm = np.memmap(
                path, dtype=dtype, mode="r", offset=offset, shape=(n_vertex,)
            )
            out = np.empty((n_vertex, 3), dtype=np.float64)
            out[:, 0] = mm["x"]
            out[:, 1] = mm["y"]
            out[:, 2] = mm["z"]
            return out
        with open(path, "rb") as f:
            f.seek(offset)
            raw = np.fromfile(f, dtype=np.uint8, count=n_vertex * vertex_size)
        if raw.size != n_vertex * vertex_size:
            raise ValueError(
                f"binary PLY truncated: got {raw.size} bytes, "
                f"expected {n_vertex * vertex_size}"
            )
        return (
            raw.reshape(n_vertex, vertex_size)[:, :24]
            .reshape(n_vertex, 3, 8)
            .view(np.float64)
            .reshape(n_vertex, 3)
            .copy()
        )

    _check_binary_payload(path, offset, n_vertex, dtype.itemsize)
    if n_vertex >= _MMAP_MIN_VERTICES:
        mm = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(n_vertex,))
        out = np.empty((n_vertex, 3), dtype=np.float64)
        out[:, 0] = mm["x"]
        out[:, 1] = mm["y"]
        out[:, 2] = mm["z"]
        return out

    with open(path, "rb") as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype, count=n_vertex)
    if len(data) != n_vertex:
        raise ValueError(
            f"binary PLY truncated: got {len(data)} of {n_vertex} vertices"
        )
    out = np.empty((n_vertex, 3), dtype=np.float64)
    out[:, 0] = data["x"]
    out[:, 1] = data["y"]
    out[:, 2] = data["z"]
    return out


def read_ply_xyz(path: str | Path) -> np.ndarray:
    """Read vertex x, y, z from a PLY file as float64, shape (N, 3)."""
    path = Path(path)
    with open(path, "rb") as f:
        fmt, n_vertex, props, offset = _parse_header(f)

        if fmt == "ascii":
            names = [p[0] for p in props]
            cols = [names.index(c) for c in ("x", "y", "z")]
            data = np.loadtxt(
                f, dtype=np.float64, max_rows=n_vertex, ndmin=2, usecols=cols
            )
            if data.shape != (n_vertex, 3):
                raise ValueError(
                    f"ascii PLY shape mismatch: got {data.shape}, "
                    f"expected ({n_vertex}, 3)"
                )
            return np.ascontiguousarray(data)

        return _read_binary_xyz(path, offset, n_vertex, props)


def write_ply_xyz(path: str | Path, points: np.ndarray, binary: bool = True) -> None:
    """Write an (N, 3) array as a double-precision PLY point cloud."""
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    fmt = "binary_little_endian" if binary else "ascii"
    header = (
        "ply\n"
        f"format {fmt} 1.0\n"
        "comment Created by cloudet\n"
        f"element vertex {len(points)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if binary:
            points.astype("<f8").tofile(f)
        else:
            np.savetxt(f, points, fmt="%.12g")
