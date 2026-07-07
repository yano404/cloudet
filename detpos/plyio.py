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


def read_ply_xyz(path: str | Path) -> np.ndarray:
    """Read vertex x, y, z from a PLY file as float64, shape (N, 3)."""
    path = Path(path)
    with open(path, "rb") as f:
        fmt, n_vertex, props, offset = _parse_header(f)

        if fmt == "ascii":
            names = [p[0] for p in props]
            cols = [names.index(c) for c in ("x", "y", "z")]
            data = np.loadtxt(f, dtype=np.float64, max_rows=n_vertex, ndmin=2)
            if data.shape != (n_vertex, len(props)):
                raise ValueError(
                    f"ascii PLY shape mismatch: got {data.shape}, "
                    f"expected ({n_vertex}, {len(props)})"
                )
            return np.ascontiguousarray(data[:, cols])

        # binary_little_endian
        dtype = np.dtype([(name, "<" + t) for name, t in props])
        data = np.fromfile(f, dtype=dtype, count=n_vertex)
        if len(data) != n_vertex:
            raise ValueError(
                f"binary PLY truncated: got {len(data)} of {n_vertex} vertices"
            )
        out = np.empty((n_vertex, 3), dtype=np.float64)
        for j, c in enumerate(("x", "y", "z")):
            out[:, j] = data[c]
        return out


def write_ply_xyz(path: str | Path, points: np.ndarray, binary: bool = True) -> None:
    """Write an (N, 3) array as a double-precision PLY point cloud."""
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    fmt = "binary_little_endian" if binary else "ascii"
    header = (
        "ply\n"
        f"format {fmt} 1.0\n"
        "comment Created by detpos\n"
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
