"""Convert delimited-text point clouds (``x,y,z`` per line) to PLY.

numpy only. The output is double-precision ``binary_little_endian`` with
``property double x/y/z`` (the same layout as :func:`cloudet.core.plyio.write_ply_xyz`
and Open3D-written files), so coordinates survive the conversion bit for bit
(float32 would only resolve ~0.1 mm at ~2 m from the origin).

Input is streamed in chunks, so memory use is independent of file size.
The output is written to ``<dst>.part`` and renamed only after the vertex
count in the header has been finalised, so an interrupted run never leaves a
file that looks valid.
"""

from __future__ import annotations

import itertools
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = ["ConvertResult", "text_xyz_to_ply"]

# Header reserves a fixed-width vertex count so it can be patched in place
# once the number of parsed rows is known (single pass, no pre-count).
_COUNT_WIDTH = 20
_BYTES_PER_POINT = 24
_DELIMITERS = {
    "auto": "auto",
    "comma": ",",
    "semicolon": ";",
    "tab": "\t",
    "space": None,  # numpy: any whitespace
}


@dataclass(frozen=True)
class ConvertResult:
    """Outcome of one conversion."""

    src: Path
    dst: Path
    n_points: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]


def _sniff(path: Path, max_lines: int = 1000) -> tuple[str | None, float]:
    """Return (delimiter or None for whitespace, mean bytes per data line)."""
    sizes: list[int] = []
    delim: str | None = None
    found = False
    with open(path, "rb") as f:
        for raw in itertools.islice(f, max_lines * 4):
            s = raw.strip()
            if not s or s.startswith(b"#"):
                continue
            if not found:
                for cand, d in ((b",", ","), (b";", ";"), (b"\t", "\t")):
                    if cand in s:
                        delim = d
                        break
                found = True
            sizes.append(len(raw))
            if len(sizes) >= max_lines:
                break
    if not found:
        raise ValueError(f"no data lines found in {path}")
    return delim, float(np.mean(sizes))


def text_xyz_to_ply(
    src: str | Path,
    dst: str | Path | None = None,
    *,
    delimiter: str = "auto",
    chunk_rows: int = 4_000_000,
    overwrite: bool = False,
    free_margin_bytes: int = 1 << 30,
    progress: Callable[[int], None] | None = None,
) -> ConvertResult:
    """Convert a text point cloud to a double-precision binary PLY.

    Parameters
    ----------
    src:
        Text file with at least three numeric columns; the first three are
        read as x, y, z. Blank lines and ``#`` comments are skipped; CRLF and
        LF line endings both work.
    dst:
        Output path (default: ``src`` with a ``.ply`` suffix).
    delimiter:
        ``"auto"`` (sniff from the first data line), ``"comma"``,
        ``"semicolon"``, ``"tab"`` or ``"space"`` (any whitespace).
    chunk_rows:
        Rows parsed per chunk (memory ~ ``chunk_rows * 24`` bytes plus parsing
        overhead).
    overwrite:
        Replace an existing ``dst``; otherwise ``FileExistsError`` is raised.
    free_margin_bytes:
        Refuse to start unless the destination volume has the estimated output
        size plus this margin free.
    progress:
        Optional callback receiving the running number of points written.

    Raises
    ------
    ValueError
        No points, fewer than three columns, or a non-finite coordinate.
    OSError
        Not enough free disk space.
    """
    src = Path(src)
    dst = src.with_suffix(".ply") if dst is None else Path(dst)
    if delimiter not in _DELIMITERS:
        raise ValueError(f"delimiter must be one of {sorted(_DELIMITERS)}")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    if dst.exists() and not overwrite:
        raise FileExistsError(str(dst))
    if src.resolve() == dst.resolve():
        raise ValueError("source and destination are the same file")

    sniffed, bytes_per_line = _sniff(src)
    delim = sniffed if _DELIMITERS[delimiter] == "auto" else _DELIMITERS[delimiter]

    dst.parent.mkdir(parents=True, exist_ok=True)
    est_points = int(src.stat().st_size / bytes_per_line) + 1
    need = est_points * _BYTES_PER_POINT + free_margin_bytes
    free = shutil.disk_usage(dst.parent).free
    if free < need:
        raise OSError(
            f"not enough free space on {dst.parent}: about {need / 1e9:.2f} GB "
            f"needed (estimated {est_points} points), {free / 1e9:.2f} GB free"
        )

    tmp = dst.with_name(dst.name + ".part")
    n = 0
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    try:
        with open(src, "r", encoding="utf-8-sig", newline=None) as fin, open(
            tmp, "wb"
        ) as fout:
            fout.write(
                (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    "comment Created by cloudet convert\n"
                    "element vertex "
                ).encode("ascii")
            )
            count_pos = fout.tell()
            fout.write(
                (" " * _COUNT_WIDTH + "\n"
                 "property double x\n"
                 "property double y\n"
                 "property double z\n"
                 "end_header\n").encode("ascii")
            )
            while True:
                lines = list(itertools.islice(fin, chunk_rows))
                if not lines:
                    break
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)  # all-blank chunk
                    try:
                        a = np.loadtxt(
                            lines,
                            dtype=np.float64,
                            delimiter=delim,
                            comments="#",
                            usecols=(0, 1, 2),
                            ndmin=2,
                        )
                    except IndexError as e:
                        raise ValueError(
                            f"{src}: expected at least 3 columns "
                            f"(delimiter {delimiter!r}; around row {n + 1})"
                        ) from e
                    except ValueError as e:
                        raise ValueError(
                            f"{src}: could not parse numbers around row {n + 1}: {e}"
                        ) from e
                if a.size == 0:
                    continue
                if not np.isfinite(a).all():
                    bad = int(np.argmax(~np.isfinite(a).all(axis=1)))
                    raise ValueError(
                        f"{src}: non-finite coordinate near data row {n + bad + 1}"
                    )
                a = np.ascontiguousarray(a, dtype="<f8")
                a.tofile(fout)
                lo = np.minimum(lo, a.min(axis=0))
                hi = np.maximum(hi, a.max(axis=0))
                n += len(a)
                if progress is not None:
                    progress(n)
            if n == 0:
                raise ValueError(f"{src}: no points parsed")
            fout.seek(count_pos)
            fout.write(f"{n:<{_COUNT_WIDTH}d}".encode("ascii"))
        os.replace(tmp, dst)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return ConvertResult(
        src=src,
        dst=dst,
        n_points=n,
        bbox_min=tuple(float(v) for v in lo),
        bbox_max=tuple(float(v) for v in hi),
    )
