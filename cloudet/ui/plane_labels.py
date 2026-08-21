"""Fitted-entity display-name helpers (planes, cylinders, circles)."""

from __future__ import annotations

import re

__all__ = [
    "plane_label",
    "plane_id_token",
    "cylinder_label",
    "cylinder_id_token",
    "circle_label",
    "circle_id_token",
]


def _named_or_default(entry: dict, *, index_key: str, prefix: str) -> str:
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"{prefix}{int(entry.get(index_key, 0))}"


def _id_token(label: str, *, fallback: str) -> str:
    token = re.sub(r"[^\w.-]+", "_", label, flags=re.UNICODE).strip("._")
    return token or fallback


def plane_label(p: dict) -> str:
    """User-facing plane name; defaults to p0, p1, …"""
    return _named_or_default(p, index_key="plane_index", prefix="p")


def plane_id_token(p: dict) -> str:
    """Reduction / file-safe token derived from the display name."""
    return _id_token(plane_label(p), fallback=f"p{int(p.get('plane_index', 0))}")


def cylinder_label(c: dict) -> str:
    """User-facing cylinder name; defaults to cyl0, cyl1, …"""
    return _named_or_default(c, index_key="cylinder_index", prefix="cyl")


def cylinder_id_token(c: dict) -> str:
    """Reduction / file-safe token for a cylinder axis."""
    return _id_token(
        cylinder_label(c), fallback=f"cyl{int(c.get('cylinder_index', 0))}"
    )


def circle_label(c: dict) -> str:
    """User-facing circle name; defaults to cir0, cir1, …"""
    return _named_or_default(c, index_key="circle_index", prefix="cir")


def circle_id_token(c: dict) -> str:
    """Reduction / file-safe token for a circle center."""
    return _id_token(circle_label(c), fallback=f"cir{int(c.get('circle_index', 0))}")
