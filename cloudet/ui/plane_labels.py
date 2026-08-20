"""Plane display-name helpers."""

from __future__ import annotations

import re

__all__ = ["plane_label", "plane_id_token"]


def plane_label(p: dict) -> str:
    """User-facing plane name; defaults to p0, p1, …"""
    name = p.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"p{int(p.get('plane_index', 0))}"


def plane_id_token(p: dict) -> str:
    """Reduction / file-safe token derived from the display name."""
    token = re.sub(r"[^\w.-]+", "_", plane_label(p), flags=re.UNICODE).strip("._")
    return token or f"p{int(p.get('plane_index', 0))}"


# Backward-compatible aliases (prefer the public names above).
_plane_label = plane_label
_plane_id_token = plane_id_token
