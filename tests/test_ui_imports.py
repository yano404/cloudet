"""Regression: Qt symbols used in cloudet.ui must be imported."""

from __future__ import annotations

import ast
import re
from pathlib import Path


def _missing_qt_names(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "Qt" in node.module:
            for alias in node.names:
                imported.add(alias.asname or alias.name)
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and re.match(r"^Q[A-Z]", node.id):
            used.add(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if re.match(r"^Q[A-Z]", node.value.id):
                used.add(node.value.id)
    return sorted(used - imported)


def test_ui_modules_import_all_qt_symbols():
    ui_dir = Path(__file__).resolve().parents[1] / "cloudet" / "ui"
    problems: list[str] = []
    for path in sorted(ui_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        missing = _missing_qt_names(path)
        if missing:
            problems.append(f"{path.name}: {missing}")
    assert not problems, "Missing Qt imports:\n" + "\n".join(problems)
