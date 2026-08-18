"""Regression: symbols used in cloudet.ui must be imported."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_STDLIB_MODULES = frozenset({"os", "sys", "time", "traceback", "re", "json", "copy"})


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                imported.add(name)
                if module == "datetime":
                    imported.add("datetime")
    return imported


def _missing_qt_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = _imported_names(path)
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and re.match(r"^Q[A-Z]", node.id):
            used.add(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if re.match(r"^Q[A-Z]", node.value.id):
                used.add(node.value.id)
    return sorted(used - imported)


def _missing_stdlib_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = _imported_names(path)
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in _STDLIB_MODULES:
                used.add(node.value.id)
        if isinstance(node, ast.Name) and node.id == "datetime":
            used.add("datetime")
    return sorted(used - imported)


def test_ui_modules_import_all_qt_symbols():
    ui_dir = Path(__file__).resolve().parents[1] / "cloudet" / "ui"
    problems: list[str] = []
    for path in sorted(ui_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        missing = _missing_qt_names(path)
        if missing:
            problems.append(f"{path.name} Qt: {missing}")
        missing_std = _missing_stdlib_modules(path)
        if missing_std:
            problems.append(f"{path.name} stdlib: {missing_std}")
    assert not problems, "Missing imports in cloudet.ui:\n" + "\n".join(problems)
