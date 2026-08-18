"""Regression: symbols used in cloudet.ui must be imported."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_STDLIB_MODULES = frozenset({"os", "sys", "time", "traceback", "re", "json", "copy"})
_LOCAL_OK = frozenset(
    {
        "dict",
        "list",
        "set",
        "tuple",
        "float",
        "int",
        "str",
        "bool",
        "len",
        "max",
        "min",
        "range",
        "sorted",
        "isinstance",
        "getattr",
        "hasattr",
        "print",
        "Path",
        "Exception",
        "ValueError",
        "KeyError",
        "TypeError",
        "any",
        "all",
        "enumerate",
        "zip",
        "abs",
        "round",
        "super",
        "fn",  # _guard(fn) parameter name in lambdas
        "open",
        "sum",
        "next",
        "type",
        "FileNotFoundError",
        "_QT_MSG_PREV",
        "_QT_MSG_FILTER_INSTALLED",
        "_VTK_LOG_KEEPALIVE",
    }
)


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    defined: set[str] = set()
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
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
    imported |= defined
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


def _missing_call_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = _imported_names(path)
    missing: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in imported and name not in _LOCAL_OK:
                missing.add(name)
    return sorted(missing)


def test_ui_modules_import_all_symbols():
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
        missing_calls = _missing_call_names(path)
        if missing_calls:
            problems.append(f"{path.name} calls: {missing_calls}")
    assert not problems, "Missing imports in cloudet.ui:\n" + "\n".join(problems)
