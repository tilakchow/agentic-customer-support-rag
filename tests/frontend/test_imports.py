"""Architectural boundary tests for frontend package."""

from __future__ import annotations

import ast
from pathlib import Path


def test_frontend_does_not_import_agenticrag() -> None:
    """Enforces that frontend modules NEVER import from agenticrag internals."""
    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    assert frontend_dir.is_dir(), f"frontend dir not found at {frontend_dir}"

    py_files = list(frontend_dir.glob("*.py"))
    assert len(py_files) > 0, "No python files found in frontend"

    violations: list[str] = []
    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("agenticrag"):
                        violations.append(f"{py_file.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("agenticrag"):
                    violations.append(f"{py_file.name}: from {node.module} import ...")

    assert not violations, f"Forbidden backend imports detected in frontend: {violations}"
