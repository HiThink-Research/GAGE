from __future__ import annotations

import ast
from pathlib import Path


SUMMARY_GENERATOR_DIR = Path(__file__).resolve().parents[4] / "src/gage_eval/reporting/summary_generators"


def test_builtin_summary_generators_use_semantic_local_section_ids() -> None:
    repeated: list[str] = []
    for path in SUMMARY_GENERATOR_DIR.glob("*.py"):
        if path.name in {"__init__.py", "base.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        generator_ids = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.endswith("_summary")
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "section":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            section_id = node.args[0].value
            if isinstance(section_id, str) and section_id in generator_ids:
                repeated.append(f"{path.name}:{section_id}")

    assert repeated == []
