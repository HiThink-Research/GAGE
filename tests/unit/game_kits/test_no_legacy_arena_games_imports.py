from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCAN_ROOTS = (_REPO_ROOT / "src", _REPO_ROOT / "tests")
_LEGACY_ARENA_GAMES_ROOT = ".".join(("gage_eval", "role", "arena", "games"))
_BOARD_LEGACY_PARSER_MODULE = ".".join(
    ("gage_eval", "role", "arena", "parsers", "gomoku_parser")
)
_ARENA_MANIFEST_PATH = _REPO_ROOT / "src/gage_eval/registry/manifests/arena.json"


def _is_legacy_arena_game_import(target: str) -> bool:
    normalized = str(target or "").strip()
    return (
        normalized == _LEGACY_ARENA_GAMES_ROOT
        or normalized.startswith(f"{_LEGACY_ARENA_GAMES_ROOT}.")
    )


def _is_legacy_board_parser_import(target: str) -> bool:
    normalized = str(target or "").strip()
    return (
        normalized == _BOARD_LEGACY_PARSER_MODULE
        or normalized.startswith(f"{_BOARD_LEGACY_PARSER_MODULE}.")
    )


def _iter_legacy_arena_game_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_legacy_arena_game_import(alias.name):
                    matches.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if _is_legacy_arena_game_import(node.module or ""):
                matches.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} from {node.module}")
        elif isinstance(node, ast.Call):
            if not node.args:
                continue
            func = node.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name not in {"import_module", "__import__"}:
                continue
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                if _is_legacy_arena_game_import(first_arg.value):
                    matches.append(
                        f"{path.relative_to(_REPO_ROOT)}:{node.lineno} dynamic {first_arg.value}"
                    )

    return matches


def _iter_legacy_board_parser_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_legacy_board_parser_import(alias.name):
                    matches.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if _is_legacy_board_parser_import(node.module or ""):
                matches.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} from {node.module}")
        elif isinstance(node, ast.Call):
            if not node.args:
                continue
            func = node.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name not in {"import_module", "__import__"}:
                continue
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                if _is_legacy_board_parser_import(first_arg.value):
                    matches.append(
                        f"{path.relative_to(_REPO_ROOT)}:{node.lineno} dynamic {first_arg.value}"
                    )

    return matches


@pytest.mark.fast
def test_legacy_arena_game_imports_are_absent() -> None:
    matches: list[str] = []
    for root in _SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            matches.extend(_iter_legacy_arena_game_imports(path))

    assert not matches, "Legacy arena-game imports remain:\n" + "\n".join(matches)


@pytest.mark.fast
def test_board_game_parser_imports_are_absent_from_arena_parsers() -> None:
    matches: list[str] = []
    for root in _SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            matches.extend(_iter_legacy_board_parser_imports(path))

    assert not matches, "Legacy board parser imports remain:\n" + "\n".join(matches)


@pytest.mark.fast
def test_board_game_parser_manifest_entries_are_owned_by_game_kits() -> None:
    manifest = yaml.safe_load(_ARENA_MANIFEST_PATH.read_text(encoding="utf-8"))
    entries_by_kind = {
        kind: {
            entry["name"]: entry
            for entry in manifest["entries"]
            if entry.get("kind") == kind
        }
        for kind in ("arena_impls", "parser_impls", "renderer_impls")
    }

    assert entries_by_kind["arena_impls"]["gomoku_local_v1"]["module"] == (
        "gage_eval.game_kits.board_game.gomoku.environment"
    )
    assert entries_by_kind["arena_impls"]["tictactoe_v1"]["module"] == (
        "gage_eval.game_kits.board_game.tictactoe.environment"
    )
    assert entries_by_kind["parser_impls"]["gomoku_v1"]["module"] == (
        "gage_eval.game_kits.board_game.gomoku.parser"
    )
    assert entries_by_kind["parser_impls"]["grid_parser_v1"]["module"] == (
        "gage_eval.game_kits.board_game.tictactoe.parser"
    )
    assert entries_by_kind["renderer_impls"]["gomoku_board_v1"]["module"] == (
        "gage_eval.game_kits.board_game.gomoku.board_renderer"
    )
    assert entries_by_kind["renderer_impls"]["tictactoe_board_v1"]["module"] == (
        "gage_eval.game_kits.board_game.tictactoe.renderer"
    )
