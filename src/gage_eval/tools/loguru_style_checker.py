"""Detect non-brace formatting in loguru logger calls."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence
import warnings

LOGURU_METHODS = {
    "trace",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "critical",
    "exception",
    "log",
}
LOGURU_CHAIN_METHODS = {"bind", "opt", "patch"}
PERCENT_PLACEHOLDER_RE = re.compile(r"(?<!%)%[sdri]")


@dataclass(frozen=True, slots=True)
class LoguruStyleViolation:
    """Represents one loguru formatting style violation."""

    path: str
    line: int
    col: int
    message: str


def find_violations(paths: Sequence[Path]) -> list[LoguruStyleViolation]:
    """Return all loguru formatting violations under the provided paths."""

    violations: list[LoguruStyleViolation] = []
    for file_path in _iter_python_files(paths):
        source = file_path.read_text(encoding="utf-8")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            continue
        aliases = _collect_loguru_aliases(tree)
        if not aliases.logger_names and not aliases.module_names:
            continue
        visitor = _LoguruViolationVisitor(file_path=file_path, aliases=aliases)
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the loguru style checker CLI."""

    parser = argparse.ArgumentParser(description="Check that loguru calls use brace-style formatting.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src/gage_eval"],
        help="Files or directories to scan.",
    )
    args = parser.parse_args(argv)

    violations = find_violations([Path(item) for item in args.paths])
    if not violations:
        print("loguru style check passed")
        return 0

    for violation in violations:
        print(f"{violation.path}:{violation.line}:{violation.col}: {violation.message}")
    print(f"loguru style check failed with {len(violations)} violation(s)")
    return 1


@dataclass(frozen=True, slots=True)
class _LoguruAliases:
    logger_names: frozenset[str]
    module_names: frozenset[str]


class _LoguruViolationVisitor(ast.NodeVisitor):
    def __init__(self, *, file_path: Path, aliases: _LoguruAliases) -> None:
        self._file_path = file_path
        self._aliases = aliases
        self.violations: list[LoguruStyleViolation] = []

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_loguru_call(node):
            message = self._string_literal(node.args[0]) if node.args else None
            if message and len(node.args) > 1 and PERCENT_PLACEHOLDER_RE.search(message):
                self.violations.append(
                    LoguruStyleViolation(
                        path=str(self._file_path),
                        line=node.lineno,
                        col=node.col_offset + 1,
                        message="Use brace-style formatting in loguru calls instead of % placeholders.",
                    )
                )
        self.generic_visit(node)

    def _is_loguru_call(self, node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr not in LOGURU_METHODS:
            return False
        return _is_loguru_logger_expr(node.func.value, self._aliases)

    @staticmethod
    def _string_literal(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None


def _collect_loguru_aliases(tree: ast.AST) -> _LoguruAliases:
    logger_names: set[str] = set()
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "loguru":
            for alias in node.names:
                if alias.name == "logger":
                    logger_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "loguru":
                    module_names.add(alias.asname or alias.name)
    return _LoguruAliases(
        logger_names=frozenset(logger_names),
        module_names=frozenset(module_names),
    )


def _is_loguru_logger_expr(node: ast.AST, aliases: _LoguruAliases) -> bool:
    if isinstance(node, ast.Name):
        return node.id in aliases.logger_names
    if isinstance(node, ast.Attribute):
        return (
            node.attr == "logger"
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases.module_names
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in LOGURU_CHAIN_METHODS and _is_loguru_logger_expr(node.func.value, aliases)
    return False


def _iter_python_files(paths: Sequence[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
            continue
        if not path.is_dir():
            continue
        yield from sorted(candidate for candidate in path.rglob("*.py") if candidate.is_file())
