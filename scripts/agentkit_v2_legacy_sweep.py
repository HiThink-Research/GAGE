from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


AGENT_LEGACY_JUDGE_MODULES = frozenset(
    {
        "appworld_evaluate",
        "swebench_docker",
        "tau2_eval",
    }
)
STATIC_ROLE_JUDGE_MODULES = frozenset({"base.py", "gomoku_referee.py"})
LEGACY_BENCHMARK_HELPERS_MODULE = "gage_eval.utils.benchmark_helpers"
TRANSITION_SHIM_PATHS = frozenset(
    {
        Path("src/gage_eval/agent_eval_kits/swebench/judge_bridge.py"),
        Path("src/gage_eval/agent_eval_kits/swebench/resources.py"),
        Path("src/gage_eval/agent_eval_kits/swebench/units.py"),
        Path("src/gage_eval/agent_eval_kits/tau2/judge_bridge.py"),
        Path("src/gage_eval/agent_eval_kits/tau2/resources.py"),
        Path("src/gage_eval/agent_eval_kits/tau2/units.py"),
        Path("src/gage_eval/role/agent/human_gateway.py"),
    }
)
REQUIRED_V2_KIT_KEYWORDS = frozenset(
    {
        "default_environment_provider",
        "default_environment_profile_by_provider",
        "environment_profiles",
        "verifier_environment_policy",
        "verifier_environment_profile_id",
        "verifier_adapter_factory",
        "artifact_manifest_factory",
    }
)
FORBIDDEN_V1_KIT_KEYWORDS = frozenset(
    {
        "benchmark_kit_id",
        "verifier_kind",
        "resource_requirements",
        "lifecycle_policy",
        "state_schema_keys",
        "verifier_resource_resolver",
        "trace_mapper",
    }
)
SWEEP_TOOLING_EXCLUDED_PATHS = frozenset(
    {
        Path("scripts/agentkit_v2_legacy_sweep.py"),
        Path("tests/unit/architecture/test_agentkit_v2_boundaries.py"),
        Path("tests/unit/scripts/test_agentkit_v2_legacy_sweep.py"),
    }
)


@dataclass(frozen=True, order=True)
class LegacySweepFinding:
    category: str
    path: str
    line: int
    detail: str


@dataclass(frozen=True)
class LegacySweepReport:
    root: Path
    findings: tuple[LegacySweepFinding, ...]
    static_retained: tuple[str, ...]

    def counts_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.category] = counts.get(finding.category, 0) + 1
        return dict(sorted(counts.items()))

    def for_category(self, category: str) -> tuple[LegacySweepFinding, ...]:
        return tuple(finding for finding in self.findings if finding.category == category)

    def to_markdown(self) -> str:
        lines = [
            "# AgentKit v2 Legacy Sweep Report",
            "",
            f"- Root: `{self.root}`",
            f"- Total findings: {len(self.findings)}",
            "",
            "## Summary",
            "",
        ]
        counts = self.counts_by_category()
        if counts:
            lines.extend(["| Category | Count |", "| --- | ---: |"])
            lines.extend(f"| `{category}` | {count} |" for category, count in counts.items())
        else:
            lines.append("No legacy blockers found.")

        lines.extend(["", "## Findings", ""])
        if self.findings:
            for category in sorted(counts):
                lines.extend([f"### `{category}`", ""])
                for finding in self.for_category(category):
                    lines.append(
                        f"- `{finding.path}:{finding.line}` - {finding.detail}"
                    )
                lines.append("")
        else:
            lines.append("No findings.")
            lines.append("")

        lines.extend(["## Static Modules Intentionally Retained", ""])
        if self.static_retained:
            lines.extend(f"- `{path}`" for path in self.static_retained)
        else:
            lines.append("No static role modules detected.")

        return "\n".join(lines).rstrip() + "\n"


def build_legacy_sweep_report(root: Path) -> LegacySweepReport:
    root = root.resolve()
    findings: set[LegacySweepFinding] = set()
    static_retained = _collect_static_retained(root)

    legacy_game_eval_kits = root / "src/gage_eval/game_eval_kits"
    if legacy_game_eval_kits.exists():
        _add(
            findings,
            "legacy_game_eval_kits_package",
            root,
            legacy_game_eval_kits,
            1,
            "legacy game_eval_kits package is still present",
        )

    role_agent_backends = root / "src/gage_eval/role/agent/backends"
    if role_agent_backends.exists():
        _add(
            findings,
            "role_agent_backends_package",
            root,
            role_agent_backends,
            1,
            "legacy role.agent.backends package is still present",
        )

    for module_name in sorted(AGENT_LEGACY_JUDGE_MODULES):
        module_path = root / "src/gage_eval/role/judge" / f"{module_name}.py"
        if module_path.exists():
            _add(
                findings,
                "agent_legacy_judge_module",
                root,
                module_path,
                1,
                f"legacy agent judge module gage_eval.role.judge.{module_name} is still present",
            )

    helpers_package = root / "src/gage_eval/utils/benchmark_helpers"
    if helpers_package.exists():
        _add(
            findings,
            "utils_benchmark_helpers_package",
            root,
            helpers_package,
            1,
            "benchmark helper package must be kit-private under agent_eval_kits/<kit>",
        )

    for rel_path in sorted(TRANSITION_SHIM_PATHS):
        path = root / rel_path
        if path.exists():
            _add(
                findings,
                "transition_shim_module",
                root,
                path,
                1,
                "transition shim module must be removed after v2 owner migration",
            )

    for path in _python_files(root):
        rel_path = path.relative_to(root)
        if rel_path in SWEEP_TOOLING_EXCLUDED_PATHS:
            continue
        text = path.read_text(encoding="utf-8")
        _scan_imports(root=root, path=path, text=text, findings=findings)
        _scan_text_references(root=root, path=path, text=text, findings=findings)
        _scan_kit_entry_keywords(root=root, path=path, text=text, findings=findings)

    return LegacySweepReport(
        root=root,
        findings=tuple(sorted(findings)),
        static_retained=tuple(sorted(static_retained)),
    )


def _scan_imports(
    *, root: Path, path: Path, text: str, findings: set[LegacySweepFinding]
) -> None:
    rel_path = path.relative_to(root)
    try:
        tree = ast.parse(text, filename=rel_path.as_posix())
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _record_module_reference(
                    root=root,
                    path=path,
                    line=node.lineno,
                    module=alias.name,
                    findings=findings,
                )
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from_module(rel_path, node)
            if module is None:
                continue
            parent_matched = _record_module_reference(
                root=root,
                path=path,
                line=node.lineno,
                module=module,
                findings=findings,
            )
            if parent_matched:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                _record_module_reference(
                    root=root,
                    path=path,
                    line=node.lineno,
                    module=f"{module}.{alias.name}",
                    findings=findings,
                )


def _scan_text_references(
    *, root: Path, path: Path, text: str, findings: set[LegacySweepFinding]
) -> None:
    rel_path = path.relative_to(root)
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("from ", "import ")):
            continue
        if "game_eval_kits" in line and not _is_under(
            rel_path, "src/gage_eval/game_eval_kits"
        ):
            _add(
                findings,
                "legacy_game_eval_kits_reference",
                root,
                path,
                line_number,
                "references legacy game_eval_kits",
            )
        if "gage_eval.sandbox" in line and not _is_under(
            rel_path, "src/gage_eval/sandbox"
        ):
            _add(
                findings,
                "sandbox_import_reference",
                root,
                path,
                line_number,
                "references gage_eval.sandbox outside the sandbox package",
            )
        if "role.agent.backends" in line or "gage_eval.role.agent.backends" in line:
            _add(
                findings,
                "role_agent_backends_reference",
                root,
                path,
                line_number,
                "references legacy role.agent.backends",
            )
        for module_name in AGENT_LEGACY_JUDGE_MODULES:
            needle = f"gage_eval.role.judge.{module_name}"
            if needle in line:
                _add(
                    findings,
                    "agent_legacy_judge_import",
                    root,
                    path,
                    line_number,
                    f"references legacy agent judge module {needle}",
                )


def _record_module_reference(
    *,
    root: Path,
    path: Path,
    line: int,
    module: str,
    findings: set[LegacySweepFinding],
) -> bool:
    matched = False
    rel_path = path.relative_to(root)
    if _module_matches(module, "gage_eval.game_eval_kits") and not _is_under(
        rel_path, "src/gage_eval/game_eval_kits"
    ):
        _add(
            findings,
            "legacy_game_eval_kits_reference",
            root,
            path,
            line,
            f"imports legacy {module}",
        )
        matched = True
    if _module_matches(module, "gage_eval.sandbox") and not _is_under(
        rel_path, "src/gage_eval/sandbox"
    ):
        _add(
            findings,
            "sandbox_import_reference",
            root,
            path,
            line,
            f"imports {module} outside the sandbox package",
        )
        matched = True
    if _module_matches(module, "gage_eval.role.agent.backends"):
        _add(
            findings,
            "role_agent_backends_reference",
            root,
            path,
            line,
            f"imports legacy {module}",
        )
        matched = True
    for module_name in AGENT_LEGACY_JUDGE_MODULES:
        legacy_module = f"gage_eval.role.judge.{module_name}"
        if _module_matches(module, legacy_module):
            _add(
                findings,
                "agent_legacy_judge_import",
                root,
                path,
                line,
                f"imports legacy agent judge module {legacy_module}",
            )
            matched = True
    if _module_matches(module, LEGACY_BENCHMARK_HELPERS_MODULE):
        _add(
            findings,
            "utils_benchmark_helpers_reference",
            root,
            path,
            line,
            f"imports non-kit benchmark helper module {module}",
        )
        matched = True
    if _is_transition_shim_module(module):
        _add(
            findings,
            "transition_shim_import",
            root,
            path,
            line,
            f"imports transition shim module {module}",
        )
        matched = True
    return matched


def _scan_kit_entry_keywords(
    *, root: Path, path: Path, text: str, findings: set[LegacySweepFinding]
) -> None:
    rel_path = path.relative_to(root)
    if rel_path not in {
        Path("src/gage_eval/agent_eval_kits/appworld/kit.py"),
        Path("src/gage_eval/agent_eval_kits/swebench/kit.py"),
        Path("src/gage_eval/agent_eval_kits/tau2/kit.py"),
        Path("src/gage_eval/agent_eval_kits/terminal_bench/kit.py"),
    }:
        return
    try:
        tree = ast.parse(text, filename=rel_path.as_posix())
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "BenchmarkKitEntry":
            continue
        keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
        missing = sorted(REQUIRED_V2_KIT_KEYWORDS - keywords)
        if missing:
            _add(
                findings,
                "benchmark_kit_entry_missing_v2_keywords",
                root,
                path,
                getattr(node, "lineno", 1),
                f"BenchmarkKitEntry missing explicit v2 keywords: {', '.join(missing)}",
            )
        forbidden = sorted(FORBIDDEN_V1_KIT_KEYWORDS & keywords)
        if forbidden:
            _add(
                findings,
                "benchmark_kit_entry_v1_keywords",
                root,
                path,
                getattr(node, "lineno", 1),
                f"BenchmarkKitEntry still declares v1 keywords: {', '.join(forbidden)}",
            )


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_transition_shim_module(module: str) -> bool:
    return any(
        _module_matches(module, shim)
        for shim in (
            "gage_eval.agent_eval_kits.swebench.judge_bridge",
            "gage_eval.agent_eval_kits.swebench.resources",
            "gage_eval.agent_eval_kits.swebench.units",
            "gage_eval.agent_eval_kits.tau2.judge_bridge",
            "gage_eval.agent_eval_kits.tau2.resources",
            "gage_eval.agent_eval_kits.tau2.units",
            "gage_eval.role.agent.human_gateway",
        )
    )


def _collect_static_retained(root: Path) -> set[str]:
    retained: set[str] = set()
    role_model_backends = root / "src/gage_eval/role/model/backends"
    if role_model_backends.exists():
        retained.add(_relative_path(root, role_model_backends))

    role_judge = root / "src/gage_eval/role/judge"
    for filename in sorted(STATIC_ROLE_JUDGE_MODULES):
        path = role_judge / filename
        if path.exists():
            retained.add(_relative_path(root, path))
    return retained


def _add(
    findings: set[LegacySweepFinding],
    category: str,
    root: Path,
    path: Path,
    line: int,
    detail: str,
) -> None:
    findings.add(
        LegacySweepFinding(
            category=category,
            path=_relative_path(root, path),
            line=line,
            detail=detail,
        )
    )


def _python_files(root: Path) -> Iterable[Path]:
    for top_level in ("src", "tests", "scripts"):
        directory = root / top_level
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            if "__pycache__" not in path.parts:
                yield path


def _module_matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _is_under(rel_path: Path, prefix: str) -> bool:
    prefix_parts = Path(prefix).parts
    return rel_path.parts[: len(prefix_parts)] == prefix_parts


def _resolve_import_from_module(rel_path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    if not _is_under(rel_path, "src/gage_eval"):
        return "." * node.level + (node.module or "")

    package_parts = _package_parts_for_module_file(rel_path)
    keep_count = len(package_parts) - node.level + 1
    if keep_count < 1:
        return "." * node.level + (node.module or "")

    resolved_parts = list(package_parts[:keep_count])
    if node.module:
        resolved_parts.extend(node.module.split("."))
    return ".".join(resolved_parts)


def _package_parts_for_module_file(rel_path: Path) -> tuple[str, ...]:
    src_relative = rel_path.relative_to(Path("src"))
    module_parts = src_relative.with_suffix("").parts
    if module_parts[-1] == "__init__":
        return module_parts[:-1]
    return module_parts[:-1]


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to scan. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--write",
        type=Path,
        help="Optional markdown report path. Relative paths are resolved from cwd.",
    )
    args = parser.parse_args()

    report = build_legacy_sweep_report(args.root)
    markdown = report.to_markdown()
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
