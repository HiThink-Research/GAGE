from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Sequence

import pytest

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink


BOUNDARY_MODE_ENV = "GAGE_AGENTKIT_V2_BOUNDARY_MODE"
ADVISORY = "advisory"
STRICT = "strict"

PROVIDER_SDK_MODULES = frozenset({"docker", "e2b", "opensandbox", "paramiko"})
HOST_LLM_MODULES = frozenset({"litellm", "openai"})
MODULE_OPEN_ALLOWLIST = frozenset({"tarfile", "zipfile", "gzip", "bz2", "lzma"})
LEGACY_SANDBOX_MODULE = ".".join(("gage_eval", "sandbox"))
LEGACY_GAME_EVAL_KITS_MODULE = ".".join(("gage_eval", "game_eval_kits"))
LEGACY_ROLE_AGENT_BACKENDS_MODULE = ".".join(("gage_eval", "role", "agent", "backends"))
LEGACY_ROLE_AGENT_CORE_MODULES = frozenset(
    {
        LEGACY_ROLE_AGENT_BACKENDS_MODULE,
        "gage_eval.role.agent.human_gateway",
        "gage_eval.role.agent.loop",
        "gage_eval.role.agent.tool_router",
    }
)
TAU2_HOST_LLM_ALLOWLIST = frozenset(
    {Path("src/gage_eval/agent_eval_kits/tau2/runtime.py")}
)
ARTIFACT_WRITE_ALLOWED_PATH_PARTS = (
    ("src", "gage_eval", "agent_runtime", "artifacts.py"),
    ("scripts", "tools", "config"),
    ("scripts", "tools", "migrations"),
    ("scripts", "migration"),
    ("scripts", "migrations"),
)


@dataclass(frozen=True, order=True)
class BoundaryViolation:
    rule: str
    path: str
    line: int
    detail: str


@dataclass(frozen=True)
class BoundaryReport:
    root: Path
    violations: tuple[BoundaryViolation, ...]

    def for_rule(self, rule: str) -> tuple[BoundaryViolation, ...]:
        return tuple(violation for violation in self.violations if violation.rule == rule)

    def counts_by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for violation in self.violations:
            counts[violation.rule] = counts.get(violation.rule, 0) + 1
        return dict(sorted(counts.items()))

    def format(self, *, limit: int | None = None) -> str:
        violations = self.violations if limit is None else self.violations[:limit]
        lines = [
            f"{violation.rule}: {violation.path}:{violation.line} - {violation.detail}"
            for violation in violations
        ]
        remaining = len(self.violations) - len(violations)
        if remaining > 0:
            lines.append(f"... {remaining} more violation(s)")
        return "\n".join(lines)


class _BoundaryVisitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: Path) -> None:
        self.rel_path = rel_path
        self.violations: list[BoundaryViolation] = []
        self.base_environment_aliases = {"BaseEnvironment"}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(module=alias.name, node=node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _resolve_import_from_module(self.rel_path, node)
        if module is None:
            self.generic_visit(node)
            return

        parent_had_violation = self._check_import(module=module, node=node)
        if not parent_had_violation:
            for alias in node.names:
                if alias.name == "*":
                    continue
                self._check_import(module=f"{module}.{alias.name}", node=node)
        if module in {"gage_eval.environment", "gage_eval.environment.contracts"}:
            for alias in node.names:
                if alias.name == "BaseEnvironment":
                    self.base_environment_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _annotation_mentions_base_environment(
            node.value, aliases=self.base_environment_aliases
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.base_environment_aliases.add(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and _annotation_mentions_base_environment(
            node.value, aliases=self.base_environment_aliases
        ):
            if isinstance(node.target, ast.Name):
                self.base_environment_aliases.add(node.target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_environment_parameter(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_environment_parameter(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._check_environment_duck_typing(node)
        self._check_artifact_direct_write(node)
        self.generic_visit(node)

    def _check_import(self, *, module: str, node: ast.AST) -> bool:
        added_violation = False
        if _is_provider_sdk_module(module) and not _is_provider_sdk_allowed(self.rel_path):
            self._add(
                "provider_sdk_import_outside_environment_providers",
                node,
                f"provider SDK import {module!r} is only allowed under environment/providers",
            )
            added_violation = True

        if _is_legacy_role_agent_core_import(module) and not _is_legacy_role_agent_shim(
            self.rel_path
        ):
            self._add(
                "legacy_role_agent_core_import",
                node,
                f"legacy role.agent core import {module!r} must be migrated to agent_runtime",
            )
            added_violation = True

        if not _is_under(self.rel_path, "src/gage_eval"):
            return added_violation

        if _is_under(self.rel_path, "src/gage_eval/agent_runtime") and _module_matches(
            module, "gage_eval.role.agent"
        ):
            self._add(
                "agent_runtime_imports_role_agent",
                node,
                f"agent_runtime must not import {module!r}",
            )
            added_violation = True

        if (
            _is_under(self.rel_path, "src/gage_eval/agent_eval_kits")
            and _is_host_llm_module(module)
            and self.rel_path not in TAU2_HOST_LLM_ALLOWLIST
        ):
            self._add(
                "non_tau2_kit_imports_host_llm",
                node,
                f"non-Tau2 kit file must not import host-side LLM module {module!r}",
            )
            added_violation = True

        if _is_under(self.rel_path, "src/gage_eval/environment/providers") and _module_matches(
            module, "gage_eval.agent_eval_kits"
        ):
            self._add(
                "environment_provider_imports_agent_eval_kits",
                node,
                f"environment provider must not import benchmark kit module {module!r}",
            )
            added_violation = True

        if _module_matches(module, LEGACY_SANDBOX_MODULE) and not _is_under(
            self.rel_path, "src/gage_eval/sandbox"
        ):
            self._add(
                "sandbox_import",
                node,
                f"new imports from {module!r} are blocked outside the legacy environment package",
            )
            added_violation = True

        if _module_matches(module, LEGACY_GAME_EVAL_KITS_MODULE) and not _is_under(
            self.rel_path, "src/gage_eval/game_eval_kits"
        ):
            self._add(
                "game_eval_kits_import",
                node,
                f"imports from legacy {module!r} are blocked",
            )
            added_violation = True

        return added_violation

    def _check_environment_parameter(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        if not _is_kit_runtime_judge_or_tool_file(self.rel_path):
            return

        args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        for arg in args:
            if arg.arg != "environment":
                continue
            if arg.annotation is not None and _annotation_mentions_base_environment(
                arg.annotation, aliases=self.base_environment_aliases
            ):
                continue
            self._add(
                "environment_param_not_base_environment_typed",
                arg,
                "kit runtime/judge/tool environment parameter must be typed as BaseEnvironment or an alias",
            )

    def _check_environment_duck_typing(self, node: ast.Call) -> None:
        if not _is_under(self.rel_path, "src/gage_eval/agent_eval_kits"):
            return

        if isinstance(node.func, ast.Name) and node.func.id in {"hasattr", "getattr"}:
            if node.args and _is_environment_name(node.args[0]):
                self._add(
                    "environment_duck_typing",
                    node,
                    f"{node.func.id}(environment, ...) is blocked in AgentKit v2 kit code",
                )
            return

        if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            if len(node.args) < 2 or not _is_environment_name(node.args[0]):
                return
            type_name = _expr_name(node.args[1])
            if type_name and _looks_like_provider_or_environment_subclass(type_name):
                self._add(
                    "environment_duck_typing",
                    node,
                    f"isinstance(environment, {type_name}) is blocked in AgentKit v2 kit code",
                )

    def _check_artifact_direct_write(self, node: ast.Call) -> None:
        if _is_artifact_write_allowed(self.rel_path):
            return
        if not _is_under(self.rel_path, "src/gage_eval"):
            return

        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "write_text",
            "write_bytes",
        }:
            self._add(
                "artifact_direct_write",
                node,
                f"direct {node.func.attr} artifact writes should go through RuntimeArtifactSink",
            )
            return

        if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
            if _is_module_open_call(node.func):
                return
            mode = _literal_arg(node, 0) or _literal_kwarg(node, "mode")
            if _is_write_mode(mode):
                self._add(
                    "artifact_direct_write",
                    node,
                    "direct Path.open write-mode artifact writes should go through RuntimeArtifactSink",
                )
            return

        if isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = _literal_arg(node, 1) or _literal_kwarg(node, "mode")
            if _is_write_mode(mode):
                self._add(
                    "artifact_direct_write",
                    node,
                    "direct open write-mode artifact writes should go through RuntimeArtifactSink",
                )

    def _add(self, rule: str, node: ast.AST, detail: str) -> None:
        self.violations.append(
            BoundaryViolation(
                rule=rule,
                path=self.rel_path.as_posix(),
                line=getattr(node, "lineno", 1),
                detail=detail,
            )
        )


def collect_agentkit_v2_boundary_violations(root: Path) -> BoundaryReport:
    root = root.resolve()
    violations: list[BoundaryViolation] = []
    for path in _python_files(root):
        rel_path = path.relative_to(root)
        if "__pycache__" in rel_path.parts:
            continue
        if not _should_scan(rel_path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel_path))
        except SyntaxError as exc:
            violations.append(
                BoundaryViolation(
                    rule="python_parse_error",
                    path=rel_path.as_posix(),
                    line=exc.lineno or 1,
                    detail=str(exc),
                )
            )
            continue
        visitor = _BoundaryVisitor(rel_path=rel_path)
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return BoundaryReport(root=root, violations=tuple(sorted(violations)))


def enforce_agentkit_v2_boundary_report(
    report: BoundaryReport, mode: str | None = None
) -> str | None:
    selected_mode = (mode or os.environ.get(BOUNDARY_MODE_ENV, ADVISORY)).strip().lower()
    if selected_mode not in {ADVISORY, STRICT}:
        raise ValueError(
            f"{BOUNDARY_MODE_ENV} must be {ADVISORY!r} or {STRICT!r}, got {selected_mode!r}"
        )
    if selected_mode == ADVISORY:
        if not report.violations:
            return None
        advisory_report = _format_advisory_report(report)
        print(advisory_report)
        return advisory_report

    if not report.violations:
        return

    pytest.fail(
        "AgentKit v2 boundary lint failed in strict mode:\n"
        f"{report.format(limit=50)}",
        pytrace=False,
    )


def test_import_boundary_rules_detect_synthetic_violations(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gage_eval/agent_runtime/resolver.py",
        f"""
        from gage_eval.role.agent.tool_router import ToolRouter
        from {LEGACY_SANDBOX_MODULE}.manager import SandboxManager
        """,
    )
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/appworld/runtime.py",
        f"""
        import docker
        import litellm
        from {LEGACY_GAME_EVAL_KITS_MODULE}.board_game import tictactoe
        """,
    )
    _write(
        tmp_path / "src/gage_eval/environment/providers/docker/provider.py",
        """
        import docker
        """,
    )
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/tau2/runtime.py",
        """
        import litellm
        """,
    )
    _write(
        tmp_path / "src/gage_eval/sandbox/manager.py",
        f"""
        from {LEGACY_SANDBOX_MODULE}.base import BaseSandbox
        """,
    )

    report = collect_agentkit_v2_boundary_violations(tmp_path)

    assert _rules(report) == {
        "agent_runtime_imports_role_agent",
        "game_eval_kits_import",
        "legacy_role_agent_core_import",
        "non_tau2_kit_imports_host_llm",
        "provider_sdk_import_outside_environment_providers",
        "sandbox_import",
    }


def test_relative_imports_are_resolved_before_boundary_checks(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gage_eval/agent_runtime/resolver.py",
        """
        from ..role.agent.tool_router import ToolRouter
        from ..sandbox.provider import SandboxProvider
        """,
    )
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/demo/runtime.py",
        f"""
        from ...{LEGACY_GAME_EVAL_KITS_MODULE.split('.', 1)[1]}.board_game import tictactoe
        """,
    )

    report = collect_agentkit_v2_boundary_violations(tmp_path)

    assert _rules(report) == {
        "agent_runtime_imports_role_agent",
        "game_eval_kits_import",
        "legacy_role_agent_core_import",
        "sandbox_import",
    }
    assert any("gage_eval.role.agent.tool_router" in violation.detail for violation in report.violations)
    assert any(f"{LEGACY_SANDBOX_MODULE}.provider" in violation.detail for violation in report.violations)
    assert any(f"{LEGACY_GAME_EVAL_KITS_MODULE}.board_game" in violation.detail for violation in report.violations)


def test_import_from_aliases_are_checked_as_full_modules(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gage_eval/agent_runtime/resolver.py",
        f"""
        from gage_eval import sandbox
        from gage_eval import game_eval_kits
        from gage_eval.role import agent
        from ..role import agent as relative_agent
        """,
    )

    report = collect_agentkit_v2_boundary_violations(tmp_path)

    assert report.counts_by_rule() == {
        "agent_runtime_imports_role_agent": 2,
        "game_eval_kits_import": 1,
        "sandbox_import": 1,
    }
    assert any(LEGACY_SANDBOX_MODULE in violation.detail for violation in report.violations)
    assert any(LEGACY_GAME_EVAL_KITS_MODULE in violation.detail for violation in report.violations)
    assert any("gage_eval.role.agent" in violation.detail for violation in report.violations)


def test_legacy_role_agent_core_imports_are_flagged_outside_shims(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests/unit/role/test_old_agent_loop.py",
        f"""
        from gage_eval.role.agent.loop import AgentLoop
        from gage_eval.role.agent.tool_router import ToolRouter
        from gage_eval.role.agent.human_gateway import HumanGateway
        from {LEGACY_ROLE_AGENT_BACKENDS_MODULE}.base import AgentBackend
        """,
    )
    _write(
        tmp_path / "src/gage_eval/config/registry.py",
        f"""
        from {LEGACY_ROLE_AGENT_BACKENDS_MODULE} import build_agent_backend
        """,
    )
    _write(
        tmp_path / "src/gage_eval/role/agent/tool_router.py",
        """
        from gage_eval.agent_runtime.tooling.router import ToolRouter
        """,
    )

    report = collect_agentkit_v2_boundary_violations(tmp_path)

    assert [(violation.path, violation.line) for violation in report.for_rule(
        "legacy_role_agent_core_import"
    )] == [
        ("src/gage_eval/config/registry.py", 1),
        ("tests/unit/role/test_old_agent_loop.py", 1),
        ("tests/unit/role/test_old_agent_loop.py", 2),
        ("tests/unit/role/test_old_agent_loop.py", 3),
        ("tests/unit/role/test_old_agent_loop.py", 4),
    ]


def test_provider_sdk_import_boundary_applies_to_tests_and_scripts(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests/unit/environment/test_docker_provider.py",
        """
        import docker
        """,
    )
    _write(
        tmp_path / "scripts/tools/check_e2b.py",
        """
        from e2b import Sandbox
        """,
    )
    _write(
        tmp_path / "src/gage_eval/environment/providers/docker/provider.py",
        """
        import docker
        """,
    )

    report = collect_agentkit_v2_boundary_violations(tmp_path)

    assert [(violation.path, violation.line) for violation in report.for_rule(
        "provider_sdk_import_outside_environment_providers"
    )] == [
        ("scripts/tools/check_e2b.py", 1),
        ("tests/unit/environment/test_docker_provider.py", 1),
    ]


def test_environment_provider_does_not_import_agent_eval_kits(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gage_eval/environment/providers/local_process/provider.py",
        """
        from gage_eval.agent_eval_kits.tau2.local_runtime import Tau2Runtime
        """,
    )

    report = collect_agentkit_v2_boundary_violations(tmp_path)

    assert [(violation.path, violation.line) for violation in report.for_rule(
        "environment_provider_imports_agent_eval_kits"
    )] == [
        ("src/gage_eval/environment/providers/local_process/provider.py", 1),
    ]


def test_environment_param_must_be_base_environment_typed(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/demo/runtime.py",
        """
        from gage_eval.environment.contracts import BaseEnvironment as EnvProtocol

        async def valid_runtime(environment: EnvProtocol) -> None:
            pass

        async def invalid_runtime(environment) -> None:
            pass
        """,
    )
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/demo/judge/scorer.py",
        """
        from gage_eval.environment import BaseEnvironment
        KitEnvironment = BaseEnvironment

        def valid_judge(environment: KitEnvironment) -> None:
            pass

        def invalid_judge(environment: object) -> None:
            pass
        """,
    )

    report = collect_agentkit_v2_boundary_violations(tmp_path)

    violations = report.for_rule("environment_param_not_base_environment_typed")
    assert [(violation.path, violation.line) for violation in violations] == [
        ("src/gage_eval/agent_eval_kits/demo/judge/scorer.py", 7),
        ("src/gage_eval/agent_eval_kits/demo/runtime.py", 6),
    ]


def test_environment_param_lint_covers_tools_directories(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/foo/tools/shell.py",
        """
        def run_shell(environment) -> None:
            pass
        """,
    )

    report = collect_agentkit_v2_boundary_violations(tmp_path)

    assert [(violation.path, violation.line) for violation in report.for_rule(
        "environment_param_not_base_environment_typed"
    )] == [
        ("src/gage_eval/agent_eval_kits/foo/tools/shell.py", 1),
    ]


def test_no_hasattr_getattr_on_environment_in_kit(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/demo/runtime.py",
        """
        from gage_eval.environment import BaseEnvironment

        class DockerEnvironment:
            pass

        def run(environment: BaseEnvironment) -> None:
            hasattr(environment, "exec")
            getattr(environment, "exec")
            isinstance(environment, DockerEnvironment)
            getattr(other, "exec")
        """,
    )

    report = collect_agentkit_v2_boundary_violations(tmp_path)

    violations = report.for_rule("environment_duck_typing")
    assert [violation.line for violation in violations] == [7, 8, 9]


def test_artifact_direct_write_flags_kit_and_runtime_writes_but_allows_tests_and_migrations(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/demo/artifacts.py",
        """
        def write(target):
            target.write_text("artifact", encoding="utf-8")
        """,
    )
    _write(
        tmp_path / "src/gage_eval/agent_runtime/clients/codex.py",
        """
        def write(target):
            with target.open("w", encoding="utf-8") as handle:
                handle.write("artifact")
        """,
    )
    _write(
        tmp_path / "src/gage_eval/config/writer.py",
        """
        def write(target):
            target.write_text("config artifact", encoding="utf-8")
        """,
    )
    _write(
        tmp_path / "src/gage_eval/environment/providers/docker/provider.py",
        """
        import tarfile

        def archive(payload, target):
            with tarfile.open(fileobj=payload, mode="w") as archive:
                pass
            with target.open("w", encoding="utf-8") as handle:
                handle.write("artifact")
        """,
    )
    _write(
        tmp_path / "tests/unit/agent_eval_kits/test_demo.py",
        """
        def write(target):
            target.write_text("fixture", encoding="utf-8")
        """,
    )
    _write(
        tmp_path / "scripts/migrations/rewrite_artifacts.py",
        """
        def write(target):
            target.write_text("migration", encoding="utf-8")
        """,
    )

    report = collect_agentkit_v2_boundary_violations(tmp_path)

    assert [(violation.path, violation.line) for violation in report.for_rule("artifact_direct_write")] == [
        ("src/gage_eval/agent_eval_kits/demo/artifacts.py", 2),
        ("src/gage_eval/agent_runtime/clients/codex.py", 2),
        ("src/gage_eval/config/writer.py", 2),
        ("src/gage_eval/environment/providers/docker/provider.py", 6),
    ]


def test_effective_config_json_secret_redaction_fixture(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    secret = "sk-task-05a-secret"
    sentinel = "<redacted:reference:ENV.MODEL_API_KEY>"

    ref = sink.write_effective_config(
        run_id="run-redaction",
        task_id="task-1",
        sample_id="sample-1",
        final_config={"backend": {"api_key": secret, "model": "demo"}},
        source_layers=[
            {"name": "raw_config", "values": {"backend": {"api_key": secret}}},
            {"name": "cli_overrides", "values": {"backend": {"model": "demo"}}},
        ],
        secret_values={secret: sentinel},
    )

    text = (tmp_path / "run-redaction" / ref.path).read_text(encoding="utf-8")
    payload = json.loads(text)
    assert secret not in text
    assert sentinel in text
    assert payload["final_config"]["backend"]["api_key"] == sentinel


def test_agentkit_v2_boundary_lint_supports_advisory_then_strict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/demo/runtime.py",
        """
        import openai
        """,
    )
    report = collect_agentkit_v2_boundary_violations(tmp_path)
    assert report.for_rule("non_tau2_kit_imports_host_llm")

    enforce_agentkit_v2_boundary_report(report, mode=ADVISORY)
    advisory_output = capsys.readouterr().out
    assert "AgentKit v2 boundary lint advisory report" in advisory_output
    assert "non_tau2_kit_imports_host_llm" in advisory_output
    assert "src/gage_eval/agent_eval_kits/demo/runtime.py" in advisory_output

    with pytest.raises(pytest.fail.Exception, match="strict mode"):
        enforce_agentkit_v2_boundary_report(report, mode=STRICT)


def test_agentkit_v2_boundary_advisory_report_can_be_generated_for_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BOUNDARY_MODE_ENV, STRICT)
    repo_root = Path(__file__).resolve().parents[3]

    report = collect_agentkit_v2_boundary_violations(repo_root)

    assert isinstance(report.counts_by_rule(), dict)
    enforce_agentkit_v2_boundary_report(report, mode=ADVISORY)


def test_task06b_has_no_legacy_role_agent_runtime_ownership() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    report = collect_agentkit_v2_boundary_violations(repo_root)
    blocking = (
        report.for_rule("agent_runtime_imports_role_agent")
        + report.for_rule("legacy_role_agent_core_import")
    )

    assert not blocking, "\n".join(
        f"{violation.rule}: {violation.path}:{violation.line} - {violation.detail}"
        for violation in blocking
    )
    assert not (repo_root / "src/gage_eval/role/agent/backends").exists()
    assert not (repo_root / "src/gage_eval/role/agent/human_gateway.py").exists()


def test_repo_environment_providers_do_not_import_agent_eval_kits() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    report = collect_agentkit_v2_boundary_violations(repo_root)
    blocking = report.for_rule("environment_provider_imports_agent_eval_kits")

    assert not blocking, "\n".join(
        f"{violation.rule}: {violation.path}:{violation.line} - {violation.detail}"
        for violation in blocking
    )


def test_runtime_artifact_sink_does_not_write_under_samples_dir() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "src/gage_eval/agent_runtime/artifacts.py").read_text(encoding="utf-8")

    assert '"samples"' not in source
    assert "'samples'" not in source
    assert "samples/runtime" not in source


def test_no_legacy_sandbox_fields_in_swebench_v2_dataset_preprocess_kwargs() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config_paths = sorted(
        list((repo_root / "config/custom/manual_e2e").glob("agentkit_v2_swebench_pro_*.yaml"))
        + list((repo_root / "config/custom/swebench_pro").glob("v2_*.yaml"))
    )

    violations: list[str] = []
    for path in config_paths:
        source = path.read_text(encoding="utf-8")
        for token in ("sandbox_id:", "sandbox_runtime:", "sandbox_lifecycle:"):
            if token in source:
                violations.append(f"{path.relative_to(repo_root)} contains {token}")

    assert not violations, "\n".join(violations)


def test_static_role_model_backends_and_static_judges_are_not_deprecation_blockers(
    tmp_path: Path,
) -> None:
    build_legacy_sweep_report = _load_legacy_sweep_builder()

    _write(
        tmp_path / "src/gage_eval/role/model/backends/dummy_backend.py",
        """
        class DummyBackend:
            pass
        """,
    )
    _write(
        tmp_path / "src/gage_eval/role/judge/base.py",
        """
        class BaseJudge:
            pass
        """,
    )
    _write(
        tmp_path / "src/gage_eval/role/judge/gomoku_referee.py",
        """
        class GomokuReferee:
            pass
        """,
    )
    _write(
        tmp_path / "src/gage_eval/role/judge/tau2_eval.py",
        """
        class Tau2Evaluate:
            pass
        """,
    )

    report = build_legacy_sweep_report(tmp_path)

    assert "src/gage_eval/role/model/backends" in report.static_retained
    assert "src/gage_eval/role/judge/base.py" in report.static_retained
    assert "src/gage_eval/role/judge/gomoku_referee.py" in report.static_retained
    assert report.counts_by_category() == {"agent_legacy_judge_module": 1}
    assert not any("role/model/backends" in finding.path for finding in report.findings)
    assert not any(
        Path(finding.path).name in {"base.py", "gomoku_referee.py"}
        for finding in report.findings
    )


def _python_files(root: Path) -> Iterable[Path]:
    return root.rglob("*.py")


def _should_scan(rel_path: Path) -> bool:
    parts = rel_path.parts
    return (
        _is_under(rel_path, "src/gage_eval")
        or parts[:1] == ("tests",)
        or parts[:1] == ("scripts",)
    )


def _is_provider_sdk_module(module: str) -> bool:
    root_module = module.split(".", 1)[0]
    return root_module in PROVIDER_SDK_MODULES


def _is_provider_sdk_allowed(rel_path: Path) -> bool:
    return _is_under(rel_path, "src/gage_eval/environment/providers")


def _is_host_llm_module(module: str) -> bool:
    return module.split(".", 1)[0] in HOST_LLM_MODULES


def _module_matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _is_legacy_role_agent_core_import(module: str) -> bool:
    return any(_module_matches(module, legacy) for legacy in LEGACY_ROLE_AGENT_CORE_MODULES)


def _is_legacy_role_agent_shim(rel_path: Path) -> bool:
    return rel_path in {
        Path("src/gage_eval/role/agent/loop.py"),
        Path("src/gage_eval/role/agent/tool_router.py"),
        Path("src/gage_eval/role/agent/__init__.py"),
    }


def _is_kit_runtime_judge_or_tool_file(rel_path: Path) -> bool:
    if not _is_under(rel_path, "src/gage_eval/agent_eval_kits"):
        return False
    if rel_path.name == "runtime.py":
        return True
    if "judge" in rel_path.parts:
        return True
    if "tools" in rel_path.parts:
        return True
    return "tool" in rel_path.stem


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


def _format_advisory_report(report: BoundaryReport) -> str:
    counts = ", ".join(
        f"{rule}={count}" for rule, count in report.counts_by_rule().items()
    )
    return "\n".join(
        [
            "AgentKit v2 boundary lint advisory report",
            f"mode={ADVISORY}",
            f"root={report.root}",
            f"total={len(report.violations)}",
            f"counts={counts}",
            report.format(limit=50),
        ]
    )


def _is_artifact_write_allowed(rel_path: Path) -> bool:
    if rel_path.parts[:1] == ("tests",):
        return True
    return any(_contains_parts(rel_path, allowed) for allowed in ARTIFACT_WRITE_ALLOWED_PATH_PARTS)


def _is_under(rel_path: Path, prefix: str) -> bool:
    prefix_parts = Path(prefix).parts
    return rel_path.parts[: len(prefix_parts)] == prefix_parts


def _contains_parts(rel_path: Path, parts: Sequence[str]) -> bool:
    path_parts = rel_path.parts
    if len(parts) > len(path_parts):
        return False
    return any(
        tuple(path_parts[index : index + len(parts)]) == tuple(parts)
        for index in range(len(path_parts) - len(parts) + 1)
    )


def _annotation_mentions_base_environment(
    annotation: ast.AST, *, aliases: set[str]
) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id in aliases
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "BaseEnvironment" or annotation.attr in aliases
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        tokens = set(annotation.value.replace("[", " ").replace("]", " ").replace("|", " ").split())
        return bool(tokens & aliases) or "BaseEnvironment" in tokens
    for child in ast.iter_child_nodes(annotation):
        if _annotation_mentions_base_environment(child, aliases=aliases):
            return True
    return False


def _is_environment_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "environment"


def _expr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expr_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Tuple):
        names = [_expr_name(element) for element in node.elts]
        return ",".join(name for name in names if name)
    return None


def _looks_like_provider_or_environment_subclass(type_name: str) -> bool:
    if type_name == "BaseEnvironment" or type_name.endswith(".BaseEnvironment"):
        return False
    return any(token in type_name for token in ("Provider", "Environment", "Sandbox"))


def _literal_arg(node: ast.Call, index: int) -> str | None:
    if len(node.args) <= index:
        return None
    value = node.args[index]
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _literal_kwarg(node: ast.Call, name: str) -> str | None:
    for keyword in node.keywords:
        if keyword.arg != name:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    return None


def _is_write_mode(mode: str | None) -> bool:
    if mode is None:
        return False
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _is_module_open_call(func: ast.Attribute) -> bool:
    return isinstance(func.value, ast.Name) and func.value.id in MODULE_OPEN_ALLOWLIST


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")


def _rules(report: BoundaryReport) -> set[str]:
    return {violation.rule for violation in report.violations}


def _load_legacy_sweep_builder():
    script_path = Path(__file__).resolve().parents[3] / "scripts/agentkit_v2_legacy_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "_agentkit_v2_legacy_sweep_boundary_test", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_legacy_sweep_report
