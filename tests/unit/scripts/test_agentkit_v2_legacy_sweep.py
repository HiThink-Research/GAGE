from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent


def _load_legacy_sweep_builder():
    script_path = Path(__file__).resolve().parents[3] / "scripts/agentkit_v2_legacy_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "_agentkit_v2_legacy_sweep_under_test", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_legacy_sweep_report


build_legacy_sweep_report = _load_legacy_sweep_builder()
LEGACY_SANDBOX_MODULE = ".".join(("gage_eval", "sandbox"))
LEGACY_ROLE_AGENT_BACKENDS_MODULE = ".".join(("gage_eval", "role", "agent", "backends"))
LEGACY_TAU2_JUDGE_MODULE = ".".join(("gage_eval", "role", "judge", "tau2_eval"))


def test_legacy_sweep_reports_blockers_without_flagging_static_role_assets(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gage_eval/agent_runtime/executor.py",
        f"""
        from {LEGACY_SANDBOX_MODULE}.provider import SandboxProvider
        """,
    )
    _write(
        tmp_path / "src/gage_eval/config/registry.py",
        f"""
        from {LEGACY_ROLE_AGENT_BACKENDS_MODULE} import build_agent_backend
        from {LEGACY_TAU2_JUDGE_MODULE} import Tau2Evaluate
        """,
    )
    _write(
        tmp_path / "src/gage_eval/game_eval_kits/legacy.py",
        """
        LEGACY = True
        """,
    )
    _write(
        tmp_path / "src/gage_eval/role/judge/tau2_eval.py",
        """
        class Tau2Evaluate:
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
        tmp_path / "src/gage_eval/role/model/backends/dummy_backend.py",
        """
        class DummyBackend:
            pass
        """,
    )
    _write(
        tmp_path / "src/gage_eval/utils/benchmark_helpers/tau2.py",
        """
        def helper():
            pass
        """,
    )
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/tau2/kit.py",
        """
        from gage_eval.agent_eval_kits.common import BenchmarkKitEntry

        def load_kit():
            return BenchmarkKitEntry(
                kit_id="tau2",
                config_schema=object,
                supported_schedulers=("framework_loop",),
                workflow_resolver=lambda scheduler_type: object(),
            )
        """,
    )
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/swebench/judge_bridge.py",
        """
        LEGACY = True
        """,
    )
    _write(
        tmp_path / "src/gage_eval/agent_eval_kits/swebench/kit.py",
        """
        from gage_eval.agent_eval_kits.swebench.judge_bridge import resolve_verifier_resources
        from gage_eval.utils.benchmark_helpers.swebench import get_dockerhub_image_uri
        """,
    )

    report = build_legacy_sweep_report(tmp_path)

    counts = report.counts_by_category()
    assert counts["sandbox_import_reference"] == 1
    assert counts["role_agent_backends_reference"] == 1
    assert counts["agent_legacy_judge_import"] == 1
    assert counts["agent_legacy_judge_module"] == 1
    assert counts["legacy_game_eval_kits_package"] == 1
    assert counts["utils_benchmark_helpers_package"] == 1
    assert counts["utils_benchmark_helpers_reference"] == 1
    assert counts["transition_shim_module"] == 1
    assert counts["transition_shim_import"] == 1
    assert counts["benchmark_kit_entry_missing_v2_keywords"] == 1
    assert "src/gage_eval/role/model/backends" in report.static_retained
    assert "src/gage_eval/role/judge/gomoku_referee.py" in report.static_retained
    assert not any("role/model/backends" in finding.path for finding in report.findings)
    assert not any("gomoku_referee.py" == Path(finding.path).name for finding in report.findings)


def test_legacy_sweep_markdown_contains_required_sections(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gage_eval/role/judge/swebench_docker.py",
        """
        class SwebenchDocker:
            pass
        """,
    )

    report = build_legacy_sweep_report(tmp_path)
    markdown = report.to_markdown()

    assert "# AgentKit v2 Legacy Sweep Report" in markdown
    assert "## Summary" in markdown
    assert "## Findings" in markdown
    assert "## Static Modules Intentionally Retained" in markdown
    assert "agent_legacy_judge_module" in markdown
    assert "swebench_docker.py" in markdown


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")
