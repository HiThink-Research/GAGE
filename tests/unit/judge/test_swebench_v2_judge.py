from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from gage_eval.agent_eval_kits.swebench.judge.failure_categories import (
    resolve_swebench_failure_category,
)
from gage_eval.agent_eval_kits.swebench import artifacts
from gage_eval.agent_eval_kits.swebench.judge import adapters, executor, patch_extraction, scoring
from gage_eval.agent_eval_kits.swebench.kit import load_kit
from gage_eval.agent_runtime import executor as runtime_executor


MIGRATED_FUNCTIONS = {
    "_run_with_sandbox",
    "_run_direct_docker",
    "_resolve_patch",
    "_resolve_patch_from_agent_trace",
    "_extract_tool_stdout",
    "_load_submission_patch",
    "_read_submission_patch",
    "_emit_patch_fallback_event",
    "_clean_patch_content",
    "_extract_code_block",
    "_strip_apply_patch_markers",
    "_strip_binary_hunks",
    "_normalize_hunk_context_lines",
    "_has_diff_markers",
    "_trim_diff_tail",
    "_create_entryscript",
    "_extract_env_exports",
    "_build_write_command",
    "_load_output_json",
    "_evaluate_resolution",
    "_parse_list",
    "_resolve_instance_id",
    "_resolve_image_uri",
    "_get_meta",
    "_resolve_run_id",
    "_host_log_dir",
    "_get_docker_client",
    "_has_image",
    "_safe_lock_name",
    "_resolve_run_scripts_mount",
    "_release_sandbox_provider",
    "_exec_sandbox_command",
    "_SandboxFallback",
    "_is_missing_file_error",
    "_read_text",
    "_write_text",
    "_build_swebench_diagnostics",
    "resolve_swebench_failure_category",
    "_resolve_prompt_metadata",
    "_collect_trace_stats",
    "_estimate_tool_output_bytes",
    "_extract_command_preview",
    "_count_artifact_spillovers",
    "_resolve_trace_failure_category",
    "_is_tool_argument_invalid_output",
    "_read_submission_patch_from_sandbox",
    "_read_git_diff_from_sandbox",
}


def test_function_migration_matrix_covers_design_3_7_2_list() -> None:
    matrix_path = Path("tests/fixtures/swebench_v2/function_migration_matrix.yaml")
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))

    assert set(payload["functions"]) == MIGRATED_FUNCTIONS
    for function_name, row in payload["functions"].items():
        assert row["v2_owner"]
        assert row["requirement"]
        assert row["covered_by"], function_name


def test_failure_categories_cover_swebench_dind_regressions() -> None:
    cases = {
        "missing_patch": {"failure_code": "artifact_capture.patch_missing"},
        "missing_metadata": {"failure_reason": "missing_metadata"},
        "missing_run_scripts": {"failure_reason": "missing_run_scripts"},
        "missing_output": {"failure_reason": "missing_output"},
        "invalid_output": {"failure_reason": "invalid_output"},
        "sandbox_judge_error": {"failure_code": "environment.unavailable"},
        "tool_protocol_error": {"failure_code": "client_execution.tool_protocol_parse_error"},
        "parse_error": {"failure_code": "client_execution.tool_argument_invalid"},
        "test_execution_error": {"failure_reason": "test_execution_error"},
        "context_overflow_from_listing": {
            "failure_code": "client_execution.run_scheduler.swebench.agent_loop_failed",
            "loop_exit_reason": "max_turns",
            "artifact_spillover_count": 4,
            "max_tool_output_bytes": 140000,
        },
        "endless_file_reading": {"repeated_command_count": 12, "loop_exit_reason": "max_turns"},
        "syntax_error": {
            "failure_code": "client_execution.tool_argument_invalid",
            "parse_error_count": 5,
            "recent_errors": ["SyntaxError: invalid syntax"],
        },
    }

    for expected, payload in cases.items():
        assert resolve_swebench_failure_category(payload) == expected


def test_kit_owned_judge_modules_do_not_depend_on_legacy_swebench_docker() -> None:
    for module in (adapters, executor, patch_extraction, scoring, artifacts):
        source = inspect.getsource(module)
        legacy_module = ".".join(("gage_eval", "role", "judge", "swebench_docker"))
        assert legacy_module not in source
        assert "SwebenchDocker" not in source
        assert "get_handle" not in source


def test_swebench_fresh_verifier_uses_environment_manager_acquire() -> None:
    kit = load_kit()
    adapter_source = inspect.getsource(adapters.SwebenchVerifierAdapter)
    runtime_source = inspect.getsource(runtime_executor.CompiledRuntimeExecutor._aexecute_multi_trial)

    assert kit.verifier_environment_policy == "fresh_from_profile"
    assert "docker.from_env" not in adapter_source
    assert "e2b" not in adapter_source.lower()
    assert 'verifier_environment_policy == "fresh_from_profile"' in runtime_source
    assert "self.resource_manager.acquire" in runtime_source
