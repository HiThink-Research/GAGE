from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.config import build_default_registry
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.registry import registry
from gage_eval.role.resource_profile import NodeResource, ResourceProfile


class _DummySwebenchJudge:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def invoke(self, payload: dict, _state=None) -> dict:
        sample = payload.get("sample") or {}
        metadata = sample.get("metadata") or {}
        resolved = bool(metadata.get("expected_resolved"))
        failure_reason = metadata.get("failure_reason")
        model_output = payload.get("model_output") or {}
        patch = _extract_patch(model_output)
        if not patch and resolved:
            resolved = False
            failure_reason = failure_reason or "missing_patch"
        if not resolved and not failure_reason:
            failure_reason = "tests_failed"
        return {
            "resolved": resolved,
            "failure_reason": None if resolved else failure_reason,
            "observed_patch": patch,
        }


def _register_dummy_judge() -> None:
    registry.register(
        "judge_impls",
        "test_swebench_judge",
        _DummySwebenchJudge,
        desc="dummy swebench judge for integration tests",
    )


def _extract_patch(model_output: dict) -> str:
    for key in ("patch", "diff", "answer", "text", "content"):
        value = model_output.get(key)
        if isinstance(value, str) and value.strip():
            return value
    message = model_output.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def _make_repo(root: Path) -> None:
    (root / "app").mkdir(parents=True, exist_ok=True)
    (root / "app" / "main.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (root / "app" / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "README.md").write_text("Example repo\n", encoding="utf-8")


def _build_pipeline_config(jsonl_path: Path, repo_root: Path) -> PipelineConfig:
    patch_a = "diff --git a/app/main.py b/app/main.py\n+print('patched')\n"
    patch_b = "diff --git a/app/util.py b/app/util.py\n+VALUE = 2\n"
    config_dict = {
        "api_version": "gage/v1alpha1",
        "kind": "PipelineConfig",
        "metadata": {"name": "swebench_integration"},
        "custom": {
            "steps": [
                {"step": "support", "adapter_id": "swebench_context"},
                {"step": "inference", "adapter_id": "swebench_dut"},
                {"step": "judge", "adapter_id": "swebench_judge"},
                {"step": "auto_eval"},
            ]
        },
        "datasets": [
            {
                "dataset_id": "swebench_smoke",
                "loader": "jsonl",
                "params": {"path": str(jsonl_path), "streaming": False},
            }
        ],
        "backends": [
            {
                "backend_id": "dummy_backend",
                "type": "dummy",
                "config": {"responses": [patch_a, patch_b], "cycle": True},
            }
        ],
        "role_adapters": [
            {
                "adapter_id": "swebench_context",
                "role_type": "context_provider",
                "params": {
                    "implementation": "swebench_repo",
                    "implementation_params": {
                        "repo_source": "local_path",
                        "topk_files": 5,
                        "max_tree_depth": 2,
                        "max_tree_lines": 50,
                        "max_file_lines": 50,
                    },
                },
            },
            {
                "adapter_id": "swebench_dut",
                "role_type": "dut_model",
                "backend_id": "dummy_backend",
                "capabilities": ["chat_completion"],
            },
            {
                "adapter_id": "swebench_judge",
                "role_type": "judge_extend",
                "params": {"implementation": "test_swebench_judge"},
            },
        ],
        "metrics": [
            {"metric_id": "swebench_resolve_rate", "implementation": "swebench_resolve_rate"},
            {
                "metric_id": "swebench_failure_reason",
                "implementation": "swebench_failure_reason",
                "aggregation": "categorical_count",
            },
        ],
        "tasks": [
            {
                "task_id": "swebench_integration",
                "dataset_id": "swebench_smoke",
                "max_samples": 2,
                "concurrency": 1,
            }
        ],
    }
    return PipelineConfig.from_dict(config_dict)


def _load_samples(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _concat_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for frag in content:
            if isinstance(frag, dict):
                text = frag.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return ""


def _run_pipeline(run_dir: Path, jsonl_path: Path, repo_root: Path, trace):
    _register_dummy_judge()
    config = _build_pipeline_config(jsonl_path, repo_root)
    registry_obj = build_default_registry()
    profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=0, cpus=1)])
    runtime = build_runtime(config=config, registry=registry_obj, resource_profile=profile, trace=trace)
    runtime.run()
    summary_path = run_dir / trace.run_id / "summary.json"
    samples_path = run_dir / trace.run_id / "samples.jsonl"
    return summary_path, samples_path


@pytest.mark.io
def test_swebench_pipeline_support_and_judge(temp_workspace: Path, mock_trace):
    repo_root = temp_workspace / "repo"
    _make_repo(repo_root)

    jsonl_path = temp_workspace / "samples.jsonl"
    records = [
        {
            "id": "instance_1",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Fix bug in app/main.py"}],
                }
            ],
            "metadata": {
                "repo": "org/repo-a",
                "repo_language": "python",
                "selected_test_files_to_run": ["app/main.py"],
                "expected_resolved": True,
                "local_path": str(repo_root),
            },
        },
        {
            "id": "instance_2",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Update app/util.py"}],
                }
            ],
            "metadata": {
                "repo": "org/repo-b",
                "repo_language": "python",
                "selected_test_files_to_run": ["app/util.py"],
                "expected_resolved": False,
                "failure_reason": "tests_failed",
                "local_path": str(repo_root),
            },
        },
    ]
    _write_jsonl(jsonl_path, records)

    summary_path, samples_path = _run_pipeline(temp_workspace, jsonl_path, repo_root, mock_trace)

    assert summary_path.exists()
    assert samples_path.exists()

    cached = _load_samples(samples_path)
    assert len(cached) == 2
    sample_a = next(item for item in cached if item["sample"]["id"] == "instance_1")
    sample_b = next(item for item in cached if item["sample"]["id"] == "instance_2")

    support_outputs = sample_a["sample"].get("support_outputs")
    assert support_outputs and support_outputs[0].get("repo_tree")

    messages = sample_a["sample"]["messages"]
    last_content = messages[-1]["content"]
    text = _concat_text(last_content)
    assert "Repository Tree:" in text
    assert "app/main.py" in text

    assert sample_a["model_output"]["answer"].startswith("diff --git")
    assert sample_a["judge_output"]["resolved"] is True
    assert sample_b["judge_output"]["resolved"] is False
    assert sample_b["judge_output"]["failure_reason"] == "tests_failed"


@pytest.mark.io
def test_swebench_pipeline_report_summary(temp_workspace: Path, mock_trace):
    repo_root = temp_workspace / "repo"
    _make_repo(repo_root)

    jsonl_path = temp_workspace / "samples.jsonl"
    records = [
        {
            "id": "instance_1",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Fix bug in app/main.py"}],
                }
            ],
            "metadata": {
                "repo": "org/repo-a",
                "repo_language": "python",
                "selected_test_files_to_run": ["app/main.py"],
                "expected_resolved": True,
                "local_path": str(repo_root),
            },
        },
        {
            "id": "instance_2",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Update app/util.py"}],
                }
            ],
            "metadata": {
                "repo": "org/repo-b",
                "repo_language": "python",
                "selected_test_files_to_run": ["app/util.py"],
                "expected_resolved": False,
                "failure_reason": "tests_failed",
                "local_path": str(repo_root),
            },
        },
    ]
    _write_jsonl(jsonl_path, records)

    summary_path, _ = _run_pipeline(temp_workspace, jsonl_path, repo_root, mock_trace)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    swebench = summary["swebench_summary"]
    assert swebench["overall"]["total"] == 2
    assert swebench["overall"]["resolved"] == 1
    assert swebench["overall"]["resolve_rate"] == 0.5
    assert swebench["failure_reason"]["tests_failed"] == 1

    metric_ids = {entry["metric_id"] for entry in summary["metrics"]}
    assert "swebench_resolve_rate" in metric_ids
    assert "swebench_failure_reason" in metric_ids
