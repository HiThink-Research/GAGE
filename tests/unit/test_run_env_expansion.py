from pathlib import Path
import importlib.util

import pytest


_RUN_PATH = Path(__file__).resolve().parents[2] / "run.py"
_SPEC = importlib.util.spec_from_file_location("gage_eval_run_cli", _RUN_PATH)
assert _SPEC and _SPEC.loader
run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run)


def test_expand_env_required_placeholder_uses_env_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    expanded = run._expand_env({"api_key": "${OPENAI_API_KEY:?set OPENAI_API_KEY}"})

    assert expanded == {"api_key": "sk-test"}


def test_expand_env_required_placeholder_reports_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="set OPENAI_API_KEY"):
        run._expand_env({"api_key": "${OPENAI_API_KEY:?set OPENAI_API_KEY}"})


def test_apply_cli_metric_filter_preserves_malformed_entries_for_schema_validation():
    payload = {"metrics": ["keep", 123]}

    run._apply_cli_metric_filter(payload, "keep")

    assert payload["metrics"] == ["keep", 123]


def test_apply_cli_skip_judge_removes_custom_and_task_steps():
    payload = {
        "custom": {"steps": [{"step": "inference"}, {"step": "judge"}, {"step": "auto_eval"}]},
        "tasks": [
            {
                "task_id": "t1",
                "dataset_id": "ds",
                "steps": [{"step": "inference"}, {"step": "judge"}],
            }
        ],
    }

    run._apply_cli_skip_judge(payload)

    assert payload["custom"]["steps"] == [{"step": "inference"}, {"step": "auto_eval"}]
    assert payload["tasks"][0]["steps"] == [{"step": "inference"}]
