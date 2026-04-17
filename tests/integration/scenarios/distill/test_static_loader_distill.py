from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gage_eval.config.loader import load_pipeline_config_payload
from gage_eval.tools.distill import analyze_tasks_for_distill

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


@pytest.mark.io
def test_distill_analysis_accepts_expanded_short_static_config() -> None:
    path = ROOT / "tests" / "fixtures" / "static_eval" / "aime24_short.yaml"
    payload = load_pipeline_config_payload(path)

    analysis = analyze_tasks_for_distill(payload)

    assert analysis.mode == "ATOMIC"
    assert analysis.task_ids == ("aime24",)


@pytest.mark.io
def test_run_py_distill_uses_loader_for_short_static_config(monkeypatch, tmp_path, capsys) -> None:
    import run as gage_run

    path = ROOT / "tests" / "fixtures" / "static_eval" / "aime24_short.yaml"
    output_root = tmp_path / "templates"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--config",
            str(path),
            "--distill",
            "--builtin-name",
            "static_suite",
            "--distill-output",
            str(output_root),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert "template written" in captured.out
    assert (output_root / "static_suite" / "v1.yaml").exists()


@pytest.mark.io
def test_run_py_distill_compiles_runconfig_through_loader(monkeypatch, tmp_path, capsys) -> None:
    import run as gage_run

    run_config = tmp_path / "run_config.yaml"
    run_config.write_text("api_version: gage/v1alpha1\nkind: RunConfig\nmetadata: { name: rc }\n", encoding="utf-8")
    compiled = {
        "api_version": "gage/v1alpha1",
        "kind": "PipelineConfig",
        "scene": "static",
        "metadata": {"name": "compiled_static"},
        "datasets": [{"dataset_id": "ds", "loader": "jsonl", "params": {"path": "dummy.jsonl"}}],
        "backends": [{"backend_id": "openai", "type": "litellm", "config": {"provider": "openai", "model": "gpt-4.1"}}],
        "metrics": ["exact_match"],
        "task": {},
    }
    output_root = tmp_path / "templates"
    monkeypatch.setattr(gage_run, "_compile_run_config", lambda payload: (compiled, tmp_path / "template.yaml"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--config",
            str(run_config),
            "--distill",
            "--builtin-name",
            "runconfig_suite",
            "--distill-output",
            str(output_root),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert "template written" in captured.out
    assert (output_root / "runconfig_suite" / "v1.yaml").exists()
