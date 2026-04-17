from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import run as gage_run


@pytest.mark.io
def test_show_expanded_config_prints_yaml_without_runtime_preflight(monkeypatch, capsys) -> None:
    cfg = ROOT / "tests" / "fixtures" / "static_eval" / "aime24_short.yaml"
    monkeypatch.setattr(
        gage_run,
        "_ensure_spawn_start_method",
        lambda: (_ for _ in ()).throw(AssertionError("runtime preflight should not run")),
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(cfg), "--show-expanded-config"])

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    expanded = yaml.safe_load(captured.out)
    assert excinfo.value.code == 0
    assert "scene" not in expanded
    assert expanded["role_adapters"][0]["adapter_id"] == "dut_openai"
    assert expanded["tasks"][0]["task_id"] == "aime24"
    assert expanded["tasks"][0]["steps"][0] == {"step": "inference", "adapter_id": "dut_openai"}
    for key in (
        "models",
        "agent_backends",
        "sandbox_profiles",
        "mcp_clients",
        "prompts",
        "summary_generators",
    ):
        assert key not in expanded


@pytest.mark.io
def test_show_expanded_config_preserves_non_empty_optional_sections(monkeypatch, capsys, tmp_path) -> None:
    cfg = tmp_path / "with_prompt.yaml"
    cfg.write_text(
        """
api_version: gage/v1alpha1
kind: PipelineConfig
scene: static
metadata: { name: with_prompt }
datasets:
  - dataset_id: ds
    loader: jsonl
    params: { path: dummy.jsonl }
backends:
  - backend_id: openai
    type: litellm
    config: { provider: openai, model: gpt-4.1 }
prompts:
  - prompt_id: p1
    renderer: jinja
    template: "Answer directly."
role_adapters:
  - adapter_id: dut_openai
    role_type: dut_model
    backend_id: openai
    prompt_id: p1
metrics: [exact_match]
task: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(cfg), "--show-expanded-config"])

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    expanded = yaml.safe_load(captured.out)
    assert excinfo.value.code == 0
    assert expanded["prompts"] == [
        {"prompt_id": "p1", "renderer": "jinja", "template": "Answer directly."}
    ]


@pytest.mark.io
def test_show_expanded_config_honors_backend_id(monkeypatch, capsys, tmp_path) -> None:
    cfg = tmp_path / "dual.yaml"
    cfg.write_text(
        """
api_version: gage/v1alpha1
kind: PipelineConfig
scene: static
metadata: { name: dual }
datasets:
  - dataset_id: ds
    loader: jsonl
    params: { path: dummy.jsonl }
backends:
  - backend_id: openai
    type: litellm
    config: { provider: openai, model: gpt-4.1 }
  - backend_id: vllm_qwen
    type: vllm
    config: { model_path: /models/qwen }
metrics: [exact_match]
task:
  backend: openai
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--config", str(cfg), "--show-expanded-config", "--backend-id", "vllm_qwen"],
    )

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    expanded = yaml.safe_load(captured.out)
    assert excinfo.value.code == 0
    assert expanded["tasks"][0]["steps"][0] == {"step": "inference", "adapter_id": "dut_vllm_qwen"}


@pytest.mark.io
def test_show_expanded_config_compiles_runconfig_before_materializing(monkeypatch, capsys, tmp_path) -> None:
    cfg = tmp_path / "run_config.yaml"
    cfg.write_text(
        """
api_version: gage/v1alpha1
kind: RunConfig
metadata: { name: static_run }
""",
        encoding="utf-8",
    )
    compiled = {
        "api_version": "gage/v1alpha1",
        "kind": "PipelineConfig",
        "scene": "static",
        "metadata": {"name": "compiled_static"},
        "datasets": [{"dataset_id": "ds", "loader": "jsonl", "params": {"path": "dummy.jsonl"}}],
        "backends": [
            {
                "backend_id": "openai",
                "type": "litellm",
                "config": {"provider": "openai", "model": "gpt-4.1"},
            }
        ],
        "metrics": ["exact_match"],
        "task": {},
    }

    monkeypatch.setattr(gage_run, "_compile_run_config", lambda payload: (compiled, tmp_path / "template.yaml"))
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(cfg), "--show-expanded-config"])

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    expanded = yaml.safe_load(captured.out)
    assert excinfo.value.code == 0
    assert expanded["tasks"][0]["task_id"] == "compiled_static"
    assert expanded["tasks"][0]["steps"][0] == {"step": "inference", "adapter_id": "dut_openai"}


@pytest.mark.io
def test_show_expanded_config_reports_runconfig_compile_error(monkeypatch, capsys, tmp_path) -> None:
    cfg = tmp_path / "run_config.yaml"
    cfg.write_text(
        "api_version: gage/v1alpha1\nkind: RunConfig\nmetadata: { name: static_run }\n",
        encoding="utf-8",
    )

    def fail_compile(payload):
        raise ValueError("compile failed")

    monkeypatch.setattr(gage_run, "_compile_run_config", fail_compile)
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(cfg), "--show-expanded-config"])

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert "compile failed" in captured.err


@pytest.mark.io
def test_no_smart_defaults_prints_pre_smart_payload_for_short_static_config(monkeypatch, capsys) -> None:
    cfg = ROOT / "tests" / "fixtures" / "static_eval" / "aime24_short.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--config", str(cfg), "--show-expanded-config", "--no-smart-defaults"],
    )

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    expanded = yaml.safe_load(captured.out)
    assert excinfo.value.code == 0
    assert expanded["scene"] == "static"
    assert expanded["datasets"][0]["hub_id"] == "Maxwell-Jia/AIME_2024"
    assert "task" in expanded
    assert "role_adapters" not in expanded
    assert "custom" not in expanded


@pytest.mark.io
def test_max_samples_zero_uses_materialized_static_payload(monkeypatch, capsys) -> None:
    cfg = ROOT / "tests" / "fixtures" / "static_eval" / "aime24_short.yaml"
    seen: dict[str, list[str]] = {}

    monkeypatch.delenv("GAGE_EVAL_THREADS", raising=False)
    monkeypatch.delenv("GAGE_EVAL_MAX_SAMPLES", raising=False)
    monkeypatch.setattr(gage_run, "_ensure_spawn_start_method", lambda: None)
    monkeypatch.setattr(gage_run, "_ensure_default_concurrency", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_detect_hardware_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_apply_hardware_profile_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_preflight_checks", lambda: None)
    monkeypatch.setattr(gage_run, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(gage_run, "build_default_registry", lambda: "registry")

    def fake_validate(config, **kwargs):
        seen["tasks"] = [task.task_id for task in config.tasks]
        seen["adapters"] = [adapter.adapter_id for adapter in config.role_adapters]
        return []

    monkeypatch.setattr(gage_run, "_validate_config_wiring", fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--config", str(cfg), "--max-samples", "0", "--gpus", "0", "--cpus", "1"],
    )

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert seen == {"tasks": ["aime24"], "adapters": ["dut_openai"]}
    assert "config wiring validated" in captured.out


@pytest.mark.io
def test_negative_max_samples_exits_before_config_materialization(monkeypatch, capsys) -> None:
    cfg = ROOT / "tests" / "fixtures" / "static_eval" / "aime24_short.yaml"

    monkeypatch.setattr(gage_run, "_ensure_spawn_start_method", lambda: None)
    monkeypatch.setattr(gage_run, "_detect_hardware_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_ensure_default_concurrency", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_apply_hardware_profile_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_preflight_checks", lambda: None)
    monkeypatch.setattr(gage_run, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(
        gage_run,
        "load_pipeline_config_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("config should not be materialized")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--config", str(cfg), "--max-samples", "-1", "--gpus", "0", "--cpus", "1"],
    )

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert "--max-samples must be >= 0" in captured.err
