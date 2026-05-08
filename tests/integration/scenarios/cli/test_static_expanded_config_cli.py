from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import run as gage_run
import gage_eval.config.agentkit_v2 as agentkit_v2_module


AGENTKIT_V2_CONFIG = """
kind: AgentEvalConfig
metadata:
  name: agentkit-v2-cli
backends:
  - backend_id: model
    type: litellm
    config:
      model: gpt-4.1-mini
      api_key: ${ENV.MODEL_API_KEY}
agents:
  - agent_id: agent
    scheduler:
      type: framework_loop
      backend_id: model
benchmarks:
  - benchmark_id: bench
    kit_id: tau2
    config: {}
environments:
  - env_id: env
    provider: local_process
    profile_id: tau2_local
    profile:
      asset_dir: tests/fixtures/agentkit_v2/tau2
dut_agents:
  - dut_id: dut
    agent_id: agent
    env_id: env
    benchmark_id: bench
"""

AGENTKIT_V2_DUMMY_BACKEND_CONFIG = AGENTKIT_V2_CONFIG.replace(
    "type: litellm\n    config:\n      model: gpt-4.1-mini\n      api_key: ${ENV.MODEL_API_KEY}",
    "type: dummy\n    config:\n      response: ok",
)


class _InteractiveStdin:
    def isatty(self) -> bool:
        return True


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
def test_build_cli_intent_populates_agentkit_v2_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--config",
            "config.yaml",
            "--env-provider",
            "docker",
            "--dut-id",
            "dut",
            "--env-id",
            "env",
        ],
    )

    args = gage_run.parse_args()
    intent = gage_run._build_cli_intent(args)

    assert intent.env_provider == "docker"
    assert intent.dut_id == "dut"
    assert intent.env_id == "env"


@pytest.mark.io
def test_show_expanded_agentkit_v2_config_redacts_env_secret(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "secret-value")
    cfg = tmp_path / "agentkit_v2.yaml"
    cfg.write_text(AGENTKIT_V2_CONFIG, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(cfg), "--show-expanded-config"])

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert "secret-value" not in captured.out
    assert "<redacted:reference:ENV.MODEL_API_KEY>" in captured.out
    expanded = yaml.safe_load(captured.out)
    assert expanded["backends"][0]["config"]["api_key"] == "<redacted:reference:ENV.MODEL_API_KEY>"


@pytest.mark.io
def test_show_expanded_agentkit_v2_config_redacts_bare_env_secret(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "bare-secret-value")
    cfg = tmp_path / "agentkit_v2_bare_env.yaml"
    cfg.write_text(AGENTKIT_V2_CONFIG.replace("${ENV.MODEL_API_KEY}", "${MODEL_API_KEY}"), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(cfg), "--show-expanded-config"])

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert "bare-secret-value" not in captured.out
    assert "<redacted:reference:MODEL_API_KEY>" in captured.out
    expanded = yaml.safe_load(captured.out)
    assert expanded["backends"][0]["config"]["api_key"] == "<redacted:reference:MODEL_API_KEY>"


@pytest.mark.io
def test_show_expanded_no_smart_defaults_agentkit_v2_uses_redacted_materializer(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "bare-secret-value")
    cfg = tmp_path / "agentkit_v2_no_smart.yaml"
    cfg.write_text(AGENTKIT_V2_CONFIG.replace("${ENV.MODEL_API_KEY}", "${MODEL_API_KEY}"), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--config",
            str(cfg),
            "--show-expanded-config",
            "--no-smart-defaults",
            "--env-provider",
            "docker",
            "--dut-id",
            "dut",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert "bare-secret-value" not in captured.out
    assert "<redacted:reference:MODEL_API_KEY>" in captured.out
    expanded = yaml.safe_load(captured.out)
    assert expanded["backends"][0]["config"]["api_key"] == "<redacted:reference:MODEL_API_KEY>"
    assert expanded["environments"][0]["provider"] == "docker"
    assert expanded["effective_config"]["backends"][0]["config"]["api_key"] == (
        "<redacted:reference:MODEL_API_KEY>"
    )
    assert expanded["effective_config"]["environments"][0]["provider"] == "docker"


@pytest.mark.io
def test_show_expanded_agentkit_v2_validation_error_redacts_bare_env_secret(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "validation-secret-value")
    cfg = tmp_path / "agentkit_v2_invalid_provider.yaml"
    cfg.write_text(
        AGENTKIT_V2_CONFIG.replace("provider: local_process", "provider: ${MODEL_API_KEY}"),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(cfg), "--show-expanded-config"])

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert "validation-secret-value" not in captured.err
    assert "<redacted:reference:MODEL_API_KEY>" in captured.err
    assert "environments.0.provider" in captured.err


@pytest.mark.io
@pytest.mark.parametrize("extra_args", [[], ["--max-samples", "0"]])
def test_agentkit_v2_run_mode_fails_fast_before_legacy_pipeline_validation(
    monkeypatch,
    capsys,
    tmp_path,
    extra_args,
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "secret-value")
    cfg = tmp_path / "agentkit_v2_run.yaml"
    cfg.write_text(AGENTKIT_V2_CONFIG, encoding="utf-8")
    monkeypatch.setattr(gage_run, "_ensure_spawn_start_method", lambda: None)
    monkeypatch.setattr(gage_run, "_ensure_default_concurrency", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_detect_hardware_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_apply_hardware_profile_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_preflight_checks", lambda: None)
    monkeypatch.setattr(gage_run, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(
        gage_run,
        "_validate_config_wiring",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy wiring validation should not run")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--config", str(cfg), "--gpus", "0", "--cpus", "1", *extra_args],
    )

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert "AgentEvalConfig execution is not wired yet" in captured.err
    assert "--show-expanded-config" in captured.err
    assert "dataset" not in captured.err.lower()
    assert "role adapter" not in captured.err.lower()
    assert "custom/builtin" not in captured.err.lower()


@pytest.mark.io
def test_agentkit_v2_run_mode_validates_binding_specs_without_backend_construction(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    cfg = tmp_path / "agentkit_v2_run_dummy.yaml"
    cfg.write_text(AGENTKIT_V2_DUMMY_BACKEND_CONFIG, encoding="utf-8")
    captured_specs = {}
    original_resolve_specs = agentkit_v2_module.resolve_agentkit_v2_runtime_binding_specs

    def _capture_specs(*args, **kwargs):
        specs = original_resolve_specs(*args, **kwargs)
        captured_specs.update(specs)
        return specs

    monkeypatch.setattr(agentkit_v2_module, "resolve_agentkit_v2_runtime_binding_specs", _capture_specs)
    monkeypatch.setattr(
        "gage_eval.role.model.backends.builder.build_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backend construction should not run")),
    )
    monkeypatch.setattr(gage_run, "_ensure_spawn_start_method", lambda: None)
    monkeypatch.setattr(gage_run, "_ensure_default_concurrency", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_detect_hardware_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_apply_hardware_profile_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(gage_run, "_preflight_checks", lambda: None)
    monkeypatch.setattr(gage_run, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--config", str(cfg), "--gpus", "0", "--cpus", "1"],
    )

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert "AgentEvalConfig execution is not wired yet" in captured.err
    assert captured_specs["dut"].backend_id == "model"
    assert captured_specs["dut"].environment_provider == "local_process"


@pytest.mark.io
def test_agentkit_v2_run_mode_skips_runtime_preflight_side_effects(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.delenv("GAGE_EVAL_MAX_SAMPLES", raising=False)
    monkeypatch.delenv("VLLM_NATIVE_MODEL_PATH", raising=False)
    monkeypatch.delenv("GAGE_EVAL_HUMAN_INPUT", raising=False)
    monkeypatch.setattr(sys, "stdin", _InteractiveStdin())
    cfg = tmp_path / "agentkit_v2_run_no_side_effects.yaml"
    cfg.write_text(AGENTKIT_V2_DUMMY_BACKEND_CONFIG, encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(gage_run, "_ensure_spawn_start_method", lambda: calls.append("spawn"))
    monkeypatch.setattr(gage_run, "_detect_hardware_profile", lambda *args, **kwargs: calls.append("detect"))
    monkeypatch.setattr(gage_run, "_ensure_default_concurrency", lambda *args, **kwargs: calls.append("concurrency"))
    monkeypatch.setattr(gage_run, "_apply_hardware_profile_env", lambda *args, **kwargs: calls.append("hardware_env"))
    monkeypatch.setattr(gage_run, "_preflight_checks", lambda: calls.append("preflight"))
    monkeypatch.setattr(gage_run, "_install_signal_handlers", lambda: calls.append("signals"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--config",
            str(cfg),
            "--gpus",
            "0",
            "--cpus",
            "1",
            "--max-samples",
            "0",
            "--model-path",
            "/tmp/model",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert "AgentEvalConfig execution is not wired yet" in captured.err
    assert calls == []
    assert "GAGE_EVAL_MAX_SAMPLES" not in os.environ
    assert "VLLM_NATIVE_MODEL_PATH" not in os.environ
    assert "GAGE_EVAL_HUMAN_INPUT" not in os.environ


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
