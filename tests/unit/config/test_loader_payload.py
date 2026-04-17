from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from gage_eval.config.loader import (
    expand_env,
    load_pipeline_config_payload,
    load_pre_smart_defaults_payload,
    load_yaml_mapping,
    materialize_pipeline_config_payload,
)
from gage_eval.config.loader_cli import CLIIntent
from gage_eval.config.smart_defaults import SmartDefaultsError


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "static_eval"


def _assert_dataset_paths_exist(payload: dict[str, object]) -> None:
    datasets = payload.get("datasets")
    assert isinstance(datasets, list)
    for dataset in datasets:
        assert isinstance(dataset, dict)
        params = dataset.get("params")
        assert isinstance(params, dict)
        path = params.get("path")
        assert isinstance(path, str)
        assert (REPO_ROOT / path).exists()


@pytest.mark.fast
def test_expand_env_required_placeholder_uses_env_value(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    expanded = expand_env({"api_key": "${OPENAI_API_KEY:?set OPENAI_API_KEY}"})

    assert expanded == {"api_key": "sk-test"}


@pytest.mark.fast
def test_expand_env_required_placeholder_reports_message(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="set OPENAI_API_KEY"):
        expand_env({"api_key": "${OPENAI_API_KEY:?set OPENAI_API_KEY}"})


@pytest.mark.fast
def test_expand_env_keeps_embedded_placeholder_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("API_VERSION", "v2")

    expanded = expand_env({"url": "https://api.example.com/${API_VERSION:-v1}"})

    assert expanded == {"url": "https://api.example.com/${API_VERSION:-v1}"}


@pytest.mark.fast
def test_expand_env_expands_bare_placeholder(monkeypatch) -> None:
    monkeypatch.setenv("PLAIN_VAR", "plain-value")

    expanded = expand_env({"value": "${PLAIN_VAR}"})

    assert expanded == {"value": "plain-value"}


@pytest.mark.fast
def test_expand_env_keeps_padded_placeholder_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("X", "9")

    expanded = expand_env({"value": " ${X:-1} "})

    assert expanded == {"value": " ${X:-1} "}


@pytest.mark.io
def test_load_yaml_mapping_rejects_top_level_list(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping at the top level"):
        load_yaml_mapping(config_path)


@pytest.mark.fast
def test_materialize_legacy_payload_keeps_original_payload_unchanged() -> None:
    raw = {
        "kind": "PipelineConfig",
        "datasets": [{"dataset_id": "ds1", "loader": "jsonl", "params": {"path": "dummy.jsonl"}}],
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
    }
    before = deepcopy(raw)

    normalized = materialize_pipeline_config_payload(raw, source_path=None)

    assert raw == before
    assert normalized["datasets"][0]["dataset_id"] == "ds1"
    assert "scene" not in normalized


@pytest.mark.fast
def test_materialize_rejects_non_string_pipeline_scene() -> None:
    raw = {
        "kind": "PipelineConfig",
        "scene": 123,
        "datasets": [{"dataset_id": "ds1", "loader": "jsonl", "params": {"path": "dummy.jsonl"}}],
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
    }

    with pytest.raises(SmartDefaultsError, match="Unknown PipelineConfig scene '123'"):
        materialize_pipeline_config_payload(raw, source_path=None)


@pytest.mark.fast
def test_materialize_runconfig_payload_calls_compiler_and_normalizes_compiled_payload(monkeypatch) -> None:
    monkeypatch.setenv("RUN_NAME", "demo-run")

    payload = {
        "kind": "RunConfig",
        "metadata": {"name": "${RUN_NAME:?missing RUN_NAME}"},
    }
    calls: list[dict[str, object]] = []

    def compiler(expanded: dict[str, object]) -> tuple[dict[str, object], None]:
        calls.append(expanded)
        assert expanded["metadata"]["name"] == "demo-run"
        return (
            {
                "kind": "PipelineConfig",
                "datasets": [{"dataset_id": "ds1", "loader": "jsonl"}],
                "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
                "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
                "scene": "agent",
            },
            None,
        )

    normalized = materialize_pipeline_config_payload(payload, source_path=None, run_config_compiler=compiler)

    assert len(calls) == 1
    assert normalized["datasets"][0]["dataset_id"] == "ds1"
    assert "scene" not in normalized


@pytest.mark.fast
def test_materialize_runconfig_payload_raises_value_error_before_unbounded_recursion(tmp_path) -> None:
    payload = {"kind": "RunConfig"}
    source_path = tmp_path / "runconfig.yaml"

    def compiler(expanded: dict[str, object]) -> tuple[dict[str, object], None]:
        assert expanded == {"kind": "RunConfig"}
        return {"kind": "RunConfig"}, None

    with pytest.raises(ValueError, match=r"RunConfig materialization exceeded .*runconfig\.yaml"):
        materialize_pipeline_config_payload(payload, source_path=source_path, run_config_compiler=compiler)


@pytest.mark.fast
def test_load_pre_smart_defaults_payload_calls_compiler_and_preserves_scene(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RUN_NAME", "demo-run")

    config_path = tmp_path / "runconfig.yaml"
    config_path.write_text(
        "\n".join(
            [
                "kind: RunConfig",
                "metadata:",
                "  name: ${RUN_NAME:?missing RUN_NAME}",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def compiler(expanded: dict[str, object]) -> tuple[dict[str, object], None]:
        calls.append(expanded)
        assert expanded["metadata"]["name"] == "demo-run"
        return (
            {
                "kind": "PipelineConfig",
                "datasets": [{"dataset_id": "ds1", "loader": "jsonl"}],
                "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
                "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
                "scene": "static",
            },
            None,
        )

    compiled = load_pre_smart_defaults_payload(config_path, run_config_compiler=compiler)

    assert len(calls) == 1
    assert compiled["kind"] == "PipelineConfig"
    assert compiled["scene"] == "static"


@pytest.mark.fast
def test_load_pre_smart_defaults_payload_rejects_unknown_scene(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "kind: PipelineConfig",
                "scene: arena",
                "datasets:",
                "  - dataset_id: ds1",
                "    loader: jsonl",
                "role_adapters:",
                "  - adapter_id: dut",
                "    role_type: dut_model",
                "custom:",
                "  steps:",
                "    - step: inference",
                "      adapter_id: dut",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SmartDefaultsError, match="Unknown PipelineConfig scene 'arena'"):
        load_pre_smart_defaults_payload(config_path)


@pytest.mark.io
def test_static_aime_short_fixture_matches_expanded_snapshot() -> None:
    expanded = load_pipeline_config_payload(FIXTURE_DIR / "aime24_short.yaml")
    actual = yaml.safe_dump(expanded, sort_keys=False)
    expected_text = (FIXTURE_DIR / "aime24_short.expanded.yaml").read_text(encoding="utf-8")
    expected = yaml.safe_load(expected_text)

    assert expanded == expected
    assert "scene:" not in actual
    assert "task:" not in actual
    assert "backend:" not in actual
    assert "role_adapters:" in actual
    assert "custom:" in actual
    assert "steps:" in actual
    assert "tasks:" in actual
    assert "reporting:" in actual


@pytest.mark.io
@pytest.mark.parametrize("fixture_name", ["agent_noop.yaml", "game_noop.yaml"])
def test_agent_and_game_fixture_scenes_do_not_apply_static_defaults(fixture_name: str) -> None:
    path = FIXTURE_DIR / fixture_name
    raw = load_yaml_mapping(path)

    _assert_dataset_paths_exist(raw)
    normalized = load_pipeline_config_payload(path)

    assert "scene" not in normalized
    assert normalized["datasets"] == raw["datasets"]
    assert normalized["backends"] == raw["backends"]
    assert normalized["role_adapters"] == raw["role_adapters"]
    assert normalized["custom"] == raw["custom"]
    assert normalized["tasks"][0]["backend"] == raw["tasks"][0]["backend"]
    assert normalized["tasks"][0]["steps"] == raw["tasks"][0]["steps"]
    assert normalized["tasks"][0]["reporting"] == raw["tasks"][0]["reporting"]


@pytest.mark.io
@pytest.mark.parametrize("fixture_name", ["agent_noop.yaml", "game_noop.yaml"])
def test_agent_and_game_fixture_scenes_ignore_cli_backend_id_for_static_binding(fixture_name: str) -> None:
    path = FIXTURE_DIR / fixture_name
    raw = load_yaml_mapping(path)

    _assert_dataset_paths_exist(raw)
    normalized = load_pipeline_config_payload(path, cli_intent=CLIIntent(backend_id="other_backend"))

    assert "scene" not in normalized
    assert normalized["datasets"] == raw["datasets"]
    assert normalized["backends"] == raw["backends"]
    assert normalized["role_adapters"] == raw["role_adapters"]
    assert normalized["custom"] == raw["custom"]
    assert normalized["tasks"][0]["backend"] == raw["tasks"][0]["backend"]
    assert normalized["tasks"][0]["steps"] == raw["tasks"][0]["steps"]
