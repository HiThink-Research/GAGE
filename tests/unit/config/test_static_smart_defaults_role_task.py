from __future__ import annotations

from copy import deepcopy

import pytest

from gage_eval.config.loader import materialize_pipeline_config_payload
from gage_eval.config.loader_cli import CLIIntent
from gage_eval.config.smart_defaults import SmartDefaultsError


def _short_static_payload() -> dict:
    return {
        "api_version": "gage/v1alpha1",
        "kind": "PipelineConfig",
        "scene": "static",
        "metadata": {"name": "aime24"},
        "datasets": [
            {
                "dataset_id": "aime24_ds",
                "hub_id": "Maxwell-Jia/AIME_2024",
                "split": "train",
            }
        ],
        "backends": [
            {
                "backend_id": "openai",
                "type": "litellm",
                "config": {"provider": "openai", "model": "gpt-4.1"},
            }
        ],
        "metrics": ["aime2024_accuracy"],
        "task": {"max_samples": 30},
    }


@pytest.mark.fast
def test_short_static_payload_materializes_implicit_single_dut_steps_task_and_reporting() -> None:
    normalized = materialize_pipeline_config_payload(_short_static_payload(), source_path=None)

    assert normalized["role_adapters"] == [
        {
            "adapter_id": "dut_openai",
            "role_type": "dut_model",
            "backend_id": "openai",
            "capabilities": ["chat_completion"],
        }
    ]
    assert normalized["custom"]["steps"] == [{"step": "inference"}, {"step": "auto_eval"}]
    assert normalized["tasks"][0]["task_id"] == "aime24"
    assert normalized["tasks"][0]["dataset_id"] == "aime24_ds"
    assert normalized["tasks"][0]["max_samples"] == 30
    assert normalized["tasks"][0]["steps"][0]["adapter_id"] == "dut_openai"
    assert normalized["tasks"][0]["reporting"]["sinks"] == [{"type": "console"}, {"type": "file"}]
    assert "task" not in normalized
    assert "backend" not in normalized["tasks"][0]


@pytest.mark.fast
def test_static_smart_defaults_do_not_mutate_original_payload() -> None:
    payload = _short_static_payload()
    before = deepcopy(payload)

    materialize_pipeline_config_payload(payload, source_path=None)

    assert payload == before


@pytest.mark.fast
def test_explicit_empty_tasks_are_preserved() -> None:
    payload = _short_static_payload()
    payload.pop("task")
    payload["tasks"] = []
    payload["role_adapters"] = [
        {
            "adapter_id": "dut_openai",
            "role_type": "dut_model",
            "backend_id": "openai",
            "capabilities": ["chat_completion"],
        }
    ]
    payload["custom"] = {"steps": [{"step": "inference", "adapter_id": "dut_openai"}]}

    normalized = materialize_pipeline_config_payload(payload, source_path=None)

    assert normalized["tasks"] == []


@pytest.mark.fast
def test_explicit_empty_role_adapters_still_generates_dut_adapters() -> None:
    payload = _short_static_payload()
    payload["role_adapters"] = []

    normalized = materialize_pipeline_config_payload(payload, source_path=None)

    assert normalized["role_adapters"] == [
        {
            "adapter_id": "dut_openai",
            "role_type": "dut_model",
            "backend_id": "openai",
            "capabilities": ["chat_completion"],
        }
    ]


@pytest.mark.fast
def test_cli_backend_id_overrides_task_backend_for_dut_adapter_binding() -> None:
    payload = _short_static_payload()
    payload["backends"].append(
        {"backend_id": "vllm_qwen", "type": "vllm", "config": {"model_path": "/models/qwen"}}
    )
    payload["task"]["backend"] = "openai"

    normalized = materialize_pipeline_config_payload(
        payload,
        source_path=None,
        cli_intent=CLIIntent(backend_id="vllm_qwen"),
    )

    assert normalized["tasks"][0]["steps"][0]["adapter_id"] == "dut_vllm_qwen"


@pytest.mark.fast
def test_task_backend_expand_rejects_unknown_backend() -> None:
    payload = _short_static_payload()
    payload["task"]["backend"] = "missing"

    with pytest.raises(SmartDefaultsError) as excinfo:
        materialize_pipeline_config_payload(payload, source_path=None)

    message = str(excinfo.value)
    assert "cannot find unique DUT adapter" in message
    assert "tasks[0].backend" in message


@pytest.mark.fast
def test_cli_backend_id_rejects_unknown_backend_with_cli_path() -> None:
    payload = _short_static_payload()

    with pytest.raises(SmartDefaultsError) as excinfo:
        materialize_pipeline_config_payload(
            payload,
            source_path=None,
            cli_intent=CLIIntent(backend_id="missing"),
        )

    message = str(excinfo.value)
    assert "cannot find unique DUT adapter" in message
    assert "cli.backend_id" in message
    assert "tasks[0].backend" not in message


@pytest.mark.fast
def test_task_backend_from_single_dut_rejects_ambiguous_backend() -> None:
    payload = _short_static_payload()
    payload["backends"].append(
        {"backend_id": "vllm_qwen", "type": "vllm", "config": {"model_path": "/models/qwen"}}
    )

    with pytest.raises(SmartDefaultsError, match="does not have exactly one DUT adapter"):
        materialize_pipeline_config_payload(payload, source_path=None)


@pytest.mark.fast
def test_auto_custom_steps_does_not_apply_to_judge_adapter() -> None:
    payload = _short_static_payload()
    payload["role_adapters"] = [
        {"adapter_id": "dut_openai", "role_type": "dut_model", "backend_id": "openai"},
        {"adapter_id": "judge", "role_type": "judge_model", "backend_id": "openai"},
    ]

    with pytest.raises(Exception, match="custom|steps|pipeline"):
        materialize_pipeline_config_payload(payload, source_path=None)


@pytest.mark.fast
def test_auto_custom_steps_fills_steps_when_custom_mapping_exists_without_steps() -> None:
    payload = _short_static_payload()
    payload["custom"] = {"metadata": {"owner": "test"}}

    normalized = materialize_pipeline_config_payload(payload, source_path=None)

    assert normalized["custom"]["steps"] == [{"step": "inference"}, {"step": "auto_eval"}]
    assert normalized["custom"]["metadata"] == {"owner": "test"}
