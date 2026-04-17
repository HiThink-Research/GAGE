from __future__ import annotations

from copy import deepcopy

import pytest

from gage_eval.config.loader import materialize_pipeline_config_payload
from gage_eval.config.smart_defaults import SmartDefaultsError


def _base_static_payload() -> dict:
    return {
        "kind": "PipelineConfig",
        "scene": "static",
        "metadata": {"name": "aime24"},
        "datasets": [
            {
                "dataset_id": "aime24_ds",
                "hub_id": "Maxwell-Jia/AIME_2024",
                "split": "train",
                "params": {"preprocess": "aime2024_preprocessor"},
            }
        ],
        "backends": [
            {
                "backend_id": "openai",
                "type": "litellm",
                "config": {"api_base": "https://api.openai.com/v1", "model": "gpt-4.1"},
            }
        ],
        "role_adapters": [{"adapter_id": "dut_openai", "role_type": "dut_model", "backend_id": "openai"}],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut_openai"}]},
    }


@pytest.mark.fast
def test_static_dataset_hub_sugar_is_lowered_to_hub_params() -> None:
    normalized = materialize_pipeline_config_payload(_base_static_payload(), source_path=None)

    dataset = normalized["datasets"][0]
    assert dataset["hub"] == "huggingface"
    assert dataset["loader"] == "hf_hub"
    assert dataset["hub_params"] == {"hub_id": "Maxwell-Jia/AIME_2024", "split": "train"}
    assert dataset["params"]["preprocess_kwargs"] == {}
    assert "hub_id" not in dataset
    assert "split" not in dataset


@pytest.mark.fast
def test_static_litellm_defaults_are_filled_without_generation_params() -> None:
    normalized = materialize_pipeline_config_payload(_base_static_payload(), source_path=None)

    config = normalized["backends"][0]["config"]
    assert config["provider"] == "openai"
    assert config["streaming"] is False
    assert config["max_retries"] == 6
    assert "generation_parameters" not in config


@pytest.mark.fast
def test_static_vllm_tokenizer_defaults_do_not_fill_inference_tuning_fields() -> None:
    payload = _base_static_payload()
    payload["backends"] = [{"backend_id": "vllm_qwen", "type": "vllm", "config": {"model_path": "/models/qwen"}}]
    payload["role_adapters"][0]["backend_id"] = "vllm_qwen"

    normalized = materialize_pipeline_config_payload(payload, source_path=None)

    config = normalized["backends"][0]["config"]
    assert config["tokenizer_path"] == "/models/qwen"
    assert config["force_tokenize_prompt"] is True
    assert config["tokenizer_trust_remote_code"] is True
    assert "max_tokens" not in config
    assert "max_model_len" not in config
    assert "sampling_params" not in config


@pytest.mark.fast
def test_static_local_jsonl_dataset_does_not_receive_empty_hub_params_and_keeps_original_payload() -> None:
    payload = _base_static_payload()
    payload["datasets"] = [{"dataset_id": "local_ds", "params": {"path": "tests/data/sample.jsonl"}}]
    before = deepcopy(payload)

    normalized = materialize_pipeline_config_payload(payload, source_path=None)

    assert normalized["datasets"][0]["loader"] == "jsonl"
    assert "hub_params" not in normalized["datasets"][0]
    assert payload == before


@pytest.mark.fast
def test_static_vlm_transformers_backend_is_left_untouched() -> None:
    payload = _base_static_payload()
    payload["backends"] = [
        {
            "backend_id": "local_vlm",
            "type": "vlm_transformers",
            "config": {"model_name_or_path": "/models/qwen-vl"},
        }
    ]
    payload["role_adapters"][0]["backend_id"] = "local_vlm"

    normalized = materialize_pipeline_config_payload(payload, source_path=None)

    assert normalized["backends"][0]["config"] == {"model_name_or_path": "/models/qwen-vl"}


@pytest.mark.fast
def test_static_dataset_hub_params_explicit_values_win() -> None:
    payload = _base_static_payload()
    payload["datasets"][0]["hub_params"] = {"hub_id": "explicit/dataset", "subset": "explicit-subset"}
    payload["datasets"][0]["subset"] = "sugar-subset"
    payload["datasets"][0]["revision"] = "main"

    normalized = materialize_pipeline_config_payload(payload, source_path=None)

    assert normalized["datasets"][0]["hub_params"] == {
        "hub_id": "explicit/dataset",
        "split": "train",
        "subset": "explicit-subset",
        "revision": "main",
    }
    assert "subset" not in normalized["datasets"][0]
    assert "revision" not in normalized["datasets"][0]


@pytest.mark.fast
def test_static_dataset_hub_params_must_be_mapping() -> None:
    payload = _base_static_payload()
    payload["datasets"][0]["hub_params"] = "bad"

    with pytest.raises(SmartDefaultsError, match="dataset.hub_params must be a mapping"):
        materialize_pipeline_config_payload(payload, source_path=None)
