from __future__ import annotations

import pytest
from pydantic import ValidationError

from gage_eval.role.model.config.generations import GenerationParameters
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig


def test_generation_parameters_accepts_max_tokens_alias() -> None:
    params = GenerationParameters.model_validate({"max_tokens": 16384, "temperature": 0.0})

    assert params.max_new_tokens == 16384
    assert params.temperature == 0.0
    assert params.to_dict()["max_new_tokens"] == 16384


def test_generation_parameters_default_max_new_tokens_is_swebench_friendly() -> None:
    assert GenerationParameters().max_new_tokens == 4096


def test_generation_parameters_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        GenerationParameters.model_validate({"max_tokns": 16384})


def test_backend_config_validates_nested_generation_parameters() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "openai/local",
            "generation_parameters": {"max_tokens": 8192},
        }
    )

    assert config.generation_parameters.max_new_tokens == 8192

    with pytest.raises(ValidationError):
        LiteLLMBackendConfig.model_validate(
            {
                "model": "openai/local",
                "generation_parameters": {"max_tokns": 8192},
            }
        )
