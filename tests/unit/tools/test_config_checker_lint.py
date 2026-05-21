from __future__ import annotations

from gage_eval.tools.config_checker import lint_pipeline_config_payload


def test_config_lint_flags_deprecated_preprocessor() -> None:
    issues = lint_pipeline_config_payload(
        {
            "datasets": [
                {
                    "dataset_id": "piqa_validation",
                    "loader": "hf_hub",
                    "params": {"preprocess": "piqa_struct_only"},
                }
            ]
        }
    )

    assert any("piqa_struct_only" in issue for issue in issues)
    assert any("global_piqa_chat_preprocessor" in issue for issue in issues)


def test_config_lint_flags_nonstandard_litellm_provider_alias() -> None:
    issues = lint_pipeline_config_payload(
        {
            "backends": [
                {
                    "backend_id": "lmstudio_openai_compatible",
                    "type": "litellm",
                    "config": {
                        "provider": "openai_compatible",
                        "model": "openai/qwen/qwen3.5-9b",
                    },
                }
            ]
        }
    )

    assert any("openai_compatible" in issue and "openai" in issue for issue in issues)


def test_config_lint_allows_explicit_provider_for_unprefixed_local_model() -> None:
    issues = lint_pipeline_config_payload(
        {
            "backends": [
                {
                    "backend_id": "lmstudio_openai",
                    "type": "litellm",
                    "config": {
                        "provider": "openai",
                        "custom_llm_provider": "openai",
                        "model": "qwen/qwen3.5-9b",
                    },
                }
            ]
        }
    )

    assert issues == []
