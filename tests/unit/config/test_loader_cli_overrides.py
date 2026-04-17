from __future__ import annotations

import pytest

from gage_eval.config.loader import materialize_pipeline_config_payload
from gage_eval.config.schema import SchemaValidationError
from gage_eval.config.loader_cli import CLIIntent, apply_cli_final_overrides, parse_metric_ids_csv


@pytest.mark.fast
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        (" exact_match, , full_metric ", ("exact_match", "full_metric")),
    ],
)
def test_parse_metric_ids_csv_trims_items_and_drops_empty_values(
    raw: str | None,
    expected: tuple[str, ...] | None,
) -> None:
    assert parse_metric_ids_csv(raw) == expected


@pytest.mark.fast
def test_metric_ids_filter_supports_all_metric_entry_forms() -> None:
    payload = {
        "metrics": [
            "exact_match",
            "fn_metric(case_sensitive=false)",
            {"kv_metric": {"case_sensitive": True}},
            {"metric_id": "full_metric", "implementation": "custom_impl"},
        ],
        "tasks": [{"task_id": "t1", "dataset_id": "ds"}],
    }

    apply_cli_final_overrides(payload, CLIIntent(metric_ids=("exact_match", "kv_metric", "full_metric")))

    assert payload["metrics"] == [
        "exact_match",
        {"kv_metric": {"case_sensitive": True}},
        {"metric_id": "full_metric", "implementation": "custom_impl"},
    ]


@pytest.mark.fast
def test_max_samples_override_updates_materialized_tasks_and_dataset_limits() -> None:
    payload = {
        "datasets": [{"dataset_id": "ds", "params": {}, "hub_params": {}}],
        "tasks": [{"task_id": "t1", "dataset_id": "ds"}],
    }

    apply_cli_final_overrides(payload, CLIIntent(max_samples=0))

    assert payload["tasks"][0]["max_samples"] == 0
    assert payload["datasets"][0]["params"]["limit"] == 0
    assert payload["datasets"][0]["hub_params"]["limit"] == 0


@pytest.mark.fast
def test_skip_judge_removes_judge_steps_from_custom_and_tasks() -> None:
    payload = {
        "custom": {"steps": [{"step": "inference"}, {"step": "judge"}, {"step": "auto_eval"}]},
        "tasks": [{"task_id": "t1", "dataset_id": "ds", "steps": [{"step": "inference"}, {"step": "judge"}]}],
    }

    apply_cli_final_overrides(payload, CLIIntent(skip_judge=True))

    assert payload["custom"]["steps"] == [{"step": "inference"}, {"step": "auto_eval"}]
    assert payload["tasks"][0]["steps"] == [{"step": "inference"}]


@pytest.mark.fast
def test_skip_judge_leaves_non_list_steps_unchanged() -> None:
    payload = {
        "custom": {"steps": "pipeline-template"},
        "tasks": [{"task_id": "t1", "dataset_id": "ds", "steps": "task-template"}],
    }

    apply_cli_final_overrides(payload, CLIIntent(skip_judge=True))

    assert payload["custom"]["steps"] == "pipeline-template"
    assert payload["tasks"][0]["steps"] == "task-template"


@pytest.mark.fast
def test_materialize_applies_cli_final_overrides_after_static_smart_defaults() -> None:
    payload = {
        "api_version": "gage/v1alpha1",
        "kind": "PipelineConfig",
        "scene": "static",
        "metadata": {"name": "aime24"},
        "datasets": [
            {
                "dataset_id": "aime24_ds",
                "hub_id": "Maxwell-Jia/AIME_2024",
                "split": "train",
                "params": {},
                "hub_params": {},
            }
        ],
        "backends": [
            {
                "backend_id": "openai",
                "type": "litellm",
                "config": {"provider": "openai", "model": "gpt-4.1"},
            }
        ],
        "role_adapters": [
            {"adapter_id": "dut_openai", "role_type": "dut_model", "backend_id": "openai"},
            {"adapter_id": "judge", "role_type": "judge_model", "backend_id": "openai"},
        ],
        "custom": {"steps": [{"step": "inference"}, {"step": "judge"}, {"step": "auto_eval"}]},
        "metrics": ["exact_match", "fn_metric(case_sensitive=false)"],
        "task": {"max_samples": 30},
    }

    normalized = materialize_pipeline_config_payload(
        payload,
        source_path=None,
        cli_intent=CLIIntent(max_samples=0, metric_ids=("exact_match",), skip_judge=True),
    )

    assert normalized["tasks"][0]["max_samples"] == 0
    assert normalized["datasets"][0]["params"]["limit"] == 0
    assert normalized["datasets"][0]["hub_params"]["limit"] == 0
    assert normalized["metrics"] == ["exact_match"]
    assert normalized["custom"]["steps"] == [{"step": "inference"}, {"step": "auto_eval"}]
    assert [step["step"] for step in normalized["tasks"][0]["steps"]] == ["inference", "auto_eval"]


@pytest.mark.fast
def test_metric_ids_filter_preserves_malformed_entries_for_schema_validation() -> None:
    payload = {
        "kind": "PipelineConfig",
        "datasets": [{"dataset_id": "ds", "loader": "jsonl"}],
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
        "metrics": ["keep", 123],
    }

    with pytest.raises(SchemaValidationError, match="metric entries must be dictionaries"):
        materialize_pipeline_config_payload(
            payload,
            source_path=None,
            cli_intent=CLIIntent(metric_ids=("keep",)),
            smart_defaults=False,
        )


@pytest.mark.fast
def test_metric_ids_filter_preserves_malformed_metrics_container_for_schema_validation() -> None:
    payload = {
        "kind": "PipelineConfig",
        "datasets": [{"dataset_id": "ds", "loader": "jsonl"}],
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
        "metrics": "",
    }

    with pytest.raises(SchemaValidationError, match="'metrics' must be a list"):
        materialize_pipeline_config_payload(
            payload,
            source_path=None,
            cli_intent=CLIIntent(metric_ids=("keep",)),
            smart_defaults=False,
        )


@pytest.mark.fast
def test_runconfig_materialization_preserves_cli_intent_when_smart_defaults_disabled() -> None:
    payload = {"kind": "RunConfig"}

    def compiler(expanded: dict[str, object]) -> tuple[dict[str, object], None]:
        assert expanded == {"kind": "RunConfig"}
        return (
            {
                "kind": "PipelineConfig",
                "scene": "static",
                "datasets": [{"dataset_id": "ds", "loader": "jsonl", "params": {}, "hub_params": {}}],
                "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
                "custom": {"steps": [{"step": "inference"}, {"step": "judge"}]},
                "metrics": ["keep", "drop"],
                "tasks": [
                    {
                        "task_id": "t1",
                        "dataset_id": "ds",
                        "steps": [{"step": "inference"}, {"step": "judge"}],
                        "max_samples": 5,
                    }
                ],
            },
            None,
        )

    normalized = materialize_pipeline_config_payload(
        payload,
        source_path=None,
        run_config_compiler=compiler,
        cli_intent=CLIIntent(max_samples=0, metric_ids=("keep",), skip_judge=True),
        smart_defaults=False,
    )

    assert normalized["tasks"][0]["max_samples"] == 0
    assert normalized["datasets"][0]["params"]["limit"] == 0
    assert normalized["datasets"][0]["hub_params"]["limit"] == 0
    assert normalized["metrics"] == ["keep"]
    assert normalized["custom"]["steps"] == [{"step": "inference"}]
    assert normalized["tasks"][0]["steps"] == [{"step": "inference"}]
