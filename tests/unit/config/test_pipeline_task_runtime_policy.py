from __future__ import annotations

import pytest

from gage_eval.config.pipeline_builder import PipelineConfigBuildError
from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.fast
def test_pipeline_config_parses_task_runtime_policy_fields() -> None:
    config = PipelineConfig.from_dict(
        {
            "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
            "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
            "custom": {"steps": [{"step": "inference", "role_ref": "dut"}]},
            "tasks": [
                {
                    "task_id": "t1",
                    "dataset_id": "d1",
                    "steps": [{"step": "inference", "role_ref": "dut"}],
                    "shuffle_strategy": "reservoir",
                    "shuffle_small_dataset_threshold": 512,
                    "keep_shuffle_artifacts": True,
                    "failure_policy": "graceful",
                    "metric_concurrency": 3,
                    "report_partial_on_failure": False,
                    "support_payload_policy": {"projection_mode": "compact_latest"},
                }
            ],
        }
    )

    task = config.tasks[0]
    assert task.shuffle_strategy == "reservoir"
    assert task.shuffle_small_dataset_threshold == 512
    assert task.keep_shuffle_artifacts is True
    assert task.failure_policy == "graceful"
    assert task.metric_concurrency == 3
    assert task.report_partial_on_failure is False
    assert task.support_payload_policy == {"projection_mode": "compact_latest"}


@pytest.mark.fast
def test_pipeline_config_rejects_invalid_failure_policy() -> None:
    with pytest.raises(
        PipelineConfigBuildError,
        match="unsupported failure_policy 'explode'",
    ):
        PipelineConfig.from_dict(
            {
                "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
                "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
                "custom": {"steps": [{"step": "inference", "role_ref": "dut"}]},
                "tasks": [
                    {
                        "task_id": "t1",
                        "dataset_id": "d1",
                        "steps": [{"step": "inference", "role_ref": "dut"}],
                        "failure_policy": "explode",
                    }
                ],
            }
        )


@pytest.mark.fast
def test_pipeline_config_rejects_invalid_shuffle_strategy() -> None:
    with pytest.raises(
        PipelineConfigBuildError,
        match="unsupported shuffle_strategy 'chaos'",
    ):
        PipelineConfig.from_dict(
            {
                "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
                "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
                "custom": {"steps": [{"step": "inference", "role_ref": "dut"}]},
                "tasks": [
                    {
                        "task_id": "t1",
                        "dataset_id": "d1",
                        "steps": [{"step": "inference", "role_ref": "dut"}],
                        "shuffle_strategy": "chaos",
                    }
                ],
            }
        )


@pytest.mark.fast
def test_pipeline_config_rejects_non_mapping_support_payload_policy() -> None:
    with pytest.raises(
        PipelineConfigBuildError,
        match="support_payload_policy",
    ):
        PipelineConfig.from_dict(
            {
                "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
                "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
                "custom": {"steps": [{"step": "inference", "role_ref": "dut"}]},
                "tasks": [
                    {
                        "task_id": "t1",
                        "dataset_id": "d1",
                        "steps": [{"step": "inference", "role_ref": "dut"}],
                        "support_payload_policy": "compact_latest",
                    }
                ],
            }
        )
