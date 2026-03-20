from __future__ import annotations

import pytest

from gage_eval.config.pipeline_builder import PipelineConfigBuildError
from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.fast
def test_pipeline_config_parses_support_payload_policy() -> None:
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
                    "support_payload_policy": {"projection_mode": "compact_latest"},
                }
            ],
        }
    )

    assert config.tasks[0].support_payload_policy == {"projection_mode": "compact_latest"}


@pytest.mark.fast
def test_pipeline_config_rejects_non_mapping_support_payload_policy() -> None:
    with pytest.raises(PipelineConfigBuildError, match="support_payload_policy"):
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
