from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.config.registry import ConfigRegistry


@pytest.mark.fast
def test_config_registry_injects_agent_runtime_resolver_for_dut_agent() -> None:
    config = PipelineConfig.from_dict(
        {
            "datasets": [{"dataset_id": "dataset-1", "loader": "inline", "params": {}}],
            "custom": {"steps": [{"step": "inference", "adapter_id": "dut-1", "params": {}}]},
            "agent_runtimes": [
                {
                    "agent_runtime_id": "runtime-1",
                    "scheduler": "installed_client",
                    "benchmark_kit_id": "swebench",
                }
            ],
            "role_adapters": [
                {
                    "adapter_id": "dut-1",
                    "role_type": "dut_agent",
                    "agent_runtime_id": "runtime-1",
                }
            ],
        }
    )

    adapter = ConfigRegistry().materialize_role_adapters(config)["dut-1"]

    assert getattr(adapter, "_agent_runtime_resolver", None) is not None
    assert getattr(adapter, "_agent_runtime_id", None) == "runtime-1"
