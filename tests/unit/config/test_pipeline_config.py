import pytest

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.fast
def test_pipeline_config_parses_agent_sections():
    payload = {
        "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
        "backends": [
            {
                "backend_id": "agent_model",
                "type": "dummy",
                "config": {"responses": ["done"]},
            }
        ],
        "agent_runtimes": [
            {
                "agent_runtime_id": "demo_runtime",
                "benchmark_kit_id": "terminal_bench",
                "scheduler_type": "framework_loop",
                "sandbox_profile_id": "demo",
                "resource_policy": {"resource_kind": "docker", "lifecycle": "per_sample"},
                "verifier_binding_id": "terminal_bench_native",
            }
        ],
        "sandbox_profiles": [
            {
                "sandbox_id": "demo",
                "runtime": "docker",
                "image": "demo:latest",
                "resources": {"cpu": 1},
            }
        ],
        "mcp_clients": [
            {
                "mcp_client_id": "mcp1",
                "transport": "http_sse",
                "endpoint": "http://127.0.0.1:5001/mcp",
            }
        ],
        "role_adapters": [
            {
                "adapter_id": "dut_agent_main",
                "role_type": "dut_agent",
                "backend_id": "agent_model",
                "agent_runtime_id": "demo_runtime",
            }
        ],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut_agent_main"}]},
    }
    config = PipelineConfig.from_dict(payload)
    assert config.agent_runtimes[0].agent_runtime_id == "demo_runtime"
    assert config.sandbox_profiles[0].sandbox_id == "demo"
    assert config.mcp_clients[0].mcp_client_id == "mcp1"
    assert config.role_adapters[0].backend_id == "agent_model"
    assert config.role_adapters[0].agent_runtime_id == "demo_runtime"


@pytest.mark.fast
def test_pipeline_config_preserves_arena_v2_adapter_params() -> None:
    config = PipelineConfig.from_dict(
        {
            "datasets": [{"dataset_id": "arena_ds", "loader": "dummy"}],
            "role_adapters": [
                {
                    "adapter_id": "arena_main",
                    "role_type": "arena",
                    "params": {
                        "game_kit": "gomoku",
                        "env": "gomoku_standard",
                        "scheduler": {"binding_id": "turn/default"},
                        "runtime_overrides": {"board_size": 9},
                    },
                }
            ],
            "custom": {"steps": [{"step": "arena", "adapter_id": "arena_main"}]},
        }
    )

    params = config.role_adapters[0].params
    assert params["game_kit"] == "gomoku"
    assert params["env"] == "gomoku_standard"
    assert params["scheduler"] == {"binding_id": "turn/default"}
    assert params["runtime_overrides"] == {"board_size": 9}
