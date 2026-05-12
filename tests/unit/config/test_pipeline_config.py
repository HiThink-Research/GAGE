import pytest

from gage_eval.config.pipeline_config import EnvironmentSpec, PipelineConfig


@pytest.mark.fast
def test_pipeline_config_parses_agent_sections():
    payload = {
        "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
        "agent_backends": [
            {
                "agent_backend_id": "agent_main",
                "type": "agent_class",
                "config": {"agent_class": "tests.unit.config.test_pipeline_config:DummyAgent"},
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
                "agent_backend_id": "agent_main",
            }
        ],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut_agent_main"}]},
    }
    config = PipelineConfig.from_dict(payload)
    assert config.agent_backends[0].agent_backend_id == "agent_main"
    assert config.sandbox_profiles[0].sandbox_id == "demo"
    assert config.mcp_clients[0].mcp_client_id == "mcp1"
    assert config.role_adapters[0].agent_backend_id == "agent_main"


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


@pytest.mark.fast
def test_pipeline_config_exposes_environments_as_first_class_section() -> None:
    config = PipelineConfig.from_dict(
        {
            "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
            "environments": [
                {
                    "env_id": "docker_env",
                    "provider": "docker",
                    "resources": {"cpus": 2},
                }
            ],
            "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
            "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
        }
    )

    assert config.environments == (
        EnvironmentSpec(
            env_id="docker_env",
            provider="docker",
            resources={"cpus": 2},
        ),
    )
    assert config.environments[0].to_dict() == {
        "env_id": "docker_env",
        "provider": "docker",
        "resources": {"cpus": 2},
    }
    assert not any(key.startswith("_external_harness_") for key in config.metadata)


@pytest.mark.fast
def test_pipeline_config_allows_static_configs_without_environments() -> None:
    config = PipelineConfig.from_dict(
        {
            "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
            "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
            "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
        }
    )

    assert config.environments == ()
