from __future__ import annotations

from pathlib import Path
from typing import Any

from gage_eval.config.loader import materialize_pipeline_config_payload


def _manual_pipeline_payload() -> dict[str, Any]:
    return {
        "api_version": "gage/v1alpha1",
        "kind": "PipelineConfig",
        "metadata": {"name": "manual-agentkit-v2"},
        "datasets": [
            {
                "dataset_id": "tau2_telecom_base",
                "loader": "tau2_tasks",
                "params": {
                    "source": "huggingface",
                    "hub_id": "HuggingFaceH4/tau2-bench-data",
                    "domain": "telecom",
                    "task_split": "base",
                    "data_dir": "./local-datasets/tau2",
                    "preprocess": "tau2_preprocessor",
                },
            }
        ],
        "backends": [
            {
                "backend_id": "lmstudio_openai_http",
                "type": "openai_http",
                "config": {
                    "base_url": "http://127.0.0.1:1234/v1",
                    "model": "qwen/qwen3.5-9b",
                    "api_key": "dummy",
                    "require_api_key": False,
                },
            }
        ],
        "prompts": [
            {
                "prompt_id": "dut/tau2@manual-lmstudio",
                "renderer": "jinja_chat",
                "template": "{{ sample.inputs.prompt }}",
            }
        ],
        "agents": [
            {
                "agent_id": "tau2_agent",
                "scheduler": {
                    "type": "framework_loop",
                    "backend_id": "lmstudio_openai_http",
                    "config": {"max_turns": 200, "cost_limit_usd": 3.0},
                },
                "config": {"prompt_id": "dut/tau2@manual-lmstudio"},
                "tooling": {"skill_ids": [], "mcp_servers": []},
            }
        ],
        "benchmarks": [
            {
                "benchmark_id": "tau2_telecom",
                "kit_id": "tau2",
                "config": {
                    "domain": "telecom",
                    "user_simulator": {
                        "type": "openai_http",
                        "model": "openai/qwen/qwen3.5-9b",
                        "base_url": "http://127.0.0.1:1234/v1",
                        "api_key": "dummy",
                    },
                },
            }
        ],
        "environments": [
            {
                "env_id": "tau2_local_process",
                "provider": "local_process",
                "profile_id": "tau2-local-process",
                "profile": {},
                "provider_config": {
                    "data_dir": "./local-datasets/tau2",
                    "max_steps": 200,
                    "max_errors": 10,
                    "respond_tool_name": "respond",
                },
                "lifecycle": "per_sample",
            }
        ],
        "dut_agents": [
            {
                "dut_id": "tau2_dut",
                "agent_id": "tau2_agent",
                "env_id": "tau2_local_process",
                "benchmark_id": "tau2_telecom",
            }
        ],
        "metrics": [{"metric_id": "tau2_reward", "implementation": "tau2_reward"}],
        "tasks": [
            {
                "task_id": "manual_tau2_local_lmstudio",
                "dataset_id": "tau2_telecom_base",
                "steps": [
                    {"step": "inference", "adapter_id": "tau2_dut"},
                    {"step": "auto_eval"},
                ],
                "max_samples": 1,
            }
        ],
    }


def test_pipeline_config_agentkit_v2_sections_lower_to_thin_role_adapter() -> None:
    payload = _manual_pipeline_payload()

    materialized = materialize_pipeline_config_payload(payload, source_path=Path("manual.yaml"))

    assert "role_adapters" not in payload
    assert materialized["agents"][0]["agent_id"] == "tau2_agent"
    assert materialized["environments"][0]["env_id"] == "tau2_local_process"
    assert materialized["dut_agents"][0] == {
        "dut_id": "tau2_dut",
        "agent_id": "tau2_agent",
        "env_id": "tau2_local_process",
        "benchmark_id": "tau2_telecom",
        "trial_policy": {"trials": 1},
    }

    generated = materialized["role_adapters"]
    assert generated == [
        {
            "adapter_id": "tau2_dut",
            "role_type": "dut_agent",
            "backend_id": "lmstudio_openai_http",
            "agent_runtime_id": "tau2_framework_loop",
            "prompt_id": "dut/tau2@manual-lmstudio",
            "params": {
                "environment_profile": {
                    "provider": "local_process",
                    "profile_id": "tau2-local-process",
                    "config": {
                        "data_dir": "./local-datasets/tau2",
                        "max_steps": 200,
                        "max_errors": 10,
                        "respond_tool_name": "respond",
                    },
                },
                "provider_config": {
                    "data_dir": "./local-datasets/tau2",
                    "max_steps": 200,
                    "max_errors": 10,
                    "respond_tool_name": "respond",
                    "user_simulator": {
                        "model": "openai/qwen/qwen3.5-9b",
                        "model_args": {
                            "api_base": "http://127.0.0.1:1234/v1",
                            "api_key": "dummy",
                        },
                    },
                },
                "resources": {},
                "startup_env": {},
                "lifecycle": "per_sample",
                "max_turns": 200,
                "cost_limit_usd": 3.0,
            },
        }
    ]
