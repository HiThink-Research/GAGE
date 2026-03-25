from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.registry.asset_planner import RuntimeAssetPlanner


@pytest.mark.fast
def test_runtime_asset_planner_collects_arena_defaults_and_eager_kinds() -> None:
    config = PipelineConfig.from_dict(
        {
            "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
            "role_adapters": [
                {
                    "adapter_id": "arena",
                    "role_type": "arena",
                    "params": {
                        "environment": {"impl": "tictactoe_v1"},
                        "parser": {},
                        "visualizer": {"enabled": True},
                    },
                }
            ],
            "custom": {"steps": [{"step": "arena", "adapter_id": "arena"}]},
            "summary_generators": ["arena_summary"],
        }
    )

    plan = RuntimeAssetPlanner().build_plan(config)

    assert tuple(request.name for request in plan.requests_for_kind("roles")) == ("arena",)
    assert tuple(request.name for request in plan.requests_for_kind("arena_impls")) == ("tictactoe_v1",)
    assert tuple(request.name for request in plan.requests_for_kind("parser_impls")) == ("grid_parser_v1",)
    assert tuple(request.name for request in plan.requests_for_kind("pipeline_steps")) == ("arena",)
    assert tuple(request.name for request in plan.requests_for_kind("summary_generators")) == ("arena_summary",)
    assert plan.requests_for_kind("renderer_impls") == ()
    assert set(plan.eager_kinds) >= {
        "pipeline_steps",
        "summary_generators",
        "arena_game_providers",
        "renderer_impls",
    }


@pytest.mark.fast
def test_runtime_asset_planner_collects_dataset_inline_backend_and_observability_assets() -> None:
    config = PipelineConfig.from_dict(
        {
            "datasets": [
                {
                    "dataset_id": "d1",
                    "loader": "jsonl",
                    "params": {
                        "path": "demo.jsonl",
                        "bundle": "demo_bundle",
                        "preprocess": "normalize",
                    },
                    "preprocess_chain": [
                        {"name": "normalize"},
                        {"type": "finalize"},
                    ],
                }
            ],
            "models": [{"model_id": "m1"}],
            "backends": [{"backend_id": "b1", "type": "dummy"}],
            "role_adapters": [
                {
                    "adapter_id": "dut",
                    "role_type": "dut_model",
                    "backend": {"type": "openai_http", "config": {"model": "gpt-4o-mini"}},
                }
            ],
            "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
            "observability": {"plugin": "trace", "sink": "file"},
        }
    )

    plan = RuntimeAssetPlanner().build_plan(config)

    assert tuple(request.name for request in plan.requests_for_kind("dataset_loaders")) == ("jsonl",)
    assert tuple(request.name for request in plan.requests_for_kind("dataset_hubs")) == ("inline",)
    assert tuple(request.name for request in plan.requests_for_kind("bundles")) == ("demo_bundle",)
    assert tuple(request.name for request in plan.requests_for_kind("dataset_preprocessors")) == (
        "normalize",
        "finalize",
    )
    assert tuple(request.name for request in plan.requests_for_kind("model_hubs")) == ("huggingface",)
    assert tuple(request.name for request in plan.requests_for_kind("backends")) == ("dummy", "openai_http")
    assert tuple(request.name for request in plan.requests_for_kind("observability_plugins")) == ("trace",)
    assert tuple(request.name for request in plan.requests_for_kind("reporting_sinks")) == ("file",)
