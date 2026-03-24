import pytest

from gage_eval.config.pipeline_builder import PipelineConfigBuildError, PipelineConfigBuilder
from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.fast
def test_pipeline_config_builder_materializes_pipeline_sections() -> None:
    payload = {
        "metadata": {"name": "builder-demo"},
        "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
        "backends": [{"backend_id": "b1", "type": "openai", "config": {"model": "gpt"}}],
        "models": [{"model_id": "m1", "source": "openai", "params": {"temperature": 0.1}}],
        "prompts": [{"prompt_id": "p1", "renderer": "jinja2", "template": "hi"}],
        "role_adapters": [
            {
                "adapter_id": "dut",
                "role_type": "dut_model",
                "backend_id": "b1",
                "prompt_id": "p1",
            }
        ],
        "metrics": ["exact_match"],
        "summary_generators": ["arena"],
        "custom": {"steps": [{"step": "inference", "role_ref": "dut"}]},
        "tasks": [
            {
                "task_id": "t1",
                "dataset_id": "d1",
                "steps": [{"step": "inference", "role_ref": "dut"}],
            }
        ],
    }

    config = (
        PipelineConfigBuilder(payload)
        .normalize()
        .build_root()
        .build_assets()
        .build_role_adapters()
        .build_metrics()
        .build_tasks()
        .build()
    )

    assert config.metadata["name"] == "builder-demo"
    assert config.custom is not None
    assert config.custom.steps[0].adapter_id == "dut"
    assert config.role_adapters[0].prompt_id == "p1"
    assert config.tasks[0].steps[0].adapter_id == "dut"
    assert config.summary_generators == ("arena",)


@pytest.mark.fast
def test_pipeline_config_from_dict_uses_staged_builder() -> None:
    payload = {
        "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "inference", "role_ref": "dut"}]},
    }

    config = PipelineConfig.from_dict(payload)

    assert config.custom is not None
    assert config.custom.steps[0].adapter_id == "dut"


@pytest.mark.fast
def test_pipeline_config_builder_reports_normalize_stage_errors() -> None:
    payload = {
        "datasets": "bad",
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
    }

    with pytest.raises(PipelineConfigBuildError, match="E_NORMALIZE \\[normalize\\] <payload>:") as exc_info:
        PipelineConfigBuilder(payload).normalize()

    exc = exc_info.value
    assert exc.code == "E_NORMALIZE"
    assert exc.stage == "normalize"
    assert exc.field_path == "<payload>"
    assert "'datasets' must be a list" in str(exc)


@pytest.mark.fast
def test_pipeline_config_builder_reports_summary_generator_field_path() -> None:
    payload = {
        "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
        "summary_generators": [{}],
    }

    with pytest.raises(
        PipelineConfigBuildError,
        match=r"E_BUILD_SUMMARY_GENERATORS \[build_metrics\] summary_generators\[0\]:",
    ) as exc_info:
        (
            PipelineConfigBuilder(payload)
            .normalize()
            .build_root()
            .build_assets()
            .build_role_adapters()
            .build_metrics()
        )

    exc = exc_info.value
    assert exc.code == "E_BUILD_SUMMARY_GENERATORS"
    assert exc.stage == "build_metrics"
    assert exc.field_path == "summary_generators[0]"

