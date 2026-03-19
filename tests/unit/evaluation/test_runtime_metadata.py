from gage_eval.config.pipeline_config import BackendSpec, ModelSpec, PipelineConfig, RoleAdapterSpec
from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.runtime_builder import _record_config_metadata
from gage_eval.evaluation.runtime_metadata import (
    RUN_METADATA_SCHEMA_VERSION,
    RUNTIME_METADATA_SCHEMA_VERSION,
    build_run_metadata_snapshot,
    build_runtime_metadata_snapshot,
    record_run_metadata,
    record_runtime_metadata,
)
from gage_eval.utils.run_identity import build_run_identity


def _make_config() -> PipelineConfig:
    return PipelineConfig(
        metadata={"name": "demo"},
        backends=(BackendSpec(backend_id="b1", type="openai", config={"model": "gpt"}),),
        models=(ModelSpec(model_id="m1", source="openai", hub="hf", hub_params={"rev": "main"}, params={"temperature": 0.1}),),
        role_adapters=(
            RoleAdapterSpec(
                adapter_id="dut",
                role_type="dut_model",
                backend_id="b1",
                backend={"type": "inline", "config": {"x": 1}},
                capabilities=("chat", "tool_use"),
                prompt_id="prompt-main",
                params={"ignored": True},
            ),
        ),
        summary_generators=("arena", "tau2"),
    )


def test_build_runtime_metadata_snapshot_projects_stable_fields() -> None:
    snapshot = build_runtime_metadata_snapshot(_make_config())

    assert snapshot.schema_version == RUNTIME_METADATA_SCHEMA_VERSION
    assert snapshot.pipeline_id == "demo"
    assert snapshot.backends == (
        {"backend_id": "b1", "type": "openai", "config": {"model": "gpt"}},
    )
    assert snapshot.models == (
        {
            "model_id": "m1",
            "source": "openai",
            "hub": "hf",
            "hub_params": {"rev": "main"},
            "params": {"temperature": 0.1},
        },
    )
    assert snapshot.role_adapters == (
        {
            "adapter_id": "dut",
            "role_type": "dut_model",
            "backend_id": "b1",
            "backend_inline": {"type": "inline", "config": {"x": 1}},
            "capabilities": ["chat", "tool_use"],
            "prompt_id": "prompt-main",
        },
    )
    assert snapshot.summary_generators == ("arena", "tau2")


def test_record_runtime_metadata_writes_legacy_keys_and_schema_version(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="runtime-metadata")
    snapshot = build_runtime_metadata_snapshot(_make_config())

    record_runtime_metadata(cache, snapshot)

    assert cache.get_metadata("runtime_metadata_schema_version") == RUNTIME_METADATA_SCHEMA_VERSION
    assert cache.get_metadata("backends") == list(snapshot.backends)
    assert cache.get_metadata("models") == list(snapshot.models)
    assert cache.get_metadata("role_adapters") == list(snapshot.role_adapters)
    assert cache.get_metadata("summary_generators") == ["arena", "tau2"]


def test_record_config_metadata_uses_shared_runtime_snapshot(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="runtime-metadata-shared")
    config = _make_config()

    _record_config_metadata(config, cache)

    assert cache.get_metadata("runtime_metadata_schema_version") == RUNTIME_METADATA_SCHEMA_VERSION
    assert cache.get_metadata("backends")[0]["backend_id"] == "b1"
    assert cache.get_metadata("role_adapters")[0]["adapter_id"] == "dut"


def test_record_run_metadata_writes_identity_payload(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="runtime-run-metadata")
    identity = build_run_identity("run-20260319010101-ab12cd34")
    snapshot = build_run_metadata_snapshot(identity)

    record_run_metadata(cache, snapshot)

    assert cache.get_metadata("run_metadata_schema_version") == RUN_METADATA_SCHEMA_VERSION
    assert cache.get_metadata("run_identity") == {
        "run_id": "run-20260319010101-ab12cd34",
        "source": "provided",
        "schema_version": 1,
        "created_at_iso": identity.created_at_iso,
    }
