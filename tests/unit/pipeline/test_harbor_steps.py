from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.step_contracts import (
    clear_step_contract_catalog_cache,
    collect_step_sequence_issues,
    get_step_contract_catalog,
)
from gage_eval.pipeline.steps.harbor import HarborJobHandle, HarborResultStep, HarborRunStep
from gage_eval.registry import import_asset_from_manifest, registry
from gage_eval.external_harness_kits.base import (
    TaskBatchHarnessHandle,
    TaskBatchHarnessPlan,
    TaskBatchHarnessRequest,
    TaskBatchHarnessResult,
)


class _Invocation:
    def __init__(self, tmp_path: Path) -> None:
        self.job_name = "gage_tb2"
        self.jobs_dir = tmp_path / "jobs"
        self.job_config_path = tmp_path / "harbor_job.json"
        self.job_config = {
            "environment": {
                "type": "docker",
                "api_key": "sk-test-secret",
                "safe": "visible",
            }
        }
        self.launcher_mode = "python_subprocess"
        self.launcher_argv = [
            "python",
            "-m",
            "gage_eval.external_harness_kits.harbor.launcher",
            "--config",
            str(self.job_config_path),
            "--result-file",
            str(tmp_path / "launcher_result.json"),
        ]
        self.workdir = tmp_path
        self.expected_total_trials = 1

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "job_name": self.job_name,
            "job_config": self.job_config,
            "launcher_argv": self.launcher_argv,
            "environ": {"OPENAI_API_KEY": "sk-test-secret"},
        }


class _Adapter:
    def __init__(self, tmp_path: Path) -> None:
        self.calls: list[str] = []
        self.invocation = _Invocation(tmp_path)

    def translate(self, request: TaskBatchHarnessRequest) -> TaskBatchHarnessPlan:
        self.calls.append("translate")
        assert request.payload["adapter_id"] == "harbor"
        return TaskBatchHarnessPlan(
            adapter_id=request.adapter_id,
            payload={"invocation": self.invocation},
        )

    def _initialize(self, plan: TaskBatchHarnessPlan) -> None:
        self.calls.append("initialize")
        assert plan.payload["invocation"] is self.invocation

    def launch(self, plan: TaskBatchHarnessPlan) -> TaskBatchHarnessHandle:
        self.calls.append("launch")
        return TaskBatchHarnessHandle(
            adapter_id=plan.adapter_id,
            payload={"plan": plan},
        )

    def poll_until_done(self, handle: TaskBatchHarnessHandle) -> TaskBatchHarnessResult:
        self.calls.append("poll_until_done")
        return TaskBatchHarnessResult(
            adapter_id=handle.adapter_id,
            payload={"job_dir": str(self.invocation.jobs_dir / self.invocation.job_name)},
        )

    def parse_results(self, result: TaskBatchHarnessResult):
        self.calls.append("parse_results")
        assert result.payload
        return ({"sample_id": "gpt2-codegolf"},)


class _Context:
    def __init__(self, tmp_path: Path, adapter: _Adapter | None = None) -> None:
        self.task_id = "tb2_one_case"
        self.dataset_id = "terminal_bench_2_0"
        self.adapter = adapter
        self.state: dict[str, Any] = {}
        self.trace = ObservabilityTrace(run_id="harbor-step-test")
        self.cache_store = EvalCache(base_dir=tmp_path, run_id="harbor-step-test")
        self.request_payload_calls = 0

    def get_task_batch_harness_adapter(self, adapter_id: str):
        if self.adapter is None:
            raise KeyError(
                f"Role adapter '{adapter_id}' is not registered as a task-batch harness adapter"
            )
        return self.adapter

    def request_payload(self, *, adapter_id: str) -> dict[str, Any]:
        self.request_payload_calls += 1
        return {"adapter_id": adapter_id}

    def store(self, key: str, value: Any) -> None:
        self.state[key] = value

    def load(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)


class _Writer:
    def __init__(self) -> None:
        self.records: list[Any] = []
        self.handle: HarborJobHandle | None = None

    def write(self, records, *, context, handle: HarborJobHandle):
        del context
        self.records.extend(records)
        self.handle = handle
        return {"written": len(records)}


@pytest.mark.fast
def test_registry_resolves_harbor_run_and_result() -> None:
    import_asset_from_manifest("pipeline_steps", "harbor_run", registry=registry)
    import_asset_from_manifest("pipeline_steps", "harbor_result", registry=registry)
    clear_step_contract_catalog_cache()

    assert registry.get("pipeline_steps", "harbor_run") is HarborRunStep
    assert registry.get("pipeline_steps", "harbor_result") is HarborResultStep
    catalog = get_step_contract_catalog()
    assert catalog.require("harbor_run").step_kind.value == "task"
    assert catalog.require("harbor_result").step_kind.value == "task"


@pytest.mark.fast
def test_harbor_run_calls_translate_launch_poll_until_done(tmp_path: Path) -> None:
    adapter = _Adapter(tmp_path)
    context = _Context(tmp_path, adapter)
    step = SimpleNamespace(adapter_id="harbor")

    result = HarborRunStep(adapter_id="harbor").execute_task(context, step=step)

    assert adapter.calls == ["translate", "initialize", "launch", "poll_until_done"]
    assert result["job_name"] == "gage_tb2"
    assert isinstance(context.load("harbor_job_handle"), HarborJobHandle)
    assert "sk-test-secret" not in json.dumps(result, default=str)


@pytest.mark.fast
def test_harbor_result_calls_parse_results_and_artifact_writer(tmp_path: Path) -> None:
    adapter = _Adapter(tmp_path)
    context = _Context(tmp_path, adapter)
    run_step = HarborRunStep(adapter_id="harbor")
    run_step.execute_task(context, step=SimpleNamespace(adapter_id="harbor"))
    writer = _Writer()

    result = HarborResultStep(
        adapter_id="harbor",
        artifact_writer_factory=lambda context: writer,
    ).execute_task(context, step=SimpleNamespace(adapter_id="harbor"))

    assert adapter.calls == ["translate", "initialize", "launch", "poll_until_done", "parse_results"]
    assert result["produced_sample_count"] == 1
    assert writer.records == [{"sample_id": "gpt2-codegolf"}]
    assert writer.handle == context.load("harbor_job_handle")


@pytest.mark.fast
def test_harbor_job_handle_round_trips_eight_redacted_fields(tmp_path: Path) -> None:
    handle = HarborJobHandle(
        job_name="gage_tb2",
        jobs_dir=tmp_path / "jobs",
        job_dir=tmp_path / "jobs" / "gage_tb2",
        job_config_path=tmp_path / "harbor_job.json",
        launcher_result_path=tmp_path / "launcher_result.json",
        workdir=tmp_path,
        environment={
            "type": "docker",
            "api_key": "sk-test-secret",
            "nested": {"token": "ak-nested-secret"},
        },
        invocation_metadata={
            "cmd": "api_key=sk-test-secret",
            "env": {"OPENAI_API_KEY": "sk-test-secret"},
        },
    )

    payload = handle.to_dict()
    serialized = json.dumps(payload, sort_keys=True)
    roundtrip = HarborJobHandle.from_dict(payload)

    assert set(payload) == {
        "job_name",
        "jobs_dir",
        "job_dir",
        "job_config_path",
        "launcher_result_path",
        "workdir",
        "environment",
        "invocation_metadata",
    }
    assert "sk-test-secret" not in serialized
    assert "ak-nested-secret" not in serialized
    assert payload["environment"]["api_key"] == "<redacted>"
    assert roundtrip.job_name == handle.job_name
    assert roundtrip.job_dir == handle.job_dir


@pytest.mark.fast
def test_harbor_result_does_not_read_raw_environments(tmp_path: Path) -> None:
    adapter = _Adapter(tmp_path)
    context = _Context(tmp_path, adapter)
    handle = HarborJobHandle.from_invocation(adapter.invocation)
    context.store("harbor_job_handle", handle)
    context.store(
        "harbor_task_batch_result",
        TaskBatchHarnessResult(adapter_id="harbor", payload={"handle": handle.to_dict()}),
    )

    HarborResultStep(
        adapter_id="harbor",
        artifact_writer_factory=lambda context: _Writer(),
    ).execute_task(
        context,
        step=SimpleNamespace(adapter_id="harbor"),
    )

    assert context.request_payload_calls == 0


@pytest.mark.fast
def test_missing_task_batch_adapter_has_clear_error(tmp_path: Path) -> None:
    context = _Context(tmp_path, adapter=None)

    with pytest.raises(KeyError, match="not registered as a task-batch harness adapter"):
        HarborRunStep(adapter_id="harbor").execute_task(
            context,
            step=SimpleNamespace(adapter_id="harbor"),
        )


@pytest.mark.fast
def test_harbor_task_steps_cannot_enter_sample_loop() -> None:
    import_asset_from_manifest("pipeline_steps", "harbor_run", registry=registry)
    clear_step_contract_catalog_cache()

    issues = collect_step_sequence_issues(
        [{"step": "harbor_run", "adapter_id": "harbor"}],
        owner_label="tasks[0].steps",
        adapter_ids=("harbor",),
    )

    assert len(issues) == 1
    assert issues[0].code == "invalid_step_kind"
    assert "actual kind 'task'" in issues[0].message
