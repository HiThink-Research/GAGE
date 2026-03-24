from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.assets.datasets.manager import DataManager, DataSource
from gage_eval.config.pipeline_config import (
    CustomPipelineStep,
    DatasetSpec,
    PipelineConfig,
    RoleAdapterSpec,
    TaskSpec,
)
from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.runtime_builder import (
    TaskOrchestratorRuntime,
    _prepare_task_entries,
    _record_config_metadata,
)
from gage_eval.evaluation.task_plan import build_task_plan_specs
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.reporting.recorders import InMemoryRecorder
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager

SEEN_SAMPLE_IDS: list[str] = []


class _EchoRole:
    def __init__(self, adapter_id: str, role_type: str, **kwargs) -> None:  # noqa: ARG002
        self.adapter_id = adapter_id
        self.role_type = role_type
        self.resource_requirement = {}

    def clone_for_sample(self):
        return self

    def invoke(self, payload, state=None):  # noqa: ARG002
        sample = payload.get("sample", {})
        sample_id = str(sample.get("id"))
        SEEN_SAMPLE_IDS.append(sample_id)
        return {
            "answer": f"echo-{sample_id}",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": f"echo-{sample_id}"}],
            },
        }


def _make_runtime(tmp_path: Path) -> tuple[TaskOrchestratorRuntime, EvalCache]:
    SEEN_SAMPLE_IDS.clear()
    dataset = DataSource(
        dataset_id="ds",
        records=[
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "missing id"}]}
                ],
                "choices": [],
            },
            "invalid raw record",
            {
                "id": "explicit-1",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "explicit"}]}
                ],
                "choices": [],
            },
        ],
        metadata={"path": str(tmp_path / "ds.jsonl")},
        validation={"mode": "warn"},
    )
    dm = DataManager()
    dm.register_source(dataset)

    config = PipelineConfig(
        datasets=(DatasetSpec(dataset_id="ds", loader="jsonl"),),
        role_adapters=(
            RoleAdapterSpec(
                adapter_id="dut",
                role_type="dut_model",
                class_path="tests.integration.runtime.evaluation.test_sample_validation_summary._EchoRole",
            ),
        ),
        tasks=(
            TaskSpec(
                task_id="task_a",
                dataset_id="ds",
                steps=(CustomPipelineStep(step_type="inference", adapter_id="dut"),),
            ),
        ),
    )
    plans = build_task_plan_specs(config)
    cache = EvalCache(base_dir=tmp_path, run_id="validation-summary-test")
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="validation-trace"))
    report_step = ReportStep(auto_eval_step=None, cache_store=cache)
    entries = _prepare_task_entries(
        task_plans=plans,
        config=config,
        data_manager=dm,
        datasets={"ds": dataset},
        trace=trace,
        cache_store=cache,
        resource_profile=ResourceProfile(
            [NodeResource(node_id="local", gpus=0, cpus=1)]
        ),
        sandbox_profiles={},
    )

    role_manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))
    role_manager.register_role_adapter("dut", _EchoRole("dut", "dut_model"))

    runtime = TaskOrchestratorRuntime(entries, role_manager, trace, report_step)
    _record_config_metadata(config, cache, trace=trace)
    return runtime, cache


def test_task_runtime_reports_validation_summary_and_synthetic_sample_ids(
    tmp_path: Path,
) -> None:
    runtime, cache = _make_runtime(tmp_path)

    runtime.run()

    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert SEEN_SAMPLE_IDS == ["task_a:0", "explicit-1"]
    assert summary["samples_total"] == 3
    assert summary["samples_valid"] == 2
    assert summary["samples_dropped"] == 1
    assert summary["samples_drop_ratio"] == pytest.approx(1 / 3)
    assert summary["drop_reasons_top"] == [
        {"reason": "raw_record:invalid_record_type", "count": 1}
    ]
    assert summary["validation_gate_triggered"] is False
