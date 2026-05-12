from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from gage_eval.config.loader import load_pipeline_config_payload
from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan, SchedulerWorkflowBundle
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.executor import (
    AgentRuntimeSessionFactory,
    CompiledRuntimeExecutor,
)
from gage_eval.agent_runtime.failure_mapper import FailureMapper
from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.resources.manager import RuntimeLeaseBinding
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.spec import AgentRuntimeSpec
from gage_eval.agent_runtime.verifier.binding import JudgeBinding
from gage_eval.agent_runtime.verifier.contracts import (
    RuntimeJudgeOutcome,
    VerifierInput,
    VerifierResult,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(payload, dict)
    return payload


def load_lowered_pipeline_config(path: Path) -> dict[str, Any]:
    payload = load_pipeline_config_payload(path)
    assert payload["kind"] == "PipelineConfig"
    assert payload.get("role_adapters")
    return payload


def role_adapter_by_id(payload: dict[str, Any], adapter_id: str) -> dict[str, Any]:
    for role_adapter in payload.get("role_adapters") or []:
        if role_adapter.get("adapter_id") == adapter_id:
            return role_adapter
    raise AssertionError(f"role adapter not found: {adapter_id}")


class FakeResourceManager:
    def __init__(self, *, resource_kind: str = "docker") -> None:
        self.resource_kind = resource_kind
        self.acquired_trial_ids: list[str] = []
        self.released_lease_ids: list[str] = []
        self.acquire_resource_plans: list[dict[str, Any]] = []

    def acquire(self, session, *, resource_plan, trace=None, sample=None) -> RuntimeLeaseBinding:
        del trace, sample
        trial_id = str(session.runtime_context["trial_id"])
        self.acquired_trial_ids.append(trial_id)
        self.acquire_resource_plans.append(dict(resource_plan or {}))
        lease = ResourceLease(
            lease_id=f"{self.resource_kind}-lease-{len(self.acquired_trial_ids)}",
            resource_kind=self.resource_kind,  # type: ignore[arg-type]
            profile_id="profile",
            lifecycle="per_sample",
            metadata={"trial_id": trial_id},
        )
        return RuntimeLeaseBinding(resource_lease=lease)

    def release(self, binding: RuntimeLeaseBinding) -> None:
        assert binding.resource_lease is not None
        self.released_lease_ids.append(binding.resource_lease.lease_id)


class FakeScheduler:
    def __init__(self) -> None:
        self.trial_ids: list[str] = []

    async def arun(self, *, session, sample, payload, workflow_bundle, sandbox_provider) -> SchedulerResult:
        del sample, payload, workflow_bundle, sandbox_provider
        trial_id = str(session.runtime_context["trial_id"])
        self.trial_ids.append(trial_id)
        return SchedulerResult(
            scheduler_type="framework_loop",
            benchmark_kit_id=session.benchmark_kit_id,
            status="completed",
            agent_output={"answer": trial_id, "trial_id": trial_id},
            runtime_state={"trial_id": trial_id},
        )


class FakeVerifierRunner:
    def __init__(self, *, scores: dict[str, float] | None = None) -> None:
        self.scores = dict(scores or {"trial_0001": 0.0, "trial_0002": 1.0, "trial_0003": 1.0})
        self.trial_ids: list[str] = []

    def run(
        self,
        *,
        plan,
        session,
        sample,
        scheduler_result,
        sandbox_provider=None,
        environment_lease=None,
    ) -> RuntimeJudgeOutcome:
        del plan, sandbox_provider, environment_lease
        trial_id = str(session.runtime_context["trial_id"])
        self.trial_ids.append(trial_id)
        score = float(self.scores.get(trial_id, 0.0))
        payload = {"status": "completed", "score": score, "resolved": score >= 1.0}
        verifier_input = VerifierInput(
            benchmark_kit_id=session.benchmark_kit_id,
            scheduler_type=session.scheduler_type,
            sample_id=session.sample_id,
            sample=sample,
            scheduler_result=scheduler_result.to_dict(),
        )
        return RuntimeJudgeOutcome(
            verifier_input=verifier_input,
            verifier_result=VerifierResult(status="completed", payload=payload),
            judge_output=payload,
            persisted_path=session.artifact_layout["verifier_result"],
        )

    def build_failed_outcome(self, *, plan, session, sample, failure) -> RuntimeJudgeOutcome:
        del plan, sample
        verifier_input = VerifierInput(
            benchmark_kit_id=session.benchmark_kit_id,
            scheduler_type=session.scheduler_type,
            sample_id=session.sample_id,
            sample={},
            scheduler_result={},
        )
        payload = {"status": "failed", "score": 0.0, "resolved": False, "failure": failure.to_dict()}
        return RuntimeJudgeOutcome(
            verifier_input=verifier_input,
            verifier_result=VerifierResult(status="failed", payload=payload),
            judge_output=payload,
            persisted_path=session.artifact_layout["verifier_result"],
            failure=failure,
        )


class _RuntimeBootstrap:
    def bootstrap(self, *, session, sample, payload, sandbox_provider) -> dict[str, Any]:
        del sample, payload, sandbox_provider
        return {
            "runtime_context": {"trial_id": session.runtime_context["trial_id"]},
            "prompt_context": {},
            "benchmark_state": {},
            "scheduler_state": {},
        }


class _NoopTraceEmitter:
    def emit_session_start(self, *args, **kwargs) -> None:
        return None

    def emit_session_end(self, *args, **kwargs) -> None:
        return None

    def emit_failure(self, *args, **kwargs) -> None:
        return None


def build_fake_executor(
    tmp_path: Path,
    *,
    run_id: str,
    benchmark_kit_id: str,
    trial_policy: dict[str, Any],
    resource_manager: FakeResourceManager | None = None,
    scheduler: FakeScheduler | None = None,
    verifier: FakeVerifierRunner | None = None,
    verifier_environment_policy: str = "reuse",
    verifier_environment_profile_id: str | None = None,
    environment_provider: str = "docker",
) -> tuple[CompiledRuntimeExecutor, FakeResourceManager, FakeScheduler, FakeVerifierRunner]:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    manager = resource_manager or FakeResourceManager(resource_kind=environment_provider)
    scheduler_handle = scheduler or FakeScheduler()
    verifier_runner = verifier or FakeVerifierRunner()
    runtime_spec = AgentRuntimeSpec(
        agent_runtime_id=f"{benchmark_kit_id}_framework_loop",
        benchmark_kit_id=benchmark_kit_id,
        scheduler_type="framework_loop",
        client_id="demo-client",
    )
    plan = CompiledRuntimePlan(
        run_id=run_id,
        dut_id="dut-1",
        agent_id="agent-1",
        env_id="env-1",
        benchmark_id=f"{benchmark_kit_id}_benchmark",
        trial_policy=trial_policy,
        kit_id=benchmark_kit_id,
        kit_entry=None,
        kit_config={},
        agent_config={},
        scheduler_type="framework_loop",
        scheduler_config={},
        environment_provider=environment_provider,
        environment_profile_id="profile",
        environment_profile={},
        lifecycle="per_sample",
        provider_config={},
        startup_env={},
        resources={},
        verifier_environment_policy=verifier_environment_policy,
        verifier_environment_profile_id=verifier_environment_profile_id,
        workflow_bundle=SchedulerWorkflowBundle(
            bundle_id=f"{benchmark_kit_id}.framework_loop",
            benchmark_kit_id=benchmark_kit_id,
            scheduler_type="framework_loop",
        ),
        tool_registry=None,
        tool_provider_adapter=None,
        verifier_adapter=None,
        artifact_sink=sink,
        plan_id=f"{benchmark_kit_id}-plan",
        runtime_spec=runtime_spec,
        scheduler_handle=scheduler_handle,
        kit_runtime_ref=_RuntimeBootstrap(),
        judge_binding=JudgeBinding(judge_mode="runtime_verifier"),
        resource_plan={
            "resource_kind": environment_provider,
            "environment_profile": {
                "provider": environment_provider,
                "profile_id": "profile",
            },
            "provider_config": {},
        },
    )
    executor = CompiledRuntimeExecutor(
        compiled_plan=plan,
        resource_manager=manager,
        session_factory=AgentRuntimeSessionFactory(sink),
        verifier_runner=verifier_runner,
        artifact_sink=sink,
        trace_emitter=_NoopTraceEmitter(),
        failure_mapper=FailureMapper(),
    )
    return executor, manager, scheduler_handle, verifier_runner


def flatten_samples_projection(projection: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in projection.items():
        if isinstance(value, dict) and "value" in value and "source_trial_id" in value:
            flattened[key] = value["value"]
            flattened[f"{key}_source_trial_id"] = value["source_trial_id"]
        else:
            flattened[key] = value
    return flattened


def write_samples_jsonl(run_dir: Path, record: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "samples.jsonl"
    target.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return target


def trial_root(base_dir: Path, *, run_id: str, task_id: str, sample_id: str, trial_id: str) -> Path:
    return base_dir / run_id / "artifacts" / task_id / sample_id / "trials" / trial_id
