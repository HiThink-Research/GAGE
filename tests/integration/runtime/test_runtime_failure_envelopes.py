from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

from gage_eval.agent_runtime import build_compiled_runtime_executor, compile_agent_runtime_plan
from gage_eval.agent_runtime.executor import DefaultVerifierRunner
from gage_eval.role.agent.backends.demo_agent import RuntimeSmokeAgent


class _InstalledClientStub:
    def invoke(self, payload: dict) -> dict:
        return {
            "answer": "done",
            "agent_trace": [],
            "artifact_paths": {"stdout": "stdout.txt"},
        }


class _ExplodingInstalledClientStub:
    def invoke(self, payload: dict) -> dict:
        raise RuntimeError("client boom")


class _ExplodingResourceManager:
    def acquire(self, session, *, resource_plan, trace=None):
        raise RuntimeError("lease boom")

    def release(self, binding) -> None:
        return None


class _ExplodingVerifierRunner(DefaultVerifierRunner):
    def run(self, *, plan, session, sample, scheduler_result, sandbox_provider):
        raise RuntimeError("verifier boom")


class _ExplodingCleanupResourceManager:
    def __init__(self, delegate) -> None:
        self._delegate = delegate

    def acquire(self, session, *, resource_plan, trace=None):
        return self._delegate.acquire(session, resource_plan=resource_plan, trace=trace)

    def release(self, binding) -> None:
        raise RuntimeError("cleanup boom")


class _FlakyArtifactSink:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self._verifier_failures = 0

    def build_layout(self, *, run_id: str, task_id: str, sample_id: str) -> dict[str, str]:
        return self._delegate.build_layout(run_id=run_id, task_id=task_id, sample_id=sample_id)

    def persist_runtime_metadata(self, *, session, scheduler_result=None, failure=None) -> str:
        return self._delegate.persist_runtime_metadata(
            session=session,
            scheduler_result=scheduler_result,
            failure=failure,
        )

    def persist_verifier_result(self, outcome) -> str:
        if self._verifier_failures == 0:
            self._verifier_failures += 1
            raise OSError("disk full")
        return self._delegate.persist_verifier_result(outcome)

    def persist_raw_error(self, *, session, error: BaseException) -> str:
        return self._delegate.persist_raw_error(session=session, error=error)


def _sample() -> dict:
    return {
        "id": "terminal-1",
        "instruction": "say done",
        "expected_answer": "done",
        "messages": [{"role": "user", "content": "say done"}],
    }


def _payload() -> dict:
    return {
        "sample": _sample(),
        "execution_context": {
            "run_id": "runtime-failure-run",
            "task_id": "runtime-failure-task",
            "sample_id": "terminal-1",
        },
    }


def _run_executor(executor) -> dict:
    return asyncio.run(
        executor.aexecute(
            sample=_sample(),
            payload=_payload(),
        )
    )


def _assert_failure_result(result: dict, *, domain: str, stage: str) -> None:
    failure = result["runtime_failure"]
    assert failure["failure_domain"] == domain
    assert failure["failure_stage"] == stage
    assert failure["retryable"] is False
    assert failure["suspect_files"]

    verifier_path = Path(result["runtime_session"]["verifier_result_path"])
    assert verifier_path.exists()
    verifier_payload = json.loads(verifier_path.read_text(encoding="utf-8"))
    assert verifier_payload["judge_output"]["failure_domain"] == domain
    assert verifier_payload["judge_output"]["failure_reason"] == failure["failure_code"]

    raw_error_path = failure.get("raw_error_path")
    assert raw_error_path
    assert Path(raw_error_path).exists()


def _build_installed_client_executor():
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")
    plan = replace(
        plan,
        resource_plan={"resource_kind": "docker", "sandbox_config": {}},
    )
    return build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=_InstalledClientStub(),
        max_turns=4,
    )


def _build_framework_loop_executor(*, workflow_bundle=None):
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_framework_loop")
    plan = replace(
        plan,
        resource_plan={"resource_kind": "docker", "sandbox_config": {}},
    )
    if workflow_bundle is not None:
        plan = replace(plan, workflow_bundle=workflow_bundle)
    return build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=RuntimeSmokeAgent(),
        max_turns=4,
    )


def test_runtime_executor_maps_environment_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    executor = _build_installed_client_executor()
    executor.resource_manager = _ExplodingResourceManager()

    result = _run_executor(executor)

    _assert_failure_result(result, domain="environment", stage="acquire_lease")


def test_runtime_executor_maps_input_projection_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_framework_loop")
    failing_bundle = replace(
        plan.workflow_bundle,
        build_loop_inputs=lambda **_: (_ for _ in ()).throw(RuntimeError("input boom")),
    )
    executor = _build_framework_loop_executor(workflow_bundle=failing_bundle)

    result = _run_executor(executor)

    _assert_failure_result(result, domain="input_projection", stage="prepare_inputs")


def test_runtime_executor_maps_client_execution_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")
    plan = replace(
        plan,
        resource_plan={"resource_kind": "docker", "sandbox_config": {}},
    )
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=_ExplodingInstalledClientStub(),
        max_turns=4,
    )

    result = _run_executor(executor)

    _assert_failure_result(result, domain="client_execution", stage="run_scheduler")


def test_runtime_executor_maps_artifact_capture_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")
    failing_bundle = replace(
        plan.workflow_bundle,
        finalize_result=lambda **_: (_ for _ in ()).throw(RuntimeError("artifact boom")),
    )
    plan = replace(
        plan,
        workflow_bundle=failing_bundle,
        resource_plan={"resource_kind": "docker", "sandbox_config": {}},
    )
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=_InstalledClientStub(),
        max_turns=4,
    )

    result = _run_executor(executor)

    _assert_failure_result(result, domain="artifact_capture", stage="normalize_result")


def test_runtime_executor_maps_verifier_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    executor = _build_installed_client_executor()
    executor.verifier_runner = _ExplodingVerifierRunner()

    result = _run_executor(executor)

    _assert_failure_result(result, domain="verifier", stage="run_verifier")


def test_runtime_executor_maps_persistence_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    executor = _build_installed_client_executor()
    executor.artifact_sink = _FlakyArtifactSink(executor.artifact_sink)

    result = _run_executor(executor)

    _assert_failure_result(result, domain="persistence", stage="persist_outputs")


def test_runtime_executor_maps_cleanup_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    executor = _build_installed_client_executor()
    executor.resource_manager = _ExplodingCleanupResourceManager(executor.resource_manager)

    result = _run_executor(executor)

    _assert_failure_result(result, domain="persistence", stage="cleanup")
