from __future__ import annotations

import json
from pathlib import Path
import pytest

from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.agent_runtime.schedulers import SchedulerResult
from gage_eval.agent_runtime.spec import AgentRuntimeSpec
from gage_eval.agent_runtime.verifier.base import VerifierInput
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter, _persist_runtime_verifier_result
from gage_eval.observability.trace import ObservabilityTrace


class _FakeScheduler:
    def run(self, session):
        return SchedulerResult(
            status="success",
            answer="patched",
            patch_path="/tmp/submission.patch",
            stdout_path="/tmp/stdout.log",
            trajectory_path="/tmp/trajectory.json",
            artifacts={"report": "/tmp/report.json"},
            metrics={"score": 1.0},
        )


class _FakeResolver:
    def resolve(self, runtime_id: str):
        return type(
            "Plan",
            (),
            {
                "runtime_spec": AgentRuntimeSpec(
                    agent_runtime_id=runtime_id,
                    scheduler="installed_client",
                    benchmark_kit_id="swebench",
                )
            },
        )()

    def build_scheduler(self, plan):
        return _FakeScheduler()


@pytest.mark.fast
def test_dut_agent_uses_agent_runtime_path(monkeypatch) -> None:
    class _FakeEnvironment:
        pass

    captured = {}

    class _CapturingScheduler(_FakeScheduler):
        def run(self, session):
            captured["artifacts"] = session.artifacts
            return super().run(session)

    class _CapturingResolver(_FakeResolver):
        def build_scheduler(self, plan):
            return _CapturingScheduler()

    from gage_eval.agent_runtime.environment import provider as provider_module

    monkeypatch.setattr(
        provider_module.EnvironmentProvider,
        "build",
        lambda self, plan, sample: _FakeEnvironment(),
    )
    adapter = DUTAgentAdapter(
        adapter_id="dut-1",
        role_type="dut_agent",
        capabilities=(),
        agent_runtime_resolver=_CapturingResolver(),
        agent_runtime_id="runtime-1",
    )

    result = adapter.invoke(
        {
            "task_id": "task/demo",
            "sample": {"instruction": "fix the failing test", "instance_id": "sample-1"},
            "trace": ObservabilityTrace(),
        },
        RoleAdapterState(),
    )

    assert result["status"] == "success"
    assert result["answer"] == "patched"
    assert result["patch_path"] == "/tmp/submission.patch"
    assert captured["artifacts"].task_dir.endswith("/samples/task_demo")
    assert captured["artifacts"].sample_file.endswith("/samples/task_demo/sample-1/sample.json")


@pytest.mark.fast
def test_dut_agent_uses_sample_gage_task_id_for_artifact_path(monkeypatch) -> None:
    class _FakeEnvironment:
        pass

    captured = {}

    class _CapturingScheduler(_FakeScheduler):
        def run(self, session):
            captured["artifacts"] = session.artifacts
            return super().run(session)

    class _CapturingResolver(_FakeResolver):
        def build_scheduler(self, plan):
            return _CapturingScheduler()

    from gage_eval.agent_runtime.environment import provider as provider_module

    monkeypatch.setattr(
        provider_module.EnvironmentProvider,
        "build",
        lambda self, plan, sample: _FakeEnvironment(),
    )
    adapter = DUTAgentAdapter(
        adapter_id="dut-2",
        role_type="dut_agent",
        capabilities=(),
        agent_runtime_resolver=_CapturingResolver(),
        agent_runtime_id="runtime-2",
    )

    result = adapter.invoke(
        {
            "sample": {
                "instruction": "fix the failing test",
                "instance_id": "sample-2",
                "_gage_task_id": "task/from-sample",
            },
            "trace": ObservabilityTrace(),
        },
        RoleAdapterState(),
    )

    assert result["status"] == "success"
    assert captured["artifacts"].task_dir.endswith("/samples/task_from-sample")
    assert captured["artifacts"].sample_file.endswith("/samples/task_from-sample/sample-2/sample.json")


@pytest.mark.fast
def test_dut_agent_runtime_writes_swebench_eval_result(monkeypatch) -> None:
    class _FakeEnvironment:
        pass

    captured = {}

    class _SwebenchScheduler:
        def run(self, session):
            captured["artifacts"] = session.artifacts
            return SchedulerResult(
                status="success",
                answer="diff --git a/answer.py b/answer.py\n+    return 42\n",
                patch_path="/tmp/submission.patch",
                stdout_path="/tmp/stdout.log",
                trajectory_path="/tmp/trajectory.json",
                raw_output={"patch_content": "diff --git a/answer.py b/answer.py\n+    return 42\n"},
            )

    class _SwebenchResolver(_FakeResolver):
        def resolve(self, runtime_id: str):
            return type(
                "Plan",
                (),
                {
                    "runtime_spec": AgentRuntimeSpec(
                        agent_runtime_id=runtime_id,
                        scheduler="installed_client",
                        benchmark_kit_id="swebench",
                    ),
                    "benchmark_kit_id": "swebench",
                    "params": {"verifier": {"mode": "patch_presence"}},
                },
            )()

        def build_scheduler(self, plan):
            return _SwebenchScheduler()

    from gage_eval.agent_runtime.environment import provider as provider_module

    monkeypatch.setattr(
        provider_module.EnvironmentProvider,
        "build",
        lambda self, plan, sample: _FakeEnvironment(),
    )
    adapter = DUTAgentAdapter(
        adapter_id="dut-swebench",
        role_type="dut_agent",
        capabilities=(),
        agent_runtime_resolver=_SwebenchResolver(),
        agent_runtime_id="runtime-swebench",
    )
    sample = {
        "instruction": "Change answer.py so compute_answer returns 42.",
        "instance_id": "swebench__smoke_1",
        "metadata": {"expected_patch_contains": ["return 42"]},
    }

    result = adapter.invoke({"sample": sample, "trace": ObservabilityTrace()}, RoleAdapterState())

    assert result["status"] == "success"
    assert result["verifier_result"]["status"] == "pass"
    assert sample["eval_result"]["status"] == "pass"
    persisted = json.loads(Path(captured["artifacts"].verifier_result_file).read_text(encoding="utf-8"))
    assert persisted["status"] == "pass"


@pytest.mark.fast
def test_dut_agent_runtime_writes_skillsbench_eval_result(monkeypatch) -> None:
    class _FakeEnvironment:
        pass

    captured = {}

    class _SkillsBenchScheduler:
        def run(self, session):
            captured["artifacts"] = session.artifacts
            return SchedulerResult(
                status="success",
                answer="done",
                patch_path="/tmp/submission.patch",
                stdout_path="/tmp/stdout.log",
                trajectory_path="/tmp/trajectory.json",
                artifacts={"agent_workspace_dir": "/tmp/workspace"},
                raw_output={"answer": "done"},
            )

    class _SkillsBenchResolver(_FakeResolver):
        def resolve(self, runtime_id: str):
            return type(
                "Plan",
                (),
                {
                    "runtime_spec": AgentRuntimeSpec(
                        agent_runtime_id=runtime_id,
                        scheduler="installed_client",
                        benchmark_kit_id="skillsbench",
                    ),
                    "benchmark_kit_id": "skillsbench",
                    "params": {"verifier": {}},
                },
            )()

        def build_scheduler(self, plan):
            return _SkillsBenchScheduler()

    from gage_eval.agent_runtime.environment import provider as provider_module
    import gage_eval.role.judge.skillsbench_evaluate as skillsbench_module

    monkeypatch.setattr(
        provider_module.EnvironmentProvider,
        "build",
        lambda self, plan, sample: _FakeEnvironment(),
    )
    monkeypatch.setattr(
        skillsbench_module.SkillsBenchEvaluate,
        "invoke",
        lambda self, payload, state=None: {"resolved": True, "score": 1.0, "failure_reason": None},
    )
    adapter = DUTAgentAdapter(
        adapter_id="dut-skillsbench",
        role_type="dut_agent",
        capabilities=(),
        agent_runtime_resolver=_SkillsBenchResolver(),
        agent_runtime_id="runtime-skillsbench",
    )
    sample = {
        "instruction": "Complete the task.",
        "instance_id": "skillsbench__1",
        "metadata": {"skillsbench": {"task_id": "skillsbench__1"}},
    }

    result = adapter.invoke({"sample": sample, "trace": ObservabilityTrace()}, RoleAdapterState())

    assert result["status"] == "success"
    assert result["verifier_result"]["status"] == "pass"
    assert sample["eval_result"]["status"] == "pass"
    persisted = json.loads(Path(captured["artifacts"].verifier_result_file).read_text(encoding="utf-8"))
    assert persisted["status"] == "pass"


@pytest.mark.fast
def test_persist_runtime_verifier_result_serializes_dataclass_payload(tmp_path) -> None:
    artifacts = ArtifactLayout.for_sample(str(tmp_path), "run-1", "sample-1")

    _persist_runtime_verifier_result(
        artifacts,
        {
            "status": "failed",
            "summary": "serialization smoke",
            "raw_output": {
                "verifier_input": VerifierInput(
                    benchmark_kit_id="terminal_bench",
                    sample_id="sample-1",
                    payload={"scheduler_result": {"status": "error"}},
                )
            },
        },
    )

    persisted = json.loads(Path(artifacts.verifier_result_file).read_text(encoding="utf-8"))
    assert persisted["raw_output"]["verifier_input"]["sample_id"] == "sample-1"


@pytest.mark.fast
def test_persist_runtime_verifier_result_handles_recursive_payload(tmp_path) -> None:
    artifacts = ArtifactLayout.for_sample(str(tmp_path), "run-1", "sample-1")
    payload = {"status": "pass"}
    payload["self"] = payload

    _persist_runtime_verifier_result(artifacts, payload)

    persisted = json.loads(Path(artifacts.verifier_result_file).read_text(encoding="utf-8"))
    assert persisted["status"] == "pass"
    assert persisted["self"] == "<recursive_ref>"


@pytest.mark.fast
def test_dut_agent_runtime_writes_terminal_bench_eval_result(monkeypatch) -> None:
    class _FakeEnvironment:
        pass

    class _TerminalResolver(_FakeResolver):
        def resolve(self, runtime_id: str):
            return type(
                "Plan",
                (),
                {
                    "runtime_spec": AgentRuntimeSpec(
                        agent_runtime_id=runtime_id,
                        scheduler="installed_client",
                        benchmark_kit_id="terminal_bench",
                    ),
                    "benchmark_kit_id": "terminal_bench",
                },
            )()

    from gage_eval.agent_runtime.environment import provider as provider_module

    monkeypatch.setattr(
        provider_module.EnvironmentProvider,
        "build",
        lambda self, plan, sample: _FakeEnvironment(),
    )
    adapter = DUTAgentAdapter(
        adapter_id="dut-terminal",
        role_type="dut_agent",
        capabilities=(),
        agent_runtime_resolver=_TerminalResolver(),
        agent_runtime_id="runtime-terminal",
    )
    sample = {"instruction": "Create hello.txt", "instance_id": "tb2__smoke_1"}

    result = adapter.invoke({"sample": sample, "trace": ObservabilityTrace()}, RoleAdapterState())

    assert result["status"] == "success"
    assert result["verifier_result"]["status"] == "passed"
    assert sample["eval_result"]["status"] == "passed"


@pytest.mark.fast
def test_dut_agent_runtime_writes_appworld_eval_result(monkeypatch) -> None:
    class _FakeEnvironment:
        pass

    class _AppWorldScheduler:
        def run(self, session):
            return SchedulerResult(
                status="success",
                stdout_path="/tmp/stdout.log",
                raw_output={"runtime_handle": {"container_name": "appworld-smoke"}},
            )

    class _AppWorldResolver(_FakeResolver):
        def resolve(self, runtime_id: str):
            return type(
                "Plan",
                (),
                {
                    "runtime_spec": AgentRuntimeSpec(
                        agent_runtime_id=runtime_id,
                        scheduler="installed_client",
                        benchmark_kit_id="appworld",
                    ),
                    "benchmark_kit_id": "appworld",
                    "params": {},
                },
            )()

        def build_scheduler(self, plan):
            return _AppWorldScheduler()

    from gage_eval.agent_runtime.environment import provider as provider_module
    from gage_eval.role.judge import appworld_evaluate as appworld_module

    monkeypatch.setattr(
        provider_module.EnvironmentProvider,
        "build",
        lambda self, plan, sample: _FakeEnvironment(),
    )
    monkeypatch.setattr(
        appworld_module.AppWorldEvaluate,
        "invoke",
        lambda self, payload, state=None: {"appworld": {"task_id": "task-1", "tgc": 1.0, "sgc": 0.5}},
    )
    adapter = DUTAgentAdapter(
        adapter_id="dut-appworld",
        role_type="dut_agent",
        capabilities=(),
        agent_runtime_resolver=_AppWorldResolver(),
        agent_runtime_id="runtime-appworld",
    )
    sample = {
        "id": "calendar_001",
        "metadata": {"appworld": {"task_id": "task-1", "subset": "dev"}},
    }

    result = adapter.invoke({"sample": sample, "trace": ObservabilityTrace()}, RoleAdapterState())

    assert result["status"] == "success"
    assert result["verifier_result"]["status"] == "pass"
    assert result["verifier_result"]["appworld"]["tgc"] == 1.0
    assert sample["eval_result"]["appworld"]["sgc"] == 0.5


@pytest.mark.fast
def test_dut_agent_runtime_writes_tau2_eval_result(monkeypatch) -> None:
    class _FakeEnvironment:
        pass

    class _Tau2Scheduler:
        def run(self, session):
            return SchedulerResult(
                status="success",
                raw_output={"runtime_state": {"task_id": "task-1", "domain": "airline", "reward": 1.0}},
            )

    class _Tau2Resolver(_FakeResolver):
        def resolve(self, runtime_id: str):
            return type(
                "Plan",
                (),
                {
                    "runtime_spec": AgentRuntimeSpec(
                        agent_runtime_id=runtime_id,
                        scheduler="installed_client",
                        benchmark_kit_id="tau2",
                    ),
                    "benchmark_kit_id": "tau2",
                    "params": {},
                },
            )()

        def build_scheduler(self, plan):
            return _Tau2Scheduler()

    from gage_eval.agent_runtime.environment import provider as provider_module
    from gage_eval.role.judge import tau2_eval as tau2_module

    monkeypatch.setattr(
        provider_module.EnvironmentProvider,
        "build",
        lambda self, plan, sample: _FakeEnvironment(),
    )
    monkeypatch.setattr(
        tau2_module.Tau2Evaluate,
        "invoke",
        lambda self, payload, state=None: {
            "tau2": {"task_id": "task-1", "domain": "airline", "reward": 1.0, "termination_reason": "user_stop"}
        },
    )
    adapter = DUTAgentAdapter(
        adapter_id="dut-tau2",
        role_type="dut_agent",
        capabilities=(),
        agent_runtime_resolver=_Tau2Resolver(),
        agent_runtime_id="runtime-tau2",
    )
    sample = {
        "id": "airline_task-1__trial_0",
        "metadata": {"tau2": {"task_id": "task-1", "domain": "airline"}},
    }

    result = adapter.invoke({"sample": sample, "trace": ObservabilityTrace()}, RoleAdapterState())

    assert result["status"] == "success"
    assert result["verifier_result"]["status"] == "pass"
    assert result["verifier_result"]["tau2"]["reward"] == 1.0
    assert sample["eval_result"]["tau2"]["termination_reason"] == "user_stop"
