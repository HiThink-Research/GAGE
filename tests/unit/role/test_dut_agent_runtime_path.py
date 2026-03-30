from __future__ import annotations

import pytest

from gage_eval.agent_runtime.schedulers import SchedulerResult
from gage_eval.agent_runtime.spec import AgentRuntimeSpec
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter
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
        agent_runtime_resolver=_FakeResolver(),
        agent_runtime_id="runtime-1",
    )

    result = adapter.invoke(
        {
            "sample": {"instruction": "fix the failing test", "instance_id": "sample-1"},
            "trace": ObservabilityTrace(),
        },
        RoleAdapterState(),
    )

    assert result["status"] == "success"
    assert result["answer"] == "patched"
    assert result["patch_path"] == "/tmp/submission.patch"
