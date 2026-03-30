from __future__ import annotations

import pytest

from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.environment.fake import FakeEnvironment
from gage_eval.agent_runtime.resources.bundle import ResourceBundle
from gage_eval.agent_runtime.schedulers.acp_client import AcpClientScheduler
from gage_eval.agent_runtime.schedulers.framework_loop import FrameworkLoopScheduler
from gage_eval.agent_runtime.schedulers.installed_client import InstalledClientScheduler
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.spec import AgentRuntimeSpec
from gage_eval.observability.trace import ObservabilityTrace


def _build_plan(*, scheduler: str = "installed_client") -> CompiledRuntimePlan:
    spec = AgentRuntimeSpec(
        agent_runtime_id=f"rt-{scheduler}",
        scheduler=scheduler,
        benchmark_kit_id="swebench",
        client_id="codex" if scheduler == "installed_client" else None,
    )
    return CompiledRuntimePlan(
        runtime_spec=spec,
        scheduler_type=scheduler,
        benchmark_kit_id=spec.benchmark_kit_id,
        client_id=spec.client_id,
        role_adapter_id=spec.role_adapter_id,
        environment_kind=spec.resource_policy.environment_kind,
    )


def _build_session(plan: CompiledRuntimePlan) -> AgentRuntimeSession:
    return AgentRuntimeSession(
        sample={"instruction": "fix the failing test", "messages": []},
        trace=ObservabilityTrace(),
        plan=plan,
        resources=ResourceBundle(environment=FakeEnvironment()),
        artifacts=ArtifactLayout.for_sample("runs", "run-1", "sample-1"),
    )


@pytest.mark.fast
def test_installed_client_scheduler_run_with_fake_env(monkeypatch) -> None:
    scheduler = InstalledClientScheduler(_build_plan())
    session = _build_session(_build_plan())

    class _FakeClient:
        def setup(self, environment, runtime_session) -> None:
            return None

        def run(self, request, environment):
            from gage_eval.agent_runtime.clients import ClientRunResult

            assert request.instruction == "fix the failing test"
            return ClientRunResult(
                exit_code=0,
                stdout="ok",
                stderr="",
                patch_path="/tmp/submission.patch",
                trajectory_path="/tmp/trajectory.json",
                artifacts={"stdout_path": "/tmp/stdout.log"},
            )

        def cleanup(self, environment, runtime_session) -> None:
            return None

    monkeypatch.setattr(scheduler, "_build_client", lambda: _FakeClient())
    monkeypatch.setattr(scheduler, "_resolve_kit_hook", lambda *args, **kwargs: None)

    result = scheduler.run(session)

    assert result.status == "success"
    assert result.patch_path == "/tmp/submission.patch"
    assert result.stdout_path == "/tmp/stdout.log"


@pytest.mark.fast
def test_framework_loop_scheduler_is_importable() -> None:
    scheduler = FrameworkLoopScheduler(_build_plan(scheduler="framework_loop"))

    assert scheduler is not None


@pytest.mark.fast
def test_framework_loop_scheduler_has_run_method() -> None:
    scheduler = FrameworkLoopScheduler(_build_plan(scheduler="framework_loop"))

    assert callable(scheduler.run)


@pytest.mark.fast
def test_acp_client_scheduler_raises_not_implemented() -> None:
    scheduler = AcpClientScheduler(_build_plan(scheduler="acp_client"))

    with pytest.raises(NotImplementedError):
        scheduler.run(_build_session(_build_plan(scheduler="acp_client")))
