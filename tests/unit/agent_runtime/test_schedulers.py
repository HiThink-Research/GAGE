from __future__ import annotations

import json
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
from gage_eval.agent_runtime.verifier.base import VerifierInput
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.sandbox.surfaces import ClientSurface


def _build_plan(*, scheduler: str = "installed_client") -> CompiledRuntimePlan:
    spec = AgentRuntimeSpec(
        agent_runtime_id=f"rt-{scheduler}",
        scheduler=scheduler,
        benchmark_kit_id="swebench",
        client_id="codex" if scheduler == "installed_client" else None,
        params={"client_default_args": ["--dangerously-bypass-approvals-and-sandbox"]}
        if scheduler == "installed_client"
        else {},
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
def test_installed_client_scheduler_run_with_fake_env(monkeypatch, tmp_path) -> None:
    scheduler = InstalledClientScheduler(_build_plan())
    session = AgentRuntimeSession(
        sample={"instruction": "fix the failing test", "messages": []},
        trace=ObservabilityTrace(),
        plan=_build_plan(),
        resources=ResourceBundle(environment=FakeEnvironment()),
        artifacts=ArtifactLayout.for_sample(str(tmp_path), "run-1", "sample-1"),
    )

    class _FakeClient:
        def setup(self, environment, runtime_session) -> None:
            return None

        def run(self, request, environment):
            from gage_eval.agent_runtime.clients import ClientRunResult

            assert request.instruction == "fix the failing test"
            assert request.metadata["patch_path"].endswith("submission.patch")
            assert request.metadata["stdout_path"].endswith("stdout.log")
            assert request.metadata["trajectory_path"].endswith("trajectory.json")
            assert request.metadata["artifacts"]["patch_path"].endswith("submission.patch")
            return ClientRunResult(
                exit_code=0,
                stdout="ok",
                stderr="warn",
                patch_path="/tmp/submission.patch",
                patch_content="diff --git a/foo b/foo\n",
                trajectory_path="/tmp/trajectory.json",
                artifacts={"stdout_path": "/tmp/stdout.log"},
            )

        def cleanup(self, environment, runtime_session) -> None:
            return None

    def _resolve_kit_hook(_module_name, attr_name):
        if attr_name != "finalize_result":
            return None

        def _finalize(_sample, result, _artifacts):
            return {
                "verifier_input": VerifierInput(
                    benchmark_kit_id="swebench",
                    sample_id="sample-1",
                    payload={"scheduler_result": {"status": result.status}},
                )
            }

        return _finalize

    monkeypatch.setattr(scheduler, "_build_client", lambda: _FakeClient())
    monkeypatch.setattr(scheduler, "_resolve_kit_hook", _resolve_kit_hook)

    result = scheduler.run(session)

    assert result.status == "success"
    assert result.answer.startswith("diff --git")
    assert result.patch_path == "/tmp/submission.patch"
    assert result.stdout_path == "/tmp/stdout.log"
    metadata = json.loads(tmp_path.joinpath("run-1/samples/global/sample-1/runtime_metadata.json").read_text(encoding="utf-8"))
    assert metadata["scheduler_result"]["status"] == "success"
    assert metadata["scheduler_result"]["raw_output"]["verifier_input"]["sample_id"] == "sample-1"
    assert metadata["artifact_layout"]["sample_file"].endswith("/samples/global/sample-1/sample.json")
    assert tmp_path.joinpath("run-1/samples/global/sample-1/agent/stderr.log").read_text(encoding="utf-8") == "warn"
    assert tmp_path.joinpath("run-1/samples/global/sample-1/agent/final_message.md").read_text(encoding="utf-8").startswith("diff --git")


@pytest.mark.fast
def test_installed_client_scheduler_captures_runtime_handle_and_surfaces(monkeypatch) -> None:
    scheduler = InstalledClientScheduler(_build_plan())

    class _RemoteLikeEnvironment(FakeEnvironment):
        def runtime_handle(self):
            return {"workspace_root": "/workspace/repo", "exec_url": "http://remote/exec"}

        def surfaces(self):
            return {
                "terminal": ClientSurface(surface_type="terminal"),
                "fs": ClientSurface(surface_type="fs"),
            }

    session = AgentRuntimeSession(
        sample={"instruction": "fix the failing test", "messages": []},
        trace=ObservabilityTrace(),
        plan=_build_plan(),
        resources=ResourceBundle(environment=_RemoteLikeEnvironment()),
        artifacts=ArtifactLayout.for_sample("runs", "run-1", "sample-1"),
    )

    class _FakeClient:
        def setup(self, environment, runtime_session) -> None:
            return None

        def run(self, request, environment):
            from gage_eval.agent_runtime.clients import ClientRunResult

            return ClientRunResult(exit_code=0, stdout="ok", stderr="")

        def cleanup(self, environment, runtime_session) -> None:
            return None

    monkeypatch.setattr(scheduler, "_build_client", lambda: _FakeClient())
    monkeypatch.setattr(scheduler, "_resolve_kit_hook", lambda *args, **kwargs: None)

    result = scheduler.run(session)

    assert result.raw_output["runtime_handle"]["workspace_root"] == "/workspace/repo"
    assert result.raw_output["surface_names"] == ("terminal", "fs")
    assert session.resources.metadata["workspace_root"] == "/workspace/repo"


@pytest.mark.fast
def test_installed_client_scheduler_builds_codex_client_with_default_args() -> None:
    scheduler = InstalledClientScheduler(_build_plan())
    client = scheduler._build_client()

    assert client._default_args == ("--dangerously-bypass-approvals-and-sandbox",)


@pytest.mark.fast
def test_installed_client_scheduler_runs_environment_hooks(monkeypatch, tmp_path) -> None:
    scheduler = InstalledClientScheduler(_build_plan())
    session = AgentRuntimeSession(
        sample={"instruction": "fix the failing test", "messages": []},
        trace=ObservabilityTrace(),
        plan=_build_plan(),
        resources=ResourceBundle(environment=FakeEnvironment()),
        artifacts=ArtifactLayout.for_sample(str(tmp_path), "run-1", "sample-1"),
    )

    class _FakeClient:
        def setup(self, environment, runtime_session) -> None:
            return None

        def run(self, request, environment):
            from gage_eval.agent_runtime.clients import ClientRunResult

            return ClientRunResult(exit_code=0, stdout="ok", stderr="")

        def cleanup(self, environment, runtime_session) -> None:
            return None

    def _resolve_kit_hook(_module_name, attr_name):
        if attr_name == "prepare_environment":
            return lambda sample, runtime_session, environment, request: {"prepare_environment_exit_code": 0}
        if attr_name == "capture_environment_artifacts":
            return (
                lambda sample, runtime_session, environment, client_result, request: {
                    "agent_workspace_dir": str(tmp_path / "captured-workspace")
                }
            )
        return None

    monkeypatch.setattr(scheduler, "_build_client", lambda: _FakeClient())
    monkeypatch.setattr(scheduler, "_resolve_kit_hook", _resolve_kit_hook)

    result = scheduler.run(session)

    assert result.raw_output["prepare_environment"]["prepare_environment_exit_code"] == 0
    assert result.raw_output["capture_environment_artifacts"]["agent_workspace_dir"].endswith(
        "captured-workspace"
    )
    assert result.artifacts["agent_workspace_dir"].endswith("captured-workspace")


@pytest.mark.fast
def test_installed_client_scheduler_persists_recursive_metadata(monkeypatch, tmp_path) -> None:
    scheduler = InstalledClientScheduler(_build_plan())
    session = AgentRuntimeSession(
        sample={"instruction": "fix the failing test", "messages": []},
        trace=ObservabilityTrace(),
        plan=_build_plan(),
        resources=ResourceBundle(environment=FakeEnvironment()),
        artifacts=ArtifactLayout.for_sample(str(tmp_path), "run-1", "sample-1"),
    )

    class _FakeClient:
        def setup(self, environment, runtime_session) -> None:
            return None

        def run(self, request, environment):
            from gage_eval.agent_runtime.clients import ClientRunResult

            return ClientRunResult(exit_code=0, stdout="ok", stderr="")

        def cleanup(self, environment, runtime_session) -> None:
            return None

    def _resolve_kit_hook(_module_name, attr_name):
        if attr_name != "finalize_result":
            return None

        def _finalize(_sample, result, _artifacts):
            recursive = {"status": result.status}
            recursive["self"] = recursive
            return {"recursive_payload": recursive}

        return _finalize

    monkeypatch.setattr(scheduler, "_build_client", lambda: _FakeClient())
    monkeypatch.setattr(scheduler, "_resolve_kit_hook", _resolve_kit_hook)

    result = scheduler.run(session)

    assert result.status == "success"
    metadata = json.loads(
        tmp_path.joinpath("run-1/samples/global/sample-1/runtime_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        metadata["scheduler_result"]["raw_output"]["recursive_payload"]["self"]
        == "<recursive_ref>"
    )


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
