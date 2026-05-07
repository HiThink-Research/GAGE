from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.clients.contracts import (
    AcpClientSchedulerConfig,
    InstalledClientSchedulerConfig,
)
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_runtime.contracts.failure import FailureEnvelopeError
from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.resources.manager import RuntimeLeaseBinding
from gage_eval.agent_runtime.schedulers.acp_client import AcpClientScheduler
from gage_eval.agent_runtime.schedulers.installed_client import InstalledClientScheduler
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.config.agentkit_v2 import materialize_agentkit_v2_config_payload
from gage_eval.agent_runtime.executor import _append_environment_handle_projected


class _CapturingClient:
    def __init__(self) -> None:
        self.environments: list[dict] = []

    def run(self, request: dict, environment: dict) -> dict:
        self.environments.append(dict(environment))
        return {"answer": "done", "status": "completed"}


class _SessionInspectingClient:
    def __init__(self) -> None:
        self.setup_session = None

    def setup(self, environment: dict, session) -> dict:
        del environment
        self.setup_session = session
        return {}

    def run(self, request: dict, environment: dict) -> dict:
        del request, environment
        return {"answer": "done", "status": "completed"}


class _LegacyInspectingClient:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def invoke(self, payload: dict) -> dict:
        self.payloads.append(dict(payload))
        return {"answer": "done", "status": "completed"}


def test_installed_client_config_accepts_command_array_and_timeout() -> None:
    config = InstalledClientSchedulerConfig.model_validate(
        {"client": {"command": ["codex", "run"], "timeout_s": 300}}
    )

    assert config.client.command == ["codex", "run"]
    assert config.client.timeout_s == 300

    with pytest.raises(ValidationError):
        InstalledClientSchedulerConfig.model_validate(
            {"client": {"command": "codex run", "timeout_s": 300}}
        )

    materialized = materialize_agentkit_v2_config_payload(
        _minimal_agentkit_payload(
            scheduler_type="installed_client",
            scheduler_config={"client": {"command": ["codex", "run"], "timeout_s": 300}},
        ),
        None,
    )
    assert materialized["agents"][0]["scheduler"]["config"]["client"]["command"] == [
        "codex",
        "run",
    ]

    with pytest.raises(Exception, match="command"):
        materialize_agentkit_v2_config_payload(
            _minimal_agentkit_payload(
                scheduler_type="installed_client",
                scheduler_config={"client": {"command": "codex run", "timeout_s": 300}},
            ),
            None,
        )


def test_acp_client_config_accepts_endpoint_and_capabilities() -> None:
    config = AcpClientSchedulerConfig.model_validate(
        {
            "client": {
                "endpoint": "http://127.0.0.1:7345",
                "capabilities": {"tools": True, "streaming": False},
            }
        }
    )

    assert config.client.endpoint == "http://127.0.0.1:7345"
    assert config.client.capabilities.tools is True
    assert config.client.capabilities.streaming is False

    materialized = materialize_agentkit_v2_config_payload(
        _minimal_agentkit_payload(scheduler_type="acp_client"),
        None,
    )
    assert materialized["agents"][0]["scheduler"]["type"] == "acp_client"


def test_acp_client_agentkit_binding_builds_stub_executor() -> None:
    from gage_eval.agent_runtime.schedulers.acp_client import AcpClientScheduler
    from gage_eval.config.agentkit_v2 import (
        build_agentkit_v2_runtime_bindings,
        materialize_agentkit_v2_runtime_config_payload,
    )

    payload = _minimal_agentkit_payload(scheduler_type="acp_client")
    materialized = materialize_agentkit_v2_config_payload(payload, None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, None)

    bindings = build_agentkit_v2_runtime_bindings(
        materialized,
        runtime_config=runtime_config,
        backends={"model": object()},
    )

    executor = bindings["dut"].executor_ref
    assert isinstance(executor.compiled_plan.scheduler_handle, AcpClientScheduler)


def test_acp_client_executor_surfaces_scheduler_failure_as_trial_failure(tmp_path: Path) -> None:
    from gage_eval.config.agentkit_v2 import (
        build_agentkit_v2_runtime_bindings,
        materialize_agentkit_v2_runtime_config_payload,
    )

    payload = _minimal_agentkit_payload(scheduler_type="acp_client")
    materialized = materialize_agentkit_v2_config_payload(payload, None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, None)
    bindings = build_agentkit_v2_runtime_bindings(
        materialized,
        runtime_config=runtime_config,
        backends={"model": object()},
    )
    executor = bindings["dut"].executor_ref
    executor.artifact_sink._base_dir = tmp_path
    lease = ResourceLease(
        lease_id="lease-1",
        resource_kind="e2b",
        profile_id="terminal_bench_runtime",
        lifecycle="per_sample",
        handle_ref={},
    )
    executor.resource_manager.acquire = lambda *_, **__: RuntimeLeaseBinding(
        resource_lease=lease,
        environment_lease=None,
    )
    executor.resource_manager.release = lambda *_args, **_kwargs: None

    output = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "instruction": "say done", "expected_answer": "done"},
            payload={
                "execution_context": {
                    "run_id": "run-acp",
                    "task_id": "task-acp",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    assert output["runtime_failure"]["failure_code"] == "client_execution.scheduler.acp_unsupported"
    metadata_path = Path(output["runtime_session"]["runtime_metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["failure"]["failure_code"] == "client_execution.scheduler.acp_unsupported"
    trace_path = tmp_path / "run-acp" / "artifacts/task-acp/sample-1/trials/trial_0001/infra/trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event_type"] == "trial.end"
    assert events[-1]["payload"]["status"] == "failed"
    assert events[-1]["payload"]["failure"]["failure_code"] == "client_execution.scheduler.acp_unsupported"


def test_installed_client_uses_external_client_environment_handle_only(tmp_path: Path) -> None:
    client = _CapturingClient()
    scheduler = InstalledClientScheduler(client)
    session = _session(
        tmp_path,
        resource_kind="docker",
        handle_ref={
            "container_id": "raw-container-id",
            "container_name": "raw-container-name",
        },
        provider_config={"workdir": "/workspace", "exec_workdir": "/workspace"},
    )

    result = asyncio.run(
        scheduler.arun(
            session=session,
            sample={"id": "sample-1", "instruction": "say done"},
            payload={},
            workflow_bundle=_bundle(),
            sandbox_provider=object(),
        )
    )

    assert result.status == "completed"
    environment = client.environments[0]
    handle = environment["environment_handle"]
    assert environment["external_environment_handle"] == handle
    assert handle["provider"] == "docker"
    assert handle["transport"] == "mounted_workdir"
    assert handle["workdir"] == "/workspace"
    assert "sandbox_provider" not in environment
    assert "runtime_handle" not in environment
    assert "container_id" not in json.dumps(environment)
    assert "raw-container-id" not in json.dumps(environment)


def test_installed_client_setup_receives_sanitized_session_context(tmp_path: Path) -> None:
    client = _SessionInspectingClient()
    scheduler = InstalledClientScheduler(client)
    session = _session(
        tmp_path,
        resource_kind="docker",
        handle_ref={"container_id": "raw-container-id"},
        provider_config={"workdir": "/workspace"},
    )

    asyncio.run(
        scheduler.arun(
            session=session,
            sample={"id": "sample-1"},
            payload={},
            workflow_bundle=_bundle(),
            sandbox_provider=None,
        )
    )

    setup_session = client.setup_session
    assert setup_session is not None
    assert setup_session.session_id == "session-1"
    assert setup_session.environment_handle.provider == "docker"
    assert not hasattr(setup_session, "resource_lease")
    assert "raw-container-id" not in json.dumps(setup_session.model_dump(mode="json"))


def test_legacy_installed_client_payload_receives_sanitized_session_context(tmp_path: Path) -> None:
    client = _LegacyInspectingClient()
    scheduler = InstalledClientScheduler(client)

    asyncio.run(
        scheduler.arun(
            session=_session(
                tmp_path,
                resource_kind="docker",
                handle_ref={"container_id": "raw-container-id"},
                provider_config={"workdir": "/workspace"},
            ),
            sample={"id": "sample-1"},
            payload={},
            workflow_bundle=_bundle(),
            sandbox_provider=None,
        )
    )

    session = client.payloads[0]["session"]
    assert session.client_id == "codex"
    assert not hasattr(session, "resource_lease")
    assert "raw-container-id" not in json.dumps(session.model_dump(mode="json"))


def test_installed_client_cached_environment_handle_is_revalidated(tmp_path: Path) -> None:
    scheduler = InstalledClientScheduler(_CapturingClient())
    session = _session(tmp_path, resource_kind="docker", provider_config={"workdir": "/workspace"})
    session.runtime_context["external_environment_handle"] = {
        "provider": "docker",
        "transport": "mounted_workdir",
        "container_id": "raw-container-id",
    }

    with pytest.raises(FailureEnvelopeError) as exc_info:
        asyncio.run(
            scheduler.arun(
                session=session,
                sample={"id": "sample-1"},
                payload={},
                workflow_bundle=_bundle(),
                sandbox_provider=None,
            )
        )

    assert exc_info.value.failure.failure_code == (
        "client_execution.client_environment_projection_denied"
    )


def test_e2b_without_projection_proxy_returns_unsupported(tmp_path: Path) -> None:
    scheduler = InstalledClientScheduler(_CapturingClient())

    with pytest.raises(FailureEnvelopeError) as exc_info:
        asyncio.run(
            scheduler.arun(
                session=_session(tmp_path, resource_kind="e2b", handle_ref={}),
                sample={"id": "sample-1"},
                payload={},
                workflow_bundle=_bundle(),
                sandbox_provider=None,
            )
        )

    assert exc_info.value.failure.failure_code == (
        "client_execution.client_environment_projection_unsupported"
    )


def test_external_client_raw_provider_id_access_returns_denied(tmp_path: Path) -> None:
    scheduler = InstalledClientScheduler(_CapturingClient())

    with pytest.raises(FailureEnvelopeError) as exc_info:
        asyncio.run(
            scheduler.arun(
                session=_session(
                    tmp_path,
                    resource_kind="docker",
                    handle_ref={"container_id": "raw-container-id"},
                    provider_config={"workdir": "/workspace"},
                ),
                sample={"id": "sample-1"},
                payload={"external_client_requested_fields": ["container_id"]},
                workflow_bundle=_bundle(),
                sandbox_provider=None,
            )
        )

    assert exc_info.value.failure.failure_code == (
        "client_execution.client_environment_projection_denied"
    )


def test_environment_handle_proxy_startup_failure_returns_projection_failed(tmp_path: Path) -> None:
    scheduler = InstalledClientScheduler(_CapturingClient())

    def _fail_proxy_start(_handle: dict) -> None:
        raise RuntimeError("proxy failed")

    with pytest.raises(FailureEnvelopeError) as exc_info:
        asyncio.run(
            scheduler.arun(
                session=_session(
                    tmp_path,
                    resource_kind="e2b",
                    handle_ref={"jsonrpc_proxy_endpoint": "http://127.0.0.1:7345"},
                ),
                sample={"id": "sample-1"},
                payload={"environment_proxy_starter": _fail_proxy_start},
                workflow_bundle=_bundle(),
                sandbox_provider=None,
            )
        )

    assert exc_info.value.failure.failure_code == (
        "client_execution.client_environment_projection_failed"
    )


def test_environment_handle_projected_trace_is_desensitized(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    session = _session(
        tmp_path,
        resource_kind="docker",
        handle_ref={"container_id": "raw-container-id"},
        provider_config={"workdir": "/workspace"},
        startup_env={"OPENAI_API_KEY": "secret-token"},
    )
    binding = RuntimeLeaseBinding(
        resource_lease=session.resource_lease,
        environment_lease=None,
    )

    _append_environment_handle_projected(
        artifact_sink=sink,
        session=session,
        trial_id="trial_0001",
        environment_lease=None,
        lease_binding=binding,
    )

    trace_path = tmp_path / "run-1" / "artifacts/task-1/sample-1/trials/trial_0001/infra/trace.jsonl"
    event = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["event_type"] == "client.environment_handle.projected"
    assert event["payload"]["environment_handle"]["transport"] == "mounted_workdir"
    assert event["payload"]["environment_handle"]["env_vars"]["OPENAI_API_KEY"] == "<redacted>"
    assert "container_id" not in json.dumps(event)
    assert "raw-container-id" not in json.dumps(event)
    assert "secret-token" not in json.dumps(event)


def test_acp_projection_does_not_preempt_stub_failure_code(tmp_path: Path) -> None:
    session = _session(tmp_path, scheduler_type="acp_client", resource_kind="e2b")
    binding = RuntimeLeaseBinding(
        resource_lease=session.resource_lease,
        environment_lease=None,
    )

    _append_environment_handle_projected(
        artifact_sink=RuntimeArtifactSink(base_dir=str(tmp_path)),
        session=session,
        trial_id="trial_0001",
        environment_lease=None,
        lease_binding=binding,
    )


def test_acp_client_returns_single_unsupported_code(tmp_path: Path) -> None:
    result = asyncio.run(
        AcpClientScheduler().arun(
            session=_session(tmp_path, scheduler_type="acp_client"),
            sample={"id": "sample-1"},
            payload={},
            workflow_bundle=_bundle(scheduler_type="acp_client"),
            sandbox_provider=None,
        )
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.failure_code == "client_execution.scheduler.acp_unsupported"


def _session(
    tmp_path: Path,
    *,
    scheduler_type: str = "installed_client",
    resource_kind: str = "docker",
    handle_ref: dict | None = None,
    provider_config: dict | None = None,
    startup_env: dict | None = None,
) -> AgentRuntimeSession:
    return AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="terminal_bench",
        scheduler_type=scheduler_type,
        client_id="codex",
        artifact_layout={
            "sample_root": str(tmp_path / "sample"),
            "artifacts_dir": str(tmp_path / "sample" / "artifacts"),
            "verifier_result": str(tmp_path / "sample" / "verifier" / "result.json"),
            "runtime_metadata": str(tmp_path / "sample" / "runtime_metadata.json"),
            "raw_error": str(tmp_path / "sample" / "logs" / "raw_error.json"),
        },
        resource_lease=ResourceLease(
            lease_id="lease-1",
            resource_kind=resource_kind,  # type: ignore[arg-type]
            profile_id="terminal_bench_runtime",
            lifecycle="per_sample",
            handle_ref=dict(handle_ref or {}),
            metadata={
                "environment_profile": {
                    "profile_id": "terminal_bench_runtime",
                    "provider": resource_kind,
                    "config": dict(provider_config or {}),
                    "startup_env": dict(startup_env or {}),
                },
                "provider_config": dict(provider_config or {}),
            },
        ),
    )


def _bundle(*, scheduler_type: str = "installed_client") -> SchedulerWorkflowBundle:
    return SchedulerWorkflowBundle(
        bundle_id=f"terminal_bench.{scheduler_type}",
        benchmark_kit_id="terminal_bench",
        scheduler_type=scheduler_type,
        prepare_inputs=lambda **_: {"instruction": "say done"},
        failure_normalizer=lambda **_: {},
    )


def _minimal_agentkit_payload(
    *,
    scheduler_type: str,
    scheduler_config: dict | None = None,
) -> dict:
    if scheduler_config is None:
        scheduler_config = {
            "client": {
                "endpoint": "http://127.0.0.1:7345",
                "capabilities": {"tools": True, "streaming": False},
            }
        } if scheduler_type == "acp_client" else {}
    return {
        "kind": "AgentEvalConfig",
        "metadata": {"name": "acp-smoke"},
        "backends": [{"backend_id": "model", "type": "litellm", "config": {"model": "demo"}}],
        "agents": [
            {
                "agent_id": "agent",
                "scheduler": {
                    "type": scheduler_type,
                    "config": scheduler_config,
                },
            }
        ],
        "benchmarks": [{"benchmark_id": "bench", "kit_id": "terminal_bench", "config": {}}],
        "environments": [
            {
                "env_id": "env",
                "provider": "docker",
                "profile_id": "terminal_bench_runtime",
                "profile": {"asset_dir": "assets/terminal_bench"},
            }
        ],
        "dut_agents": [
            {
                "dut_id": "dut",
                "agent_id": "agent",
                "env_id": "env",
                "benchmark_id": "bench",
            }
        ],
    }
