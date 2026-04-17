from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import gage_eval.agent_runtime.resolver as resolver_module
from gage_eval.agent_runtime import (
    RuntimeCompileError,
    build_compiled_runtime_executor,
    compile_agent_runtime_plan,
)
from gage_eval.agent_runtime.resources.manager import RuntimeLeaseBinding
from gage_eval.agent_runtime.clients import (
    LegacyInvokeClientSurface,
    StructuredClientSurfaceAdapter,
    build_client_surface,
)
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.dut_agent import DUTAgentAdapter
from gage_eval.sandbox.provider import SandboxProvider


class _StructuredClient:
    def __init__(self) -> None:
        self.setup_calls: list[tuple[dict, str]] = []
        self.run_calls: list[tuple[dict, dict]] = []

    def setup(self, environment: dict, session) -> dict:
        self.setup_calls.append((dict(environment), session.session_id))
        return {"prepared": True}

    def run(self, request: dict, environment: dict) -> dict:
        self.run_calls.append((dict(request), dict(environment)))
        return {"answer": "done", "artifact_paths": {"stdout": "stdout.txt"}}


class _LegacyInvokeClient:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def invoke(self, payload: dict) -> dict:
        self.payloads.append(dict(payload))
        return {"answer": "done"}


class _BuiltinInstalledClient(_StructuredClient):
    pass


def test_build_client_surface_prefers_structured_client_contract() -> None:
    surface = build_client_surface(_StructuredClient())

    assert isinstance(surface, StructuredClientSurfaceAdapter)


def test_build_client_surface_wraps_legacy_invoke_clients() -> None:
    surface = build_client_surface(_LegacyInvokeClient())

    assert isinstance(surface, LegacyInvokeClientSurface)


def test_installed_client_runner_uses_setup_then_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    builtin_client = _BuiltinInstalledClient()
    monkeypatch.setattr(
        "gage_eval.agent_runtime.clients.builder.instantiate_builtin_client",
        lambda client_id: builtin_client,
    )
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")
    plan = replace(plan, resource_plan={"resource_kind": "docker", "sandbox_config": {}})
    client = _StructuredClient()
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=client,
        installed_client_override=client,
        max_turns=4,
    )

    result = __import__("asyncio").run(
        executor.aexecute(
            sample={
                "id": "terminal-1",
                "instruction": "say done",
                "expected_answer": "done",
                "messages": [{"role": "user", "content": "say done"}],
            },
            payload={
                "sample": {
                    "id": "terminal-1",
                    "instruction": "say done",
                    "expected_answer": "done",
                    "messages": [{"role": "user", "content": "say done"}],
                },
                "execution_context": {
                    "run_id": "run-terminal",
                    "task_id": "task-terminal",
                    "sample_id": "terminal-1",
                },
            },
        )
    )

    assert not builtin_client.setup_calls
    assert not builtin_client.run_calls
    assert client.setup_calls
    assert client.run_calls
    request, environment = client.run_calls[0]
    assert request["instruction"].startswith("say done")
    assert "reply with exactly `done`" in request["instruction"]
    assert "session" not in request
    assert "sample" not in request
    assert "sandbox_provider" not in request
    assert request["metadata"]["sample_id"] == "terminal-1"
    assert environment["prepared"] is True
    runtime_metadata = json.loads(
        Path(result["runtime_session"]["runtime_metadata_path"]).read_text(encoding="utf-8")
    )
    assert runtime_metadata["client_id"] == "codex"
    assert runtime_metadata["scheduler_state"]["client_surface"] == "StructuredClientSurfaceAdapter"


def test_legacy_client_surface_receives_flattened_request_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")
    plan = replace(plan, resource_plan={"resource_kind": "docker", "sandbox_config": {}})
    client = _LegacyInvokeClient()
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=client,
        installed_client_override=client,
        max_turns=4,
    )

    __import__("asyncio").run(
        executor.aexecute(
            sample={
                "id": "terminal-1",
                "instruction": "say done",
                "expected_answer": "done",
                "messages": [{"role": "user", "content": "say done"}],
            },
            payload={
                "sample": {
                    "id": "terminal-1",
                    "instruction": "say done",
                    "expected_answer": "done",
                    "messages": [{"role": "user", "content": "say done"}],
                },
                "execution_context": {
                    "run_id": "run-terminal",
                    "task_id": "task-terminal",
                    "sample_id": "terminal-1",
                },
            },
        )
    )

    assert client.payloads
    payload = client.payloads[0]
    assert payload["instruction"].startswith("say done")
    assert "reply with exactly `done`" in payload["instruction"]
    assert payload["request"]["instruction"] == payload["instruction"]
    assert payload["environment"]["client_id"] == "codex"
    assert payload["session"].client_id == "codex"


def test_runtime_executor_passes_sample_sandbox_to_resource_manager(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "gage_eval.agent_runtime.clients.builder.instantiate_builtin_client",
        lambda client_id: _BuiltinInstalledClient(),
    )
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")
    plan = replace(plan, resource_plan={"resource_kind": "docker", "sandbox_config": {}})
    client = _StructuredClient()
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=client,
        max_turns=4,
    )
    captured: dict[str, object] = {}

    def fake_acquire(session, *, resource_plan, trace=None, sample=None):
        captured["sample"] = dict(sample or {})
        return RuntimeLeaseBinding(
            resource_lease=None,
            sandbox_provider=None,
            sandbox_handle=None,
        )

    executor.resource_manager.acquire = fake_acquire

    asyncio.run(
        executor.aexecute(
            sample={
                "id": "terminal-1",
                "instruction": "say done",
                "expected_answer": "done",
                "messages": [{"role": "user", "content": "say done"}],
                "sandbox": {
                    "sandbox_id": "terminal_bench_runtime",
                    "runtime": "docker",
                    "image": "fake-image:1",
                },
            },
            payload={
                "sample": {
                    "id": "terminal-1",
                    "instruction": "say done",
                    "expected_answer": "done",
                    "messages": [{"role": "user", "content": "say done"}],
                },
                "execution_context": {
                    "run_id": "run-terminal",
                    "task_id": "task-terminal",
                    "sample_id": "terminal-1",
                },
            },
        )
    )

    assert captured["sample"]["sandbox"]["image"] == "fake-image:1"


def test_runtime_executor_reuses_payload_sandbox_provider_without_resource_reacquire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "gage_eval.agent_runtime.clients.builder.instantiate_builtin_client",
        lambda client_id: _BuiltinInstalledClient(),
    )
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")
    plan = replace(plan, resource_plan={"resource_kind": "docker", "sandbox_config": {}})
    client = _StructuredClient()
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=client,
        max_turns=4,
    )

    manager = executor.resource_manager
    manager.acquire = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected acquire"))  # type: ignore[method-assign]
    manager.release = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected release"))  # type: ignore[method-assign]

    provider = SandboxProvider.__new__(SandboxProvider)
    provider._sandbox_config = {"sandbox_id": "terminal_bench_runtime", "runtime": "docker"}  # type: ignore[attr-defined]
    provider._handle = SimpleNamespace(  # type: ignore[attr-defined]
        runtime_handle={"container_id": "cid-terminal", "container_name": "terminal-runtime"}
    )
    provider.get_handle = lambda: provider._handle  # type: ignore[attr-defined]

    result = asyncio.run(
        executor.aexecute(
            sample={
                "id": "terminal-1",
                "instruction": "say done",
                "expected_answer": "done",
                "messages": [{"role": "user", "content": "say done"}],
            },
            payload={
                "sample": {
                    "id": "terminal-1",
                    "instruction": "say done",
                    "expected_answer": "done",
                    "messages": [{"role": "user", "content": "say done"}],
                },
                "sandbox_provider": provider,
                "execution_context": {
                    "run_id": "run-terminal",
                    "task_id": "task-terminal",
                    "sample_id": "terminal-1",
                },
            },
        )
    )

    runtime_metadata = json.loads(
        Path(result["runtime_session"]["runtime_metadata_path"]).read_text(encoding="utf-8")
    )
    assert runtime_metadata["resource_lease"]["handle_ref"]["container_name"] == "terminal-runtime"


def test_compile_runtime_plan_rejects_installed_client_without_client_id(monkeypatch) -> None:
    original = resolver_module.resolve_agent_runtime_spec

    def _resolve(agent_runtime_id: str):
        spec = original(agent_runtime_id)
        if agent_runtime_id != "terminal_bench_installed_client":
            return spec
        return replace(spec, client_id=None)

    monkeypatch.setattr(resolver_module, "resolve_agent_runtime_spec", _resolve)

    with pytest.raises(RuntimeCompileError) as exc_info:
        compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")

    assert exc_info.value.diagnostics[0]["code"] == "installed_client_missing_client_id"


def test_runtime_client_source_of_truth_does_not_reintroduce_artifacts_clients() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    assert not (repo_root / "src" / "gage_eval" / "agent_runtime" / "artifacts" / "clients").exists()


def test_build_executor_resolves_builtin_codex_client_without_override() -> None:
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")
    plan = replace(plan, resource_plan={"resource_kind": "docker", "sandbox_config": {}})
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=None,
        max_turns=4,
    )

    assert executor.compiled_plan.scheduler_handle is not None


def test_build_executor_rejects_unknown_installed_client_id() -> None:
    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")
    plan = replace(
        plan,
        runtime_spec=replace(plan.runtime_spec, client_id="unknown_client"),
        resource_plan={"resource_kind": "docker", "sandbox_config": {}},
    )

    with pytest.raises(ValueError) as exc_info:
        build_compiled_runtime_executor(
            compiled_plan=plan,
            agent_backend=None,
            max_turns=4,
        )

    assert "Unknown installed client" in str(exc_info.value)


def test_dut_agent_runtime_executor_mode_does_not_require_agent_backend() -> None:
    class _ExecutorStub:
        async def aexecute(self, *, sample, payload, trace=None):
            return {"answer": "done", "sample": sample}

    adapter = DUTAgentAdapter(
        adapter_id="dut-runtime",
        role_type="dut_agent",
        capabilities=(),
        executor_ref=_ExecutorStub(),
    )

    result = asyncio.run(
        adapter.ainvoke(
            {"sample": {"id": "sample-1"}},
            RoleAdapterState(),
        )
    )

    assert result["answer"] == "done"
