"""Installed client scheduler implementation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict

from gage_eval.agent_runtime.clients import ClientRunRequest
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.schedulers import SchedulerResult
from gage_eval.sandbox.surfaces import serialize_surfaces


class InstalledClientScheduler:
    """Runs an installed CLI client inside the selected environment."""

    def __init__(self, plan: CompiledRuntimePlan) -> None:
        self._plan = plan

    def run(self, session) -> SchedulerResult:
        environment = session.resources.environment
        started = False
        runtime_handle: dict[str, Any] = {}
        surfaces = {}
        if hasattr(environment, "start"):
            environment.start()
            started = True
        runtime_handle, surfaces = _capture_environment_resources(session, environment)
        client = self._build_client()
        try:
            client.setup(environment, session)
            request = self._build_request(session)
            client_result = client.run(request, environment)
            raw_output = {
                "exit_code": client_result.exit_code,
                "stdout": client_result.stdout,
                "stderr": client_result.stderr,
                "runtime_handle": dict(runtime_handle),
                "surface_names": tuple(surfaces.keys()),
                "surfaces": serialize_surfaces(surfaces),
                "workspace_root": session.resources.metadata.get("workspace_root"),
            }
            if client_result.patch_content:
                raw_output["patch_content"] = client_result.patch_content
            artifacts = dict(client_result.artifacts)
            if client_result.patch_path:
                artifacts.setdefault("patch_path", client_result.patch_path)
            if client_result.trajectory_path:
                artifacts.setdefault("trajectory_path", client_result.trajectory_path)
            result = SchedulerResult(
                status="success" if client_result.exit_code == 0 else "error",
                answer=client_result.patch_content,
                patch_path=client_result.patch_path,
                stdout_path=client_result.artifacts.get("stdout_path")
                or client_result.artifacts.get("stdout"),
                trajectory_path=client_result.trajectory_path,
                artifacts=artifacts,
                raw_output=raw_output,
            )
            finalizer = self._resolve_kit_hook("sub_workflow", "finalize_result")
            if finalizer is not None:
                finalized = finalizer(session.sample, result, session.artifacts) or {}
                result.raw_output.update(dict(finalized))
                if "answer" in finalized and finalized["answer"] is not None:
                    result.answer = str(finalized["answer"])
            _persist_scheduler_artifacts(
                session=session,
                request=request,
                client_result=client_result,
                runtime_handle=runtime_handle,
                surfaces=surfaces,
                result=result,
            )
            return result
        finally:
            try:
                client.cleanup(environment, session)
            finally:
                if started and hasattr(environment, "stop"):
                    environment.stop()

    def _build_client(self):
        client_id = self._plan.client_id or "codex"
        runtime_params = dict(getattr(self._plan.runtime_spec, "params", {}) or {})
        client_default_args = runtime_params.get("client_default_args")
        if not isinstance(client_default_args, (list, tuple)):
            client_default_args = None
        if client_id == "codex":
            from gage_eval.agent_runtime.clients.codex import CodexClient

            return CodexClient(default_args=client_default_args)
        if client_id == "claude":
            from gage_eval.agent_runtime.clients.claude import ClaudeClient

            return ClaudeClient()
        raise ValueError(f"Unsupported installed client '{client_id}'")

    def _build_request(self, session) -> ClientRunRequest:
        kit_prepare = self._resolve_kit_hook("sub_workflow", "prepare_inputs")
        prepared: Dict[str, Any] = {}
        if kit_prepare is not None:
            prepared = dict(kit_prepare(session.sample, session) or {})
        env = dict(self._plan.runtime_spec.resource_policy.env or {})
        env.update(prepared.get("env") or {})
        artifact_paths = _artifact_metadata(session.artifacts)
        metadata = dict(session.metadata or {})
        metadata.update(dict(session.resources.metadata or {}))
        metadata.update(artifact_paths)
        metadata.setdefault("artifacts", dict(artifact_paths))
        metadata.setdefault("timeout_sec", self._plan.runtime_spec.resource_policy.timeout_sec)
        metadata.update(prepared.get("metadata") or {})
        return ClientRunRequest(
            instruction=str(prepared.get("instruction") or session.sample.get("instruction") or ""),
            cwd=str(prepared.get("cwd") or session.sample.get("cwd") or "."),
            env=env,
            metadata=metadata,
        )

    def _resolve_kit_hook(self, module_name: str, attr_name: str) -> Callable[..., Any] | None:
        module = importlib.import_module(
            f"gage_eval.agent_eval_kits.{self._plan.benchmark_kit_id}.{module_name}"
        )
        return getattr(module, attr_name, None)


def _capture_environment_resources(session, environment) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_handle = {}
    surfaces = {}
    if hasattr(environment, "runtime_handle") and callable(environment.runtime_handle):
        runtime_handle = dict(environment.runtime_handle() or {})
    if hasattr(environment, "surfaces") and callable(environment.surfaces):
        surfaces = dict(environment.surfaces() or {})
    contract = getattr(environment, "contract", None)
    if contract is not None and session.resources.remote_sandbox is None:
        session.resources.remote_sandbox = contract
    if surfaces:
        session.resources.surfaces = dict(surfaces)
    workspace_root = (
        runtime_handle.get("workspace_root")
        or getattr(session.resources.remote_sandbox, "workspace_root", None)
        or getattr(session.resources.remote_sandbox, "attach_target", None)
    )
    session.resources.metadata.update(
        {
            "runtime_handle": runtime_handle,
            "surface_names": tuple(surfaces.keys()),
            "workspace_root": workspace_root,
        }
    )
    return runtime_handle, surfaces


def _artifact_metadata(artifacts: Any) -> Dict[str, str]:
    if artifacts is None:
        return {}
    if is_dataclass(artifacts):
        payload = asdict(artifacts)
    elif isinstance(artifacts, dict):
        payload = dict(artifacts)
    else:
        payload = {
            key: getattr(artifacts, key)
            for key in (
                "run_dir",
                "task_dir",
                "sample_dir",
                "canonical_sample_dir",
                "sample_file",
                "agent_dir",
                "verifier_dir",
                "patch_file",
                "trajectory_file",
                "stdout_file",
                "stderr_file",
                "final_message_file",
                "metadata_file",
                "verifier_result_file",
                "verifier_stdout_file",
                "verifier_stderr_file",
                "verifier_logs_dir",
                "verifier_workspace_dir",
                "attachments_dir",
            )
            if getattr(artifacts, key, None) is not None
        }
    metadata = {str(key): str(value) for key, value in payload.items() if value is not None}
    aliases = (
        ("patch_file", "patch_path"),
        ("trajectory_file", "trajectory_path"),
        ("stdout_file", "stdout_path"),
        ("stderr_file", "stderr_path"),
        ("metadata_file", "metadata_path"),
        ("final_message_file", "final_message_path"),
        ("verifier_result_file", "verifier_result_path"),
    )
    for source_key, alias_key in aliases:
        value = metadata.get(source_key)
        if value is not None:
            metadata.setdefault(alias_key, value)
    return metadata


def _persist_scheduler_artifacts(
    *,
    session,
    request: ClientRunRequest,
    client_result,
    runtime_handle: Dict[str, Any],
    surfaces: Dict[str, Any],
    result: SchedulerResult,
) -> None:
    artifacts = getattr(session, "artifacts", None)
    if artifacts is None:
        return
    stderr_file = getattr(artifacts, "stderr_file", None)
    if stderr_file:
        _write_text_artifact(stderr_file, str(getattr(client_result, "stderr", "") or ""))
    final_message_file = getattr(artifacts, "final_message_file", None)
    if final_message_file:
        final_message = result.answer or getattr(client_result, "stdout", "") or ""
        _write_text_artifact(final_message_file, str(final_message))
    metadata_file = getattr(artifacts, "metadata_file", None)
    if not metadata_file:
        return
    metadata_payload = {
        "run_id": getattr(getattr(session, "trace", None), "run_id", None),
        "task_id": getattr(session, "metadata", {}).get("task_id") if getattr(session, "metadata", None) else None,
        "sample_id": (
            session.sample.get("sample_id")
            or session.sample.get("id")
            or session.sample.get("instance_id")
        ),
        "benchmark_kit_id": getattr(getattr(session, "plan", None), "benchmark_kit_id", None),
        "client_id": getattr(getattr(session, "plan", None), "client_id", None),
        "environment_kind": getattr(getattr(session, "plan", None), "environment_kind", None),
        "workspace_root": getattr(session.resources, "metadata", {}).get("workspace_root"),
        "runtime_handle": dict(runtime_handle),
        "surface_names": tuple(surfaces.keys()),
        "surfaces": serialize_surfaces(surfaces),
        "request": {
            "instruction": request.instruction,
            "cwd": request.cwd,
            "env": dict(request.env or {}),
            "metadata": dict(request.metadata or {}),
        },
        "scheduler_result": {
            "status": result.status,
            "answer": result.answer,
            "patch_path": result.patch_path,
            "stdout_path": result.stdout_path,
            "trajectory_path": result.trajectory_path,
            "artifacts": dict(result.artifacts or {}),
            "metrics": dict(result.metrics or {}),
            "raw_output": dict(result.raw_output or {}),
        },
        "artifact_layout": _artifact_metadata(artifacts),
    }
    _write_json_artifact(metadata_file, metadata_payload)


def _write_text_artifact(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _write_json_artifact(path: str, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
