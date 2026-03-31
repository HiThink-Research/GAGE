"""Installed client scheduler implementation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib
from typing import Any, Callable, Dict

from gage_eval.agent_runtime.clients import ClientRunRequest
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.schedulers import SchedulerResult


class InstalledClientScheduler:
    """Runs an installed CLI client inside the selected environment."""

    def __init__(self, plan: CompiledRuntimePlan) -> None:
        self._plan = plan

    def run(self, session) -> SchedulerResult:
        environment = session.resources.environment
        started = False
        if hasattr(environment, "start"):
            environment.start()
            started = True
        client = self._build_client()
        try:
            client.setup(environment, session)
            request = self._build_request(session)
            client_result = client.run(request, environment)
            raw_output = {
                "exit_code": client_result.exit_code,
                "stdout": client_result.stdout,
                "stderr": client_result.stderr,
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
                "sample_dir",
                "agent_dir",
                "verifier_dir",
                "patch_file",
                "trajectory_file",
                "stdout_file",
                "metadata_file",
            )
            if getattr(artifacts, key, None) is not None
        }
    metadata = {str(key): str(value) for key, value in payload.items() if value is not None}
    aliases = (
        ("patch_file", "patch_path"),
        ("trajectory_file", "trajectory_path"),
        ("stdout_file", "stdout_path"),
        ("metadata_file", "metadata_path"),
    )
    for source_key, alias_key in aliases:
        value = metadata.get(source_key)
        if value is not None:
            metadata.setdefault(alias_key, value)
    return metadata
