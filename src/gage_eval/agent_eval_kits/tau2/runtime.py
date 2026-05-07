from __future__ import annotations

import inspect
from typing import Any

from gage_eval.agent_eval_kits.common import extract_instruction
from gage_eval.agent_eval_kits.tau2.local_runtime import Tau2Runtime


class Tau2RuntimeEntry:
    """Owns Tau2 initialize_task bootstrap."""

    benchmark_kit_id = "tau2"
    supported_schedulers = ("installed_client", "framework_loop")

    def bootstrap(self, *, session, sample, payload, sandbox_provider=None):
        """Bootstrap Tau2 runtime state through initialize_task()."""

        del sandbox_provider
        environment_lease = payload.get("environment_lease") or session.runtime_context.get("environment_lease")
        if environment_lease is None:
            raise RuntimeError("tau2 environment lease is unavailable")
        runtime = getattr(environment_lease, "environment", None)
        if not _is_tau2_runtime(runtime):
            runtime_configs = _resolve_tau2_runtime_configs(session=session, payload=payload)
            runtime = Tau2Runtime(runtime_configs=runtime_configs)
            runtime.start({"runtime_configs": runtime_configs})
            runtime_lease = Tau2RuntimeEnvironmentLease(base_lease=environment_lease, runtime=runtime)
        else:
            runtime_lease = environment_lease
        initialize_result = runtime.initialize_task(sample)
        prompt_context = build_tau2_prompt_context(sample, initialize_result)
        return {
            "runtime_context": {**prompt_context, "environment_lease": runtime_lease},
            "prompt_context": prompt_context,
            "benchmark_state": {"initialize_result": initialize_result},
            "scheduler_state": {},
        }


def build_tau2_prompt_context(sample: dict[str, object], initialize_result: dict[str, object]) -> dict[str, object]:
    """Build the Tau2 runtime-owned prompt context."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    tau2 = metadata.get("tau2") if isinstance(metadata.get("tau2"), dict) else {}
    return {
        "instruction": extract_instruction(sample),
        "domain": tau2.get("domain"),
        "policy": tau2.get("policy"),
        "agent_instruction": tau2.get("agent_instruction"),
        "gage_instruction": tau2.get("gage_instruction"),
        "tools_schema": list(initialize_result.get("tools_schema") or []),
    }


class Tau2RuntimeEnvironmentLease:
    """Kit-owned Tau2 runtime facade over a generic local-process environment lease."""

    def __init__(self, *, base_lease: Any, runtime: Tau2Runtime) -> None:
        self._base_lease = base_lease
        self._runtime = runtime

    @property
    def environment(self) -> Tau2Runtime:
        return self._runtime

    @property
    def lease_id(self) -> str:
        return str(getattr(self._base_lease, "lease_id", ""))

    @property
    def provider(self) -> str:
        return str(getattr(self._base_lease, "provider", "local_process"))

    @property
    def profile_id(self) -> str:
        return str(getattr(self._base_lease, "profile_id", "tau2-local-process"))

    @property
    def lifecycle(self) -> str:
        return str(getattr(self._base_lease, "lifecycle", "per_sample"))

    @property
    def exclusive(self) -> bool:
        return bool(getattr(self._base_lease, "exclusive", True))

    @property
    def created_at(self) -> Any:
        return getattr(self._base_lease, "created_at", None)

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = getattr(self._base_lease, "metadata", None)
        return dict(metadata or {}) if isinstance(metadata, dict) else {}

    @property
    def artifact_sink(self) -> Any:
        return getattr(self._base_lease, "artifact_sink", None)

    @artifact_sink.setter
    def artifact_sink(self, value: Any) -> None:
        setattr(self._base_lease, "artifact_sink", value)

    def to_descriptor(self) -> dict[str, Any]:
        descriptor = _call_descriptor(self._base_lease)
        descriptor.setdefault("provider", self.provider)
        descriptor.setdefault("profile_id", self.profile_id)
        descriptor.setdefault("lifecycle", self.lifecycle)
        metadata = descriptor.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("runtime_adapter", "tau2")
        return descriptor

    def describe(self) -> dict[str, Any]:
        return self.to_descriptor()

    async def exec_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._runtime.exec_tool(name, arguments)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.exec_tool(name, arguments)

    def get_state(self) -> dict[str, Any]:
        return self._runtime.get_state()

    def initialize_task(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self._runtime.initialize_task(sample)

    async def exec(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "exec")(*args, **kwargs))

    async def upload_file(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "upload_file")(*args, **kwargs))

    async def upload_dir(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "upload_dir")(*args, **kwargs))

    async def download_file(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "download_file")(*args, **kwargs))

    async def download_dir(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "download_dir")(*args, **kwargs))

    async def write_file(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "write_file")(*args, **kwargs))

    async def read_file(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "read_file")(*args, **kwargs))

    async def list_files(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "list_files")(*args, **kwargs))

    async def is_file(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "is_file")(*args, **kwargs))

    async def is_dir(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "is_dir")(*args, **kwargs))

    async def get_logs(self, *args: Any, **kwargs: Any) -> Any:
        return await _maybe_await(getattr(self._base_lease, "get_logs")(*args, **kwargs))


def _resolve_tau2_runtime_configs(*, session: Any, payload: dict[str, Any]) -> dict[str, Any]:
    provider_config = payload.get("provider_config")
    if not isinstance(provider_config, dict):
        resource_lease = getattr(session, "resource_lease", None)
        metadata = getattr(resource_lease, "metadata", None)
        provider_config = metadata.get("provider_config") if isinstance(metadata, dict) else {}
    return dict(provider_config or {}) if isinstance(provider_config, dict) else {}


def _is_tau2_runtime(runtime: Any) -> bool:
    return all(callable(getattr(runtime, name, None)) for name in ("initialize_task", "exec_tool", "get_state"))


def _call_descriptor(base_lease: Any) -> dict[str, Any]:
    descriptor = getattr(base_lease, "to_descriptor", None)
    if callable(descriptor):
        payload = descriptor()
        if isinstance(payload, dict):
            return dict(payload)
    describe = getattr(base_lease, "describe", None)
    if callable(describe):
        payload = describe()
        if isinstance(payload, dict):
            return dict(payload)
    return {}


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
