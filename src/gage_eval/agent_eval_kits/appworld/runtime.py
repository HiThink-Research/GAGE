from __future__ import annotations

from typing import Any, Callable

import requests

from gage_eval.agent_eval_kits.appworld.units import build_appworld_prompt_context


_APPWORLD_INIT_KEYS = {
    "experiment_name",
    "remote_apis_url",
    "remote_environment_url",
    "remote_mcp_url",
    "remote_docker",
    "max_interactions",
    "max_api_calls_per_interaction",
    "raise_on_unsafe_syntax",
    "raise_on_unsafe_execution",
    "load_ground_truth",
    "ground_truth_mode",
    "raise_on_failure",
    "random_seed",
    "timeout_seconds",
    "show_api_response_schemas",
    "gc_threshold",
    "include_direct_functions",
    "direct_function_separator",
    "raise_on_extra_parameters",
    "import_utils",
    "parse_datetimes",
    "allow_datetime_change",
    "add_login_shortcut",
    "wrap_response",
    "unwrap_response",
    "munchify_response",
}

_ALLOWED_GROUND_TRUTH_MODES = {"full", "partial", "minimal"}


class AppWorldRuntime:
    """Owns AppWorld initialize/save lifecycle."""

    benchmark_kit_id = "appworld"
    runtime_version = "phase1"
    supported_schedulers = ("installed_client", "framework_loop")
    verifier_kind = "judge_adapter"
    resource_requirements = {"resource_kind": "docker"}
    lifecycle_policy = {"initialize": "http_initialize", "save": "http_save", "teardown": "provider_managed"}
    state_schema_keys = ("runtime_context", "prompt_context", "benchmark_state", "scheduler_state")
    compat_mode = "legacy_support"

    def __init__(
        self,
        *,
        timeout_s: int = 30,
        requester: Callable[[str, str, dict[str, Any], int], dict[str, Any]] | None = None,
    ) -> None:
        """Initializes the AppWorld runtime entry.

        Args:
            timeout_s: HTTP timeout for lifecycle calls.
            requester: Optional request hook used by tests and local stubs.
        """

        self._timeout_s = max(1, int(timeout_s))
        self._requester = requester

    def bootstrap(self, *, session, sample, payload, sandbox_provider=None):
        """Bootstraps AppWorld by calling `/initialize`.

        Args:
            session: Sample-scoped runtime session.
            sample: Current sample payload.
            payload: Invocation payload.
            sandbox_provider: Sample-scoped sandbox provider.

        Returns:
            Runtime-owned context fragments for the executor session.

        Raises:
            ValueError: If the sandbox provider or AppWorld task id is missing.
            RuntimeError: If the AppWorld endpoint returns an invalid response.
        """

        # STEP 1: Resolve runtime handle endpoints from the sandbox lease.
        if sandbox_provider is None:
            raise ValueError("appworld runtime requires sandbox_provider")
        handle = sandbox_provider.get_handle()
        runtime_handle = handle.runtime_handle if handle is not None else {}

        # STEP 2: Initialize AppWorld state through the runtime-owned HTTP client.
        initialize_output = self._call_initialize(
            metadata=dict(sample.get("metadata") or {}),
            runtime_handle=dict(runtime_handle or {}),
        )

        # STEP 3: Project runtime-owned context for downstream workflows/verifier.
        prompt_context = build_appworld_prompt_context(sample, runtime_handle)
        return {
            "runtime_context": {**prompt_context, "initialize": initialize_output, "runtime_handle": runtime_handle},
            "prompt_context": prompt_context,
            "benchmark_state": {"initialize": initialize_output},
            "scheduler_state": {},
        }

    def save(self, *, sample, sandbox_provider=None):
        """Persists AppWorld sample state through `/save`.

        Args:
            sample: Current sample payload.
            sandbox_provider: Sample-scoped sandbox provider.

        Returns:
            The normalized AppWorld save payload when available.
        """

        if sandbox_provider is None:
            return {}
        handle = sandbox_provider.get_handle()
        runtime_handle = handle.runtime_handle if handle is not None else {}
        return self._call_save(
            metadata=dict(sample.get("metadata") or {}),
            runtime_handle=dict(runtime_handle or {}),
        )

    def _call_initialize(
        self,
        *,
        metadata: dict[str, Any],
        runtime_handle: dict[str, Any],
    ) -> dict[str, Any]:
        """Calls the AppWorld initialize endpoint."""

        appworld_meta = _extract_appworld_meta(metadata)
        task_id = appworld_meta.get("task_id")
        if not task_id:
            raise ValueError("appworld.task_id is required for initialize")

        payload = {"task_id": task_id}
        for key in _APPWORLD_INIT_KEYS:
            if key in appworld_meta:
                payload[key] = appworld_meta[key]
        _normalize_ground_truth_mode(payload)
        apis_endpoint = _resolve_endpoint(runtime_handle, "apis_endpoint", "apis_url")
        if apis_endpoint and "remote_apis_url" not in payload:
            payload["remote_apis_url"] = apis_endpoint
        mcp_endpoint = _resolve_endpoint(runtime_handle, "mcp_endpoint", "mcp_url")
        if mcp_endpoint and "remote_mcp_url" not in payload:
            payload["remote_mcp_url"] = mcp_endpoint
        env_endpoint = _resolve_required_endpoint(
            runtime_handle,
            "env_endpoint",
            "environment_endpoint",
            "env_url",
            "environment_url",
        )
        return self._post_appworld(env_endpoint, "initialize", payload)

    def _call_save(
        self,
        *,
        metadata: dict[str, Any],
        runtime_handle: dict[str, Any],
    ) -> dict[str, Any]:
        """Calls the AppWorld save endpoint."""

        appworld_meta = _extract_appworld_meta(metadata)
        task_id = appworld_meta.get("task_id")
        if not task_id:
            raise ValueError("appworld.task_id is required for save")
        env_endpoint = _resolve_required_endpoint(
            runtime_handle,
            "env_endpoint",
            "environment_endpoint",
            "env_url",
            "environment_url",
        )
        return self._post_appworld(env_endpoint, "save", {"task_id": task_id})

    def _post_appworld(
        self,
        endpoint: str,
        method: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Posts one AppWorld lifecycle request and normalizes the response."""

        if self._requester is not None:
            response_payload = self._requester(endpoint, method, payload, self._timeout_s)
            return _unwrap_output(response_payload)

        response = requests.post(
            f"{endpoint.rstrip('/')}/{method}",
            json=payload,
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return _unwrap_output(response.json())

def _extract_appworld_meta(metadata: dict[str, Any]) -> dict[str, Any]:
    appworld_meta = metadata.get("appworld")
    if not isinstance(appworld_meta, dict):
        return {}
    return dict(appworld_meta)


def _resolve_endpoint(runtime_handle: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = runtime_handle.get(name)
        if value:
            return str(value)
    return None


def _resolve_required_endpoint(runtime_handle: dict[str, Any], *names: str) -> str:
    endpoint = _resolve_endpoint(runtime_handle, *names)
    if endpoint:
        return endpoint
    joined = ", ".join(names)
    raise ValueError(f"AppWorld runtime endpoint is required ({joined})")


def _normalize_ground_truth_mode(payload: dict[str, Any]) -> None:
    candidate = payload.get("ground_truth_mode")
    if not isinstance(candidate, str):
        return
    normalized = candidate.strip().lower()
    if normalized == "auto":
        payload["ground_truth_mode"] = "minimal"
        return
    if normalized in _ALLOWED_GROUND_TRUTH_MODES:
        payload["ground_truth_mode"] = normalized
        return
    payload.pop("ground_truth_mode", None)


def _unwrap_output(response_payload: Any) -> dict[str, Any]:
    if isinstance(response_payload, dict):
        output = response_payload.get("output")
        if isinstance(output, dict):
            return dict(output)
        return dict(response_payload)
    raise RuntimeError("appworld runtime returned a non-dict payload")
