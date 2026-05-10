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
    supported_schedulers = ("installed_client", "framework_loop")

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
            sandbox_provider: Deprecated compatibility parameter; ignored.

        Returns:
            Runtime-owned context fragments for the executor session.

        Raises:
            ValueError: If the AppWorld task id or endpoint is missing.
            RuntimeError: If the AppWorld endpoint returns an invalid response.
        """

        del sandbox_provider
        runtime_handle = _resolve_runtime_handle(session=session, payload=payload)

        # STEP 2: Initialize AppWorld state through the runtime-owned HTTP client.
        initialize_output = self._call_initialize(
            metadata=dict(sample.get("metadata") or {}),
            runtime_handle=dict(runtime_handle or {}),
        )

        # STEP 3: Project runtime-owned context for downstream workflows/verifier.
        prompt_context = build_appworld_prompt_context(
            sample,
            runtime_handle,
            initialize_output,
        )
        return {
            "runtime_context": {**prompt_context, "initialize": initialize_output, "runtime_handle": runtime_handle},
            "prompt_context": prompt_context,
            "benchmark_state": {"initialize": initialize_output},
            "scheduler_state": {},
        }

    def save(self, *, sample, session=None, payload=None, sandbox_provider=None):
        """Persists AppWorld sample state through `/save`.

        Args:
            sample: Current sample payload.
            sandbox_provider: Deprecated compatibility parameter; ignored.

        Returns:
            The normalized AppWorld save payload when available.
        """

        del sandbox_provider
        runtime_handle = _resolve_runtime_handle(session=session, payload=payload or {})
        if not runtime_handle:
            return {}
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


def _resolve_runtime_handle(*, session: Any | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    handle: dict[str, Any] = {}
    payload_handle = payload.get("runtime_handle")
    if isinstance(payload_handle, dict):
        handle.update(payload_handle)

    resource_lease = getattr(session, "resource_lease", None)
    resource_metadata = getattr(resource_lease, "metadata", None)
    if isinstance(resource_metadata, dict):
        handle.update(_runtime_handle_from_resource_metadata(resource_metadata))
    resource_handle = getattr(resource_lease, "handle_ref", None)
    if isinstance(resource_handle, dict):
        handle.update(_flatten_runtime_descriptor(resource_handle))

    environment_lease = payload.get("environment_lease")
    if environment_lease is None and session is not None:
        runtime_context = getattr(session, "runtime_context", {}) or {}
        environment_lease = runtime_context.get("environment_lease")
    if environment_lease is not None and hasattr(environment_lease, "to_descriptor"):
        handle.update(_flatten_runtime_descriptor(environment_lease.to_descriptor()))
    return handle


def _runtime_handle_from_resource_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    handle: dict[str, Any] = {}
    for source in (
        metadata.get("provider_config"),
        (metadata.get("environment_profile") or {}).get("config")
        if isinstance(metadata.get("environment_profile"), dict)
        else None,
        (metadata.get("environment_profile") or {}).get("metadata")
        if isinstance(metadata.get("environment_profile"), dict)
        else None,
    ):
        if not isinstance(source, dict):
            continue
        for key in (
            "env_endpoint",
            "environment_endpoint",
            "apis_endpoint",
            "mcp_endpoint",
            "env_url",
            "apis_url",
            "mcp_url",
        ):
            if source.get(key):
                handle.setdefault(key, source[key])
    return handle


def _flatten_runtime_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(descriptor or {})
    environment_descriptor = flattened.get("environment_descriptor")
    if isinstance(environment_descriptor, dict):
        flattened.setdefault("container_name", environment_descriptor.get("name"))
        flattened.setdefault("container_id", environment_descriptor.get("env_id"))
        metadata = environment_descriptor.get("metadata")
        if isinstance(metadata, dict):
            runtime_handle = metadata.get("runtime_handle")
            if isinstance(runtime_handle, dict):
                flattened.update(runtime_handle)
            for key in (
                "env_endpoint",
                "environment_endpoint",
                "apis_endpoint",
                "mcp_endpoint",
                "env_url",
                "apis_url",
                "mcp_url",
            ):
                if key in metadata:
                    flattened.setdefault(key, metadata[key])
    return {key: value for key, value in flattened.items() if value is not None}


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
