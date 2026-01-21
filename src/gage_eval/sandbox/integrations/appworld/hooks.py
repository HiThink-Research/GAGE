"""AppWorld hook implementations for AgentLoop lifecycle."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import requests

from gage_eval.role.agent.hooks import AgentHookContext, register_hook_aliases


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


class AppWorldInitializeHook:
    """Initialize AppWorld environment before the agent loop starts."""

    def __init__(
        self,
        *,
        env_endpoint: Optional[str] = None,
        apis_endpoint: Optional[str] = None,
        timeout_s: int = 30,
        requester: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ) -> None:
        self._env_endpoint = env_endpoint
        self._apis_endpoint = apis_endpoint
        self._timeout_s = max(1, int(timeout_s))
        self._requester = requester

    def run(self, context: AgentHookContext) -> Dict[str, Any]:
        """Call AppWorld /initialize with task metadata."""

        appworld_meta = _extract_appworld_meta(context)
        task_id = appworld_meta.get("task_id")
        if not task_id:
            raise ValueError("appworld.task_id is required for initialize")
        payload = {"task_id": task_id}
        for key in _APPWORLD_INIT_KEYS:
            if key in appworld_meta:
                payload[key] = appworld_meta[key]
        _normalize_ground_truth_mode(payload)
        if "remote_apis_url" not in payload:
            apis_endpoint = self._resolve_apis_endpoint(context)
            if apis_endpoint:
                payload["remote_apis_url"] = apis_endpoint
        env_endpoint = self._resolve_env_endpoint(context)
        response = _post_appworld(env_endpoint, "initialize", payload, self._timeout_s, self._requester)
        output = _unwrap_output(response)
        context.hook_state.setdefault("appworld", {})["initialize"] = output
        return output

    def _resolve_env_endpoint(self, context: AgentHookContext) -> str:
        env_endpoint = self._env_endpoint or context.resolve_endpoint(
            "env_endpoint",
            "environment_endpoint",
            "env_url",
            "environment_url",
        )
        if not env_endpoint:
            raise ValueError("env_endpoint is required for AppWorld initialize")
        return env_endpoint

    def _resolve_apis_endpoint(self, context: AgentHookContext) -> Optional[str]:
        if self._apis_endpoint:
            return self._apis_endpoint
        return context.resolve_endpoint("apis_endpoint", "apis_url")


class AppWorldSaveHook:
    """Persist AppWorld environment after the agent loop completes."""

    def __init__(
        self,
        *,
        env_endpoint: Optional[str] = None,
        timeout_s: int = 30,
        requester: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ) -> None:
        self._env_endpoint = env_endpoint
        self._timeout_s = max(1, int(timeout_s))
        self._requester = requester

    def run(self, context: AgentHookContext) -> Dict[str, Any]:
        """Call AppWorld /save with task metadata."""

        appworld_meta = _extract_appworld_meta(context)
        task_id = appworld_meta.get("task_id")
        if not task_id:
            raise ValueError("appworld.task_id is required for save")
        env_endpoint = self._resolve_env_endpoint(context)
        response = _post_appworld(
            env_endpoint,
            "save",
            {"task_id": task_id},
            self._timeout_s,
            self._requester,
        )
        output = _unwrap_output(response)
        context.hook_state.setdefault("appworld", {})["save"] = output
        return output

    def _resolve_env_endpoint(self, context: AgentHookContext) -> str:
        env_endpoint = self._env_endpoint or context.resolve_endpoint(
            "env_endpoint",
            "environment_endpoint",
            "env_url",
            "environment_url",
        )
        if not env_endpoint:
            raise ValueError("env_endpoint is required for AppWorld save")
        return env_endpoint


def _extract_appworld_meta(context: AgentHookContext) -> Dict[str, Any]:
    metadata = context.metadata or {}
    appworld_meta = metadata.get("appworld") if isinstance(metadata, dict) else None
    if isinstance(appworld_meta, dict):
        return dict(appworld_meta)
    return {}


def _normalize_ground_truth_mode(payload: Dict[str, Any]) -> None:
    value = payload.get("ground_truth_mode")
    if value is None:
        return
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ALLOWED_GROUND_TRUTH_MODES:
            payload["ground_truth_mode"] = normalized
            return
        if normalized == "auto":
            payload["ground_truth_mode"] = "minimal"
            return
    payload.pop("ground_truth_mode", None)


def _post_appworld(
    endpoint: str,
    method: str,
    payload: Dict[str, Any],
    timeout_s: int,
    requester: Optional[Callable[[str, Dict[str, Any]], Any]],
) -> Any:
    if callable(requester):
        return requester(method, payload)
    response = requests.post(
        f"{endpoint.rstrip('/')}/{method}",
        json=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def _unwrap_output(response: Any) -> Any:
    if isinstance(response, dict) and "output" in response:
        return response.get("output")
    return response


register_hook_aliases(
    ("appworld_initialize", "appworld_init", "appworld_pre"),
    AppWorldInitializeHook,
)
register_hook_aliases(
    ("appworld_save", "appworld_post"),
    AppWorldSaveHook,
)
