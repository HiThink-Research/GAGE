from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gage_eval.agent_runtime.contracts.failure import FailureEnvelope

if TYPE_CHECKING:
    from gage_eval.agent_runtime.session import AgentRuntimeSession


@runtime_checkable
class ClientSurface(Protocol):
    """Defines the standardized installed-client execution contract."""

    def setup(
        self,
        environment: dict[str, Any],
        session: AgentRuntimeSession,
    ) -> dict[str, Any] | None: ...

    def run(
        self,
        request: dict[str, Any],
        environment: dict[str, Any],
    ) -> dict[str, Any]: ...

    async def arun(
        self,
        request: dict[str, Any],
        environment: dict[str, Any],
    ) -> dict[str, Any]: ...


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InstalledClientCommandConfig(_StrictModel):
    """Command-array configuration for external installed clients."""

    command: list[str] = Field(min_length=1)
    timeout_s: int | None = Field(default=None, ge=1, le=3600)

    @field_validator("command")
    @classmethod
    def _command_items_must_be_non_empty(cls, value: list[str]) -> list[str]:
        normalized = [str(item) for item in value]
        if any(not item.strip() for item in normalized):
            raise ValueError("installed_client.client.command items must be non-empty")
        return normalized


class InstalledClientSchedulerConfig(_StrictModel):
    """Scheduler config shape for installed-client execution."""

    client: InstalledClientCommandConfig | None = None


class AcpClientCapabilities(_StrictModel):
    """Phase-1 ACP capability declaration."""

    tools: bool = False
    streaming: bool = False


class AcpClientConfig(_StrictModel):
    """ACP client endpoint and capability config."""

    endpoint: str
    capabilities: AcpClientCapabilities = Field(default_factory=AcpClientCapabilities)

    @field_validator("endpoint")
    @classmethod
    def _endpoint_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("acp_client.client.endpoint is required")
        return value.strip()


class AcpClientSchedulerConfig(_StrictModel):
    """Scheduler config shape for ACP clients."""

    client: AcpClientConfig


class ExternalClientEnvironmentHandle(_StrictModel):
    """Serializable environment handle exposed to external clients."""

    task_id: str
    sample_id: str
    trial_id: str
    env_id: str
    provider: str
    profile_id: str
    transport: Literal["mounted_workdir", "stdio_jsonrpc_proxy", "http_jsonrpc_proxy"]
    workdir: str | None = None
    artifact_dir: str
    env_vars: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    tool_endpoint: str | None = None


class ExternalClientSessionContext(_StrictModel):
    """Sanitized session context visible to external clients."""

    session_id: str
    run_id: str
    task_id: str
    sample_id: str
    benchmark_kit_id: str
    scheduler_type: str
    client_id: str | None = None
    trial_id: str
    environment_handle: ExternalClientEnvironmentHandle


class ClientEnvironmentProjectionError(RuntimeError):
    """Stable projection error carrying the required failure code."""

    def __init__(self, code: str, summary: str, *, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.details = dict(details or {})
        super().__init__(summary)


_ALLOWED_HANDLE_FIELDS = set(ExternalClientEnvironmentHandle.model_fields)
_RAW_PROVIDER_FIELDS = {
    "container_id",
    "container_name",
    "e2b_sandbox_id",
    "provider_raw_id",
    "raw_provider_id",
    "resource_id",
    "runtime_handle",
    "sandbox_id",
    "sandbox_provider",
}


def build_external_client_environment_handle(
    *,
    session: Any,
    environment_lease: Any | None = None,
    resource_lease: Any | None = None,
    lease_binding: Any | None = None,
    requested_fields: Any | None = None,
    proxy_starter: Callable[[dict[str, Any]], Any] | None = None,
) -> ExternalClientEnvironmentHandle:
    """Project the current trial environment into the external-client contract."""

    _validate_requested_environment_fields(requested_fields)
    resolved_resource_lease = (
        resource_lease
        or getattr(lease_binding, "resource_lease", None)
        or getattr(session, "resource_lease", None)
    )
    resolved_environment_lease = environment_lease or getattr(lease_binding, "environment_lease", None)
    provider = _resolve_provider(resolved_resource_lease, resolved_environment_lease)
    profile_id = _resolve_profile_id(resolved_resource_lease, resolved_environment_lease)
    env_id = _resolve_env_id(session=session, profile_id=profile_id)
    handle_ref = _mapping(getattr(resolved_resource_lease, "handle_ref", None))
    endpoints = _mapping(getattr(resolved_resource_lease, "endpoints", None))
    resource_metadata = _mapping(getattr(resolved_resource_lease, "metadata", None))
    environment_profile = _mapping(resource_metadata.get("environment_profile"))
    provider_config = _mapping(resource_metadata.get("provider_config") or environment_profile.get("config"))
    startup_env = _mapping(environment_profile.get("startup_env"))
    artifact_dir = str(_mapping(getattr(session, "artifact_layout", None)).get("artifacts_dir") or "")

    if provider == "unknown" and resolved_resource_lease is None and resolved_environment_lease is None:
        return ExternalClientEnvironmentHandle(
            task_id=str(getattr(session, "task_id", "")),
            sample_id=str(getattr(session, "sample_id", "")),
            trial_id=str(
                _mapping(getattr(session, "runtime_context", None)).get("trial_id")
                or _mapping(getattr(session, "scheduler_state", None)).get("trial_id")
                or "trial_0001"
            ),
            env_id=env_id,
            provider="none",
            profile_id="none",
            transport="mounted_workdir",
            workdir=_resolve_workdir(session=session, provider_config=provider_config, handle_ref=handle_ref),
            artifact_dir=artifact_dir,
            env_vars={},
            capabilities=[],
        )

    if provider in {"docker", "local_process"}:
        return ExternalClientEnvironmentHandle(
            task_id=str(getattr(session, "task_id", "")),
            sample_id=str(getattr(session, "sample_id", "")),
            trial_id=str(
                _mapping(getattr(session, "runtime_context", None)).get("trial_id")
                or _mapping(getattr(session, "scheduler_state", None)).get("trial_id")
                or "trial_0001"
            ),
            env_id=env_id,
            provider=provider,
            profile_id=profile_id,
            transport="mounted_workdir",
            workdir=_resolve_workdir(session=session, provider_config=provider_config, handle_ref=handle_ref),
            artifact_dir=artifact_dir,
            env_vars=_string_map(startup_env),
            capabilities=_resolve_capabilities(resolved_environment_lease),
        )

    if provider == "e2b":
        endpoint = _resolve_proxy_endpoint(handle_ref, endpoints)
        if not endpoint:
            raise ClientEnvironmentProjectionError(
                "client_execution.client_environment_projection_unsupported",
                "external client environment projection is unsupported for e2b without a JSON-RPC proxy",
                details={"provider": provider, "profile_id": profile_id},
            )
        handle = ExternalClientEnvironmentHandle(
            task_id=str(getattr(session, "task_id", "")),
            sample_id=str(getattr(session, "sample_id", "")),
            trial_id=str(
                _mapping(getattr(session, "runtime_context", None)).get("trial_id")
                or _mapping(getattr(session, "scheduler_state", None)).get("trial_id")
                or "trial_0001"
            ),
            env_id=env_id,
            provider=provider,
            profile_id=profile_id,
            transport="http_jsonrpc_proxy" if endpoint.startswith(("http://", "https://")) else "stdio_jsonrpc_proxy",
            workdir=_resolve_workdir(session=session, provider_config=provider_config, handle_ref=handle_ref),
            artifact_dir=artifact_dir,
            env_vars=_string_map(startup_env),
            capabilities=_resolve_capabilities(resolved_environment_lease),
            tool_endpoint=endpoint,
        )
        if proxy_starter is not None:
            try:
                proxy_starter(handle.model_dump(mode="json"))
            except Exception as exc:
                raise ClientEnvironmentProjectionError(
                    "client_execution.client_environment_projection_failed",
                    "external client environment proxy startup failed",
                    details={"provider": provider, "profile_id": profile_id, "error": str(exc)},
                ) from exc
        return handle

    raise ClientEnvironmentProjectionError(
        "client_execution.client_environment_projection_unsupported",
        "external client environment projection is unsupported for this provider",
        details={"provider": provider, "profile_id": profile_id},
    )


def validate_external_client_environment_handle_payload(
    value: Mapping[str, Any],
    *,
    requested_fields: Any | None = None,
) -> ExternalClientEnvironmentHandle:
    """Validate a cached external-client handle before reuse."""

    _validate_requested_environment_fields(requested_fields)
    try:
        return ExternalClientEnvironmentHandle.model_validate(dict(value))
    except ValidationError as exc:
        raise ClientEnvironmentProjectionError(
            "client_execution.client_environment_projection_denied",
            "cached external client environment handle is outside the projection contract",
            details={"errors": exc.errors(include_url=False)},
        ) from exc


def build_external_client_session_context(
    *,
    session: Any,
    environment: Mapping[str, Any],
) -> ExternalClientSessionContext:
    """Build the only session view exposed to external clients."""

    handle = validate_external_client_environment_handle_payload(
        _mapping(environment.get("environment_handle"))
    )
    trial_id = str(
        _mapping(getattr(session, "runtime_context", None)).get("trial_id")
        or _mapping(getattr(session, "scheduler_state", None)).get("trial_id")
        or handle.trial_id
        or "trial_0001"
    )
    return ExternalClientSessionContext(
        session_id=str(getattr(session, "session_id", "")),
        run_id=str(getattr(session, "run_id", "")),
        task_id=str(getattr(session, "task_id", "")),
        sample_id=str(getattr(session, "sample_id", "")),
        benchmark_kit_id=str(getattr(session, "benchmark_kit_id", "")),
        scheduler_type=str(getattr(session, "scheduler_type", "")),
        client_id=getattr(session, "client_id", None),
        trial_id=trial_id,
        environment_handle=handle,
    )


def projection_failure_envelope(
    error: ClientEnvironmentProjectionError,
    *,
    component_id: str,
    first_bad_step: str,
    suspect_files: tuple[str, ...],
) -> FailureEnvelope:
    """Build the stable failure envelope for projection failures."""

    return FailureEnvelope(
        failure_domain="client_execution",
        failure_stage="run_scheduler",
        failure_code=error.code,
        component_kind="scheduler",
        component_id=component_id,
        owner="runtime_scheduler_core",
        retryable=False,
        summary=str(error),
        first_bad_step=first_bad_step,
        suspect_files=suspect_files,
        details=dict(error.details or {}),
    )


def _validate_requested_environment_fields(value: Any | None) -> None:
    if value is None:
        return
    requested = [str(item) for item in value] if isinstance(value, list) else [str(value)]
    denied = sorted(
        field
        for field in requested
        if field in _RAW_PROVIDER_FIELDS or field not in _ALLOWED_HANDLE_FIELDS
    )
    if denied:
        raise ClientEnvironmentProjectionError(
            "client_execution.client_environment_projection_denied",
            "external client requested environment fields outside the projected handle",
            details={"requested_fields": requested, "denied_fields": denied},
        )


def _resolve_provider(resource_lease: Any | None, environment_lease: Any | None) -> str:
    provider = getattr(environment_lease, "provider", None) or getattr(resource_lease, "resource_kind", None)
    return str(provider or "").strip() or "unknown"


def _resolve_profile_id(resource_lease: Any | None, environment_lease: Any | None) -> str:
    profile_id = getattr(environment_lease, "profile_id", None) or getattr(resource_lease, "profile_id", None)
    return str(profile_id or "").strip() or "unknown"


def _resolve_env_id(*, session: Any, profile_id: str) -> str:
    for value in (
        _mapping(getattr(session, "runtime_context", None)).get("env_id"),
        _mapping(getattr(session, "scheduler_state", None)).get("env_id"),
        profile_id,
        getattr(session, "benchmark_kit_id", None),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _resolve_workdir(
    *,
    session: Any,
    provider_config: Mapping[str, Any],
    handle_ref: Mapping[str, Any],
) -> str | None:
    for value in (
        handle_ref.get("mounted_workdir"),
        handle_ref.get("workdir"),
        provider_config.get("exec_workdir"),
        provider_config.get("workdir"),
        _mapping(getattr(session, "runtime_context", None)).get("cwd"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_capabilities(environment_lease: Any | None) -> list[str]:
    environment = getattr(environment_lease, "environment", None)
    capabilities = getattr(environment, "capabilities", None)
    if hasattr(capabilities, "model_dump"):
        payload = capabilities.model_dump(mode="python")
    elif isinstance(capabilities, Mapping):
        payload = dict(capabilities)
    else:
        return []
    return sorted(str(key) for key, value in payload.items() if value)


def _resolve_proxy_endpoint(
    handle_ref: Mapping[str, Any],
    endpoints: Mapping[str, Any],
) -> str | None:
    for key in ("jsonrpc_proxy_endpoint", "tool_endpoint", "environment_proxy_endpoint"):
        value = handle_ref.get(key) or endpoints.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items() if key is not None and item is not None}
