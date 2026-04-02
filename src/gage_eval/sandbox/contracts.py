"""Formal sandbox contracts and normalization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

RemoteSandboxMode = str


@dataclass(frozen=True)
class RemoteSandboxContract:
    """Declarative remote sandbox contract shared across consumers."""

    mode: RemoteSandboxMode
    sandbox_profile_id: Optional[str] = None
    provider: Optional[str] = None
    sandbox_id: Optional[str] = None
    image: Optional[str] = None
    workspace_root: Optional[str] = None
    attach_target: Optional[str] = None
    control_endpoint: Optional[str] = None
    exec_endpoint: Optional[str] = None
    file_endpoint: Optional[str] = None
    file_read_url: Optional[str] = None
    file_write_url: Optional[str] = None
    env_endpoint: Optional[str] = None
    apis_endpoint: Optional[str] = None
    mcp_endpoint: Optional[str] = None
    renew_supported: bool = False
    auth: Dict[str, Any] = field(default_factory=dict)
    timeouts: Dict[str, Any] = field(default_factory=dict)
    retries: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_remote_sandbox_contract(self)


@dataclass(frozen=True)
class RemoteSandboxHandle:
    """Serializable runtime handle exposed after a sandbox is started."""

    mode: RemoteSandboxMode
    status: str = "ready"
    sandbox_id: Optional[str] = None
    lease_id: Optional[str] = None
    provider: Optional[str] = None
    workspace_root: Optional[str] = None
    attach_target: Optional[str] = None
    exec_url: Optional[str] = None
    data_endpoint: Optional[str] = None
    file_read_url: Optional[str] = None
    file_write_url: Optional[str] = None
    env_endpoint: Optional[str] = None
    apis_endpoint: Optional[str] = None
    mcp_endpoint: Optional[str] = None
    surface_names: tuple[str, ...] = ()
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def validate_remote_sandbox_contract(contract: RemoteSandboxContract) -> None:
    """Validate required invariants for attached and managed sandboxes."""

    mode = str(contract.mode or "").strip().lower()
    if mode not in {"attached", "managed"}:
        raise ValueError(f"unsupported remote sandbox mode: {contract.mode!r}")
    if mode == "managed" and not contract.control_endpoint:
        raise ValueError("managed mode requires control_endpoint")
    if mode == "attached" and not contract.exec_endpoint:
        raise ValueError("attached mode requires exec_endpoint")
    if contract.attach_target is not None and not str(contract.attach_target).strip():
        raise ValueError("attach_target must be non-empty when provided")


def merge_sandbox_profile_layers(
    profiles: Optional[Mapping[str, Mapping[str, Any]]],
    profile_id: Optional[str],
    *layers: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve sandbox profile defaults and merge override layers."""

    merged: dict[str, Any] = {}
    if profile_id:
        profile_payload = (profiles or {}).get(str(profile_id))
        if isinstance(profile_payload, Mapping):
            merged = deep_merge_dicts(merged, profile_payload)
    for layer in layers:
        if isinstance(layer, Mapping):
            merged = deep_merge_dicts(merged, layer)
    return merged


def deep_merge_dicts(
    base: Optional[Mapping[str, Any]],
    override: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Deep-merge mappings while preserving nested dict structure."""

    merged: dict[str, Any] = dict(base or {})
    if not isinstance(override, Mapping):
        return merged
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[str(key)] = deep_merge_dicts(current, value)
        else:
            merged[str(key)] = _clone_json_like(value)
    return merged


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping into a plain dict."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _clone_json_like(item) for key, item in value.items()}
    return {}


def coerce_remote_sandbox_handle(payload: Mapping[str, Any]) -> RemoteSandboxHandle:
    """Build a normalized handle from an arbitrary runtime payload."""

    if isinstance(payload, RemoteSandboxHandle):
        return payload
    raw = dict(payload or {})
    metadata = raw.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    surface_names = raw.get("surface_names") or ()
    if isinstance(surface_names, str):
        surface_names = (surface_names,)
    return RemoteSandboxHandle(
        mode=str(raw.get("mode") or raw.get("remote_mode") or "attached"),
        status=str(raw.get("status") or "ready"),
        sandbox_id=_optional_string(raw.get("sandbox_id")),
        lease_id=_optional_string(raw.get("lease_id")),
        provider=_optional_string(raw.get("provider")),
        workspace_root=_optional_string(raw.get("workspace_root")),
        attach_target=_optional_string(raw.get("attach_target")),
        exec_url=_optional_string(raw.get("exec_url")),
        data_endpoint=_optional_string(raw.get("data_endpoint")),
        file_read_url=_optional_string(raw.get("file_read_url")),
        file_write_url=_optional_string(raw.get("file_write_url")),
        env_endpoint=_optional_string(raw.get("env_endpoint") or raw.get("environment_endpoint")),
        apis_endpoint=_optional_string(raw.get("apis_endpoint")),
        mcp_endpoint=_optional_string(raw.get("mcp_endpoint")),
        surface_names=tuple(str(name) for name in surface_names),
        expires_at=_optional_string(raw.get("expires_at")),
        metadata={str(key): _clone_json_like(item) for key, item in metadata.items()},
    )


def serialize_handle(handle: RemoteSandboxHandle | Mapping[str, Any]) -> dict[str, Any]:
    """Serialize a runtime handle into a JSON-safe dict."""

    if isinstance(handle, RemoteSandboxHandle):
        payload = asdict(handle)
    else:
        payload = dataclass_to_dict(handle)
    return {str(key): _clone_json_like(value) for key, value in payload.items() if value is not None}


def _clone_json_like(value: Any) -> Any:
    if is_dataclass(value):
        return dataclass_to_dict(value)
    if isinstance(value, Mapping):
        return {str(key): _clone_json_like(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_json_like(item) for item in value)
    if isinstance(value, list):
        return [_clone_json_like(item) for item in value]
    if isinstance(value, set):
        return sorted(_clone_json_like(item) for item in value)
    return value


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text or None

