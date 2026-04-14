from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import requests

from gage_eval.agent_runtime.clients.service_contract import (
    InstalledClientServiceRequest,
    InstalledClientServiceResult,
)
from gage_eval.agent_runtime.clients.types import ClientRunRequest, ClientRunResult


@dataclass(frozen=True)
class _ArtifactTargets:
    """Captures the local artifact paths owned by one runtime sample."""

    stdout_path: str | None
    patch_path: str | None
    trajectory_path: str | None


class CodexClient:
    """Calls one external Codex-compatible installed-client service over HTTP."""

    def __init__(
        self,
        *,
        service_url: str | None = None,
        auth_token: str | None = None,
        default_timeout_sec: int = 1800,
        session: requests.Session | None = None,
    ) -> None:
        """Initializes the HTTP-backed installed client.

        Args:
            service_url: Optional service URL override. When omitted, the client
                resolves the URL from request/environment/env-var inputs.
            auth_token: Optional bearer token override for the installed-client
                service. When omitted, the client resolves it from env vars.
            default_timeout_sec: Default request timeout for the service call.
            session: Optional requests session override for tests.
        """

        self._service_url = service_url
        self._auth_token = auth_token
        self._default_timeout_sec = default_timeout_sec
        self._session = session or requests.Session()

    def setup(self, environment: dict[str, Any], session: Any) -> dict[str, Any] | None:
        """Projects the resolved service URL into the per-sample environment."""

        service_url = _resolve_service_url(
            service_url=self._service_url,
            request_metadata={},
            environment=environment,
        )
        if not service_url:
            return None
        return {"client_service_url": service_url}

    def run(
        self,
        request: ClientRunRequest | Mapping[str, Any],
        environment: dict[str, Any] | Any,
    ) -> ClientRunResult:
        """Calls the external service and persists the normalized local artifacts.

        Args:
            request: Normalized or raw runtime request payload.
            environment: Runtime environment projection from the scheduler.

        Returns:
            The normalized client result payload.

        Raises:
            RuntimeError: If the service URL is missing, the HTTP call fails, or
                the response body is invalid.
        """

        normalized_request = ClientRunRequest.from_payload(request)
        normalized_environment = _normalize_environment(environment)
        service_url = _resolve_service_url(
            service_url=self._service_url,
            request_metadata=normalized_request.metadata,
            environment=normalized_environment,
        )
        if not service_url:
            raise RuntimeError("codex_service_url_missing")

        timeout_sec = _coerce_timeout(
            normalized_request.metadata.get("timeout_sec"),
            default=self._default_timeout_sec,
        )
        response_payload, response_headers = _call_service(
            session=self._session,
            service_url=service_url,
            auth_token=_resolve_auth_token(self._auth_token),
            request=normalized_request,
            environment=normalized_environment,
            timeout_sec=timeout_sec,
        )
        service_result = InstalledClientServiceResult.from_payload(
            _unwrap_service_result(response_payload)
        )
        targets = _resolve_artifact_targets(
            request=normalized_request,
            environment=normalized_environment,
        )
        artifact_paths = _persist_local_artifacts(
            targets=targets,
            service_result=service_result,
        )
        metadata = dict(service_result.metadata or {})
        metadata.setdefault("client_id", "codex")
        metadata.setdefault("service_url", service_url)
        request_id = (
            response_headers.get("x-request-id")
            or response_headers.get("X-Request-Id")
            or metadata.get("request_id")
        )
        if request_id:
            metadata["request_id"] = str(request_id)

        stdout = service_result.stdout
        stderr = service_result.stderr
        answer = service_result.answer or stdout.strip()
        patch_content = _coerce_optional_string(service_result.patch_content)
        patch_path = targets.patch_path if patch_content and targets.patch_path else None
        trajectory_path = targets.trajectory_path if targets.trajectory_path else None
        agent_trace = list(service_result.agent_trace)
        exit_code = _coerce_exit_code(service_result.exit_code)
        status = _normalize_status(service_result.status, exit_code=exit_code)

        return ClientRunResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            answer=answer,
            status=status,
            patch_path=patch_path,
            patch_content=patch_content,
            trajectory_path=trajectory_path,
            artifact_paths=artifact_paths,
            agent_trace=agent_trace,
            metadata=metadata,
            usage=dict(service_result.usage or {}),
        )


def _resolve_service_url(
    *,
    service_url: str | None,
    request_metadata: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> str | None:
    """Resolves the installed-client service URL from stable override inputs."""

    candidates = (
        service_url,
        _coerce_optional_string(request_metadata.get("service_url")),
        _coerce_optional_string(request_metadata.get("client_service_url")),
        _coerce_optional_string(environment.get("client_service_url")),
        _coerce_optional_string(environment.get("service_url")),
        os.getenv("GAGE_CODEX_CLIENT_URL"),
        os.getenv("CODEX_CLIENT_URL"),
        os.getenv("GAGE_INSTALLED_CLIENT_URL"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        normalized = candidate.rstrip("/")
        if normalized.endswith("/run"):
            return normalized
        return f"{normalized}/run"
    return None


def _call_service(
    *,
    session: requests.Session,
    service_url: str,
    auth_token: str | None,
    request: ClientRunRequest,
    environment: dict[str, Any],
    timeout_sec: int,
) -> tuple[dict[str, Any], Mapping[str, str]]:
    """Calls the external installed-client service and validates the response."""

    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    try:
        response = session.post(
            service_url,
            json=InstalledClientServiceRequest(
                request=request,
                environment=environment,
            ).to_dict(),
            headers=headers,
            timeout=timeout_sec,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"codex_service_request_failed:{exc.__class__.__name__}:{exc}") from exc

    if response.status_code >= 400:
        body = response.text.strip()
        raise RuntimeError(
            f"codex_service_http_error:{response.status_code}:{body or 'empty_response'}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("codex_service_invalid_json_response") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("codex_service_invalid_payload")
    return payload, response.headers


def _unwrap_service_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Unwraps the service payload into the canonical client result shape."""

    result = payload.get("result")
    if isinstance(result, Mapping):
        return dict(result)
    return dict(payload)


def _resolve_auth_token(token: str | None) -> str | None:
    """Resolves the optional installed-client bearer token."""

    candidates = (
        token,
        os.getenv("GAGE_CODEX_CLIENT_TOKEN"),
        os.getenv("CODEX_CLIENT_TOKEN"),
        os.getenv("GAGE_INSTALLED_CLIENT_TOKEN"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _normalize_environment(environment: dict[str, Any] | Any) -> dict[str, Any]:
    """Drops non-JSON runtime objects before sending the service request."""

    if not isinstance(environment, Mapping):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in environment.items():
        if key == "sandbox_provider":
            continue
        sanitized[str(key)] = _to_json_compatible(value)
    return sanitized


def _to_json_compatible(value: Any) -> Any:
    """Converts runtime payload values into a JSON-safe structure."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_compatible(item) for key, item in value.items() if key is not None}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _to_json_compatible(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


def _resolve_artifact_targets(
    *,
    request: ClientRunRequest,
    environment: Mapping[str, Any],
) -> _ArtifactTargets:
    """Resolves the local runtime-owned artifact targets for one sample."""

    metadata = request.metadata or {}
    artifact_layout = dict(environment.get("artifact_layout") or {})
    sample_root = Path(str(artifact_layout.get("sample_root") or "."))
    artifacts_dir = Path(str(artifact_layout.get("artifacts_dir") or sample_root / "artifacts"))
    stdout_path = _resolve_optional_path(
        metadata,
        ("stdout_path", "stdout_file", "output_path", "output_last_message_path"),
    ) or str(artifacts_dir / "stdout.log")
    trajectory_path = _resolve_optional_path(
        metadata,
        ("trajectory_path", "trajectory_log_path"),
    ) or str(artifacts_dir / "trajectory.log")
    patch_path = _resolve_optional_path(
        metadata,
        ("patch_path", "submission_patch_path"),
    )
    if not patch_path:
        submission_contract = metadata.get("submission_contract")
        if isinstance(submission_contract, str) and submission_contract.strip():
            patch_path = str(sample_root / submission_contract.strip())
    return _ArtifactTargets(
        stdout_path=stdout_path,
        patch_path=patch_path,
        trajectory_path=trajectory_path,
    )


def _persist_local_artifacts(
    *,
    targets: _ArtifactTargets,
    service_result: InstalledClientServiceResult,
) -> dict[str, str]:
    """Persists service-returned evidence under the runtime-owned sample root."""

    artifact_paths: dict[str, str] = {}
    stdout = service_result.stdout
    if targets.stdout_path:
        _write_text_file(targets.stdout_path, stdout)
        artifact_paths["stdout"] = targets.stdout_path

    patch_content = _coerce_optional_string(service_result.patch_content)
    if targets.patch_path and patch_content:
        _write_text_file(targets.patch_path, patch_content)
        artifact_paths["submission_patch"] = targets.patch_path

    trajectory_text = _resolve_trajectory_text(service_result)
    if targets.trajectory_path and trajectory_text:
        _write_text_file(targets.trajectory_path, trajectory_text)
        artifact_paths["trajectory"] = targets.trajectory_path

    return artifact_paths


def _resolve_trajectory_text(service_result: InstalledClientServiceResult) -> str:
    """Builds one stable human-readable trajectory artifact."""

    direct = _coerce_optional_string(service_result.trajectory_text)
    if direct:
        return direct
    agent_trace = list(service_result.agent_trace)
    if agent_trace:
        return json.dumps(agent_trace, ensure_ascii=False, indent=2)
    stdout = service_result.stdout
    stderr = service_result.stderr
    if stdout or stderr:
        parts = []
        if stdout:
            parts.append("[stdout]\n" + stdout)
        if stderr:
            parts.append("[stderr]\n" + stderr)
        return "\n\n".join(parts)
    return ""


def _normalize_agent_trace(value: Any) -> list[dict[str, Any]]:
    """Normalizes the optional agent trace payload into a stable list."""

    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            normalized.append({str(key): _to_json_compatible(entry) for key, entry in item.items()})
    return normalized


def _resolve_optional_path(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    """Returns the first non-empty string path declared in metadata."""

    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _write_text_file(path: str, content: str) -> None:
    """Writes one UTF-8 text artifact to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content or "", encoding="utf-8")


def _coerce_timeout(value: Any, *, default: int) -> int:
    """Coerces one timeout override into a positive integer."""

    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _coerce_optional_string(value: Any) -> str | None:
    """Returns one stripped string when the value is non-empty."""

    if isinstance(value, str) and value.strip():
        return value
    return None


def _coerce_exit_code(value: Any) -> int:
    """Normalizes the service exit code field."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_status(value: Any, *, exit_code: int) -> str:
    """Normalizes the scheduler status using the service signal and exit code."""

    if isinstance(value, str) and value in {"completed", "failed", "aborted"}:
        return value
    return "completed" if exit_code == 0 else "failed"
