"""Environment manager for AgentKit v2 sample-scoped leases."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel

from gage_eval.environment.contracts import (
    DEFAULT_EXEC_STREAM_LIMIT_BYTES,
    BaseEnvironment,
)
from gage_eval.environment.errors import (
    EnvironmentAttachError,
    EnvironmentCreateError,
    EnvironmentError,
    EnvironmentExecError,
    EnvironmentPreflightError,
    EnvironmentTimeoutError,
    EnvironmentTransferError,
)
from gage_eval.environment.lease import EnvironmentLease
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.providers.registry import ProviderRegistry
from gage_eval.environment.resources import EnvironmentResources


RetryBackoff = Callable[..., Any]


class EnvironmentManagerError(EnvironmentError):
    """Stable manager-level failure with an AgentKit environment code."""

    def __init__(self, code: str, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.__cause__ = cause

    def to_failure_payload(self) -> dict[str, Any]:
        """Return diagnostic details suitable for raw_error artifacts."""

        return {
            "failure_code": self.code,
            "summary": str(self),
            "details": _exception_details(self.__cause__),
        }


DEFAULT_RETRY_BUDGET_BY_FAILURE: dict[type[EnvironmentError], int] = {
    EnvironmentPreflightError: 0,
    EnvironmentCreateError: 2,
    EnvironmentAttachError: 1,
    EnvironmentExecError: 0,
    EnvironmentTransferError: 1,
    EnvironmentTimeoutError: 0,
}


class EnvironmentManager:
    """Acquire and release benchmark-neutral provider environments."""

    def __init__(
        self,
        *,
        registry: ProviderRegistry | None = None,
        trace: Any | None = None,
        backoff: RetryBackoff | None = None,
        artifact_sink: Any | None = None,
        stdout_limit_bytes: int = DEFAULT_EXEC_STREAM_LIMIT_BYTES,
        stderr_limit_bytes: int = DEFAULT_EXEC_STREAM_LIMIT_BYTES,
    ) -> None:
        self._registry = registry or ProviderRegistry()
        self._trace = trace
        self._backoff = backoff
        self._artifact_sink = artifact_sink
        self._stdout_limit_bytes = stdout_limit_bytes
        self._stderr_limit_bytes = stderr_limit_bytes

    @property
    def registry(self) -> ProviderRegistry:
        return self._registry

    async def acquire(
        self,
        *,
        kit_id: str,
        provider: str,
        profile_id: str,
        profile: EnvironmentProfile,
        provider_config: BaseModel,
        resources: EnvironmentResources,
        startup_env: dict[str, str],
        lifecycle: Literal["per_sample", "per_task"],
        metadata: dict[str, Any],
    ) -> EnvironmentLease:
        if lifecycle != "per_sample":
            raise EnvironmentManagerError(
                "environment.lifecycle.unsupported",
                f"lifecycle={lifecycle!r} is reserved for a later AgentKit phase",
            )

        provider_impl = self._registry.get(provider)
        request = {
            "kit_id": kit_id,
            "provider": provider,
            "profile_id": profile_id,
            "profile": profile,
            "provider_config": provider_config,
            "resources": resources,
            "startup_env": dict(startup_env),
            "lifecycle": lifecycle,
            "metadata": dict(metadata),
        }

        await self._preflight(provider_impl, request, provider=provider)
        environment = await self._create_with_retries(provider_impl, request, provider=provider)
        environment = await self._ensure_healthy(provider_impl, environment, request, provider=provider)
        try:
            lease = EnvironmentLease(
                lease_id=f"env-lease-{uuid4().hex}",
                environment=environment,
                provider=provider,
                profile_id=profile_id,
                lifecycle=lifecycle,
                exclusive=True,
                metadata=dict(metadata),
                artifact_sink=self._artifact_sink,
                stdout_limit_bytes=self._stdout_limit_bytes,
                stderr_limit_bytes=self._stderr_limit_bytes,
            )
            self._emit(
                "environment.acquire",
                {"actor": _actor(metadata), "descriptor": lease.to_descriptor()},
                sample_id=_sample_id(metadata),
            )
            return lease
        except Exception:
            await _stop_environment_safely(environment)
            raise

    async def release(self, lease: EnvironmentLease) -> None:
        if lease.lifecycle == "per_sample":
            await lease.environment.stop(delete=True)
            return
        raise EnvironmentManagerError(
            "environment.lifecycle.unsupported",
            f"lifecycle={lease.lifecycle!r} release is reserved for a later AgentKit phase",
        )

    async def _preflight(self, provider_impl: Any, request: dict[str, Any], *, provider: str) -> None:
        preflight = getattr(provider_impl, "preflight", None)
        if preflight is None:
            return
        try:
            await _maybe_await(preflight(**request))
        except Exception as exc:
            raise EnvironmentManagerError(
                "environment.preflight_failed",
                f"provider={provider} profile_id={request['profile_id']}",
                cause=exc,
            ) from exc

    async def _create_with_retries(
        self,
        provider_impl: Any,
        request: dict[str, Any],
        *,
        provider: str,
    ) -> BaseEnvironment:
        create = getattr(provider_impl, "create")
        attempt = 0
        while True:
            try:
                return await _maybe_await(create(**request))
            except Exception as exc:
                retry_budget = _retry_budget(provider_impl, exc)
                if attempt >= retry_budget:
                    code = _failure_code(exc, default_code="environment.create_failed")
                    if retry_budget > 0:
                        code = "environment.unavailable"
                    raise EnvironmentManagerError(
                        code,
                        f"provider={provider} profile_id={request['profile_id']} attempts={attempt + 1}",
                        cause=exc,
                    ) from exc
                attempt += 1
                await self._record_backoff(
                    provider=provider,
                    attempt=attempt,
                    delay_s=round(0.1 * (2 ** (attempt - 1)), 10),
                    error=exc,
                )

    async def _ensure_healthy(
        self,
        provider_impl: Any,
        environment: BaseEnvironment,
        request: dict[str, Any],
        *,
        provider: str,
    ) -> BaseEnvironment:
        health_check = getattr(provider_impl, "health_check", None)
        if health_check is None:
            return environment
        try:
            healthy = await _maybe_await(health_check(environment))
        except Exception as exc:
            await _stop_environment_safely(environment)
            raise EnvironmentManagerError(
                "environment.unavailable",
                f"provider={provider} profile_id={request['profile_id']} health_check_failed",
                cause=exc,
            ) from exc
        if healthy:
            return environment
        old_descriptor = _environment_descriptor(environment)
        await environment.stop(delete=True)
        rebuilt = await self._create_with_retries(provider_impl, request, provider=provider)
        try:
            rebuilt_healthy = await _maybe_await(health_check(rebuilt))
        except Exception as exc:
            await _stop_environment_safely(rebuilt)
            raise EnvironmentManagerError(
                "environment.unavailable",
                f"provider={provider} profile_id={request['profile_id']} rebuild_health_check_failed",
                cause=exc,
            ) from exc
        if not rebuilt_healthy:
            await rebuilt.stop(delete=True)
            raise EnvironmentManagerError(
                "environment.unavailable",
                f"provider={provider} profile_id={request['profile_id']} health_check=false",
            )
        self._emit(
            "environment.rebuild",
            {
                "actor": _actor(request["metadata"]),
                "provider": provider,
                "profile_id": request["profile_id"],
                "old_descriptor": old_descriptor,
                "new_descriptor": _environment_descriptor(rebuilt),
            },
            sample_id=_sample_id(request["metadata"]),
        )
        return rebuilt

    async def _record_backoff(
        self,
        *,
        provider: str,
        attempt: int,
        delay_s: float,
        error: BaseException,
    ) -> None:
        if self._backoff is None:
            return
        await _maybe_await(
            self._backoff(provider=provider, attempt=attempt, delay_s=delay_s, error=error)
        )

    def _emit(self, event: str, payload: dict[str, Any], *, sample_id: str | None = None) -> None:
        if self._trace is None:
            return
        emit = getattr(self._trace, "emit", None)
        if callable(emit):
            try:
                emit(event, payload, sample_id=sample_id)
            except Exception:
                return


def _retry_budget(provider_impl: Any, exc: Exception) -> int:
    provider_policy = getattr(provider_impl, "retry_budget_by_failure", None)
    if isinstance(provider_policy, dict):
        for exc_type, retries in provider_policy.items():
            if isinstance(exc, exc_type):
                return int(retries)
    for exc_type, retries in DEFAULT_RETRY_BUDGET_BY_FAILURE.items():
        if isinstance(exc, exc_type):
            return retries
    return 0


def _failure_code(exc: Exception, *, default_code: str = "environment.failed") -> str:
    if isinstance(exc, EnvironmentPreflightError):
        return "environment.preflight_failed"
    if isinstance(exc, EnvironmentCreateError):
        return "environment.create_failed"
    if isinstance(exc, EnvironmentAttachError):
        return "environment.attach_failed"
    if isinstance(exc, EnvironmentExecError):
        return "environment.exec_failed"
    if isinstance(exc, EnvironmentTransferError):
        return "environment.transfer_failed"
    if isinstance(exc, EnvironmentTimeoutError):
        return "environment.timeout"
    return default_code


def _exception_details(exc: BaseException | None) -> dict[str, Any]:
    if exc is None:
        return {}
    details: dict[str, Any] = {
        "cause_type": exc.__class__.__name__,
        "cause_message": str(exc),
    }
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            details["validation_errors"] = errors()
        except Exception:
            pass
    nested = getattr(exc, "__cause__", None)
    if nested is not None and nested is not exc:
        details["cause"] = _exception_details(nested)
    return details


async def _stop_environment_safely(environment: BaseEnvironment) -> None:
    try:
        await environment.stop(delete=True)
    except Exception:
        return


def _environment_descriptor(environment: BaseEnvironment) -> dict[str, Any]:
    capabilities = getattr(environment, "capabilities", None)
    if hasattr(capabilities, "model_dump"):
        capabilities_payload = capabilities.model_dump(mode="python")
    elif isinstance(capabilities, dict):
        capabilities_payload = dict(capabilities)
    else:
        capabilities_payload = {}
    return {
        "env_id": getattr(environment, "env_id", None),
        "name": getattr(environment, "name", None),
        "provider": getattr(environment, "provider", None),
        "capabilities": capabilities_payload,
        "metadata": dict(getattr(environment, "metadata", {}) or {}),
    }


def _actor(metadata: dict[str, Any]) -> str:
    return str(metadata.get("actor") or "environment_manager")


def _sample_id(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("sample_id")
    return str(value) if value is not None else None


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
