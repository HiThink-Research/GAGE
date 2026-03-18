"""Backend registry helpers."""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from gage_eval.role.model.backends.base_backend import Backend, build_backend_error_result
from gage_eval.role.model.runtime import HttpRetryMixin
from gage_eval.registry.utils import ensure_async

from .builder import build_backend  # re-export

__all__ = ["wrap_backend", "build_backend"]

_MISSING = object()


def wrap_backend(backend: Backend) -> Backend:
    """Wrap backend with async + retry capabilities based on metadata."""

    wrapped: Backend = _AsyncBackendProxy(backend)
    if getattr(backend, "transport", None) == "http" or hasattr(backend, "http_retry_params"):
        params = getattr(backend, "http_retry_params", {}) or {}
        attempts = int(params.get("attempts", 3))
        interval = float(params.get("interval", 1.0))
        wrapped = _HttpRetryBackendProxy(wrapped, attempts=attempts, interval=interval)
    return _ErrorNormalizingBackendProxy(wrapped)


class _AsyncBackendProxy(Backend):
    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        super().__init__(backend.config)

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        backend_call = getattr(self._backend, "ainvoke", None)
        if backend_call:
            return await backend_call(payload)

        return await ensure_async(self._backend)(payload)


class _HttpRetryBackendProxy(Backend, HttpRetryMixin):
    def __init__(self, backend: Backend, *, attempts: int, interval: float) -> None:
        self._backend = backend
        self._http_retry_attempts = max(1, attempts)
        self._http_retry_interval = max(0.0, interval)
        super().__init__(backend.config)

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async def caller(request: Dict[str, Any]) -> Dict[str, Any]:
            backend_call = getattr(self._backend, "ainvoke", None)
            if backend_call:
                return await backend_call(request)

            return await ensure_async(self._backend)(request)

        return await self._ainvoke_with_retry(payload, caller=caller)


class _ErrorNormalizingBackendProxy(Backend):
    """Proxy that converts backend exceptions into the shared error payload."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        super().__init__(backend.config)

    def __getattr__(self, name: str) -> Any:
        return _resolve_backend_attr(self._backend, name)

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            backend_call = getattr(self._backend, "ainvoke", None)
            if backend_call:
                return await backend_call(payload)

            return await ensure_async(self._backend)(payload)
        except Exception as exc:
            backend_name = _resolve_backend_name(self._backend)
            logger.error(
                "Backend {} invocation failed error_type={} error={}",
                backend_name,
                type(exc).__name__,
                exc,
            )
            return build_backend_error_result(exc, backend_name=backend_name)

    def generate_batch(self, payloads: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        batch_call = _resolve_backend_attr(self._backend, "generate_batch")
        try:
            return batch_call(payloads)
        except Exception as exc:
            backend_name = _resolve_backend_name(self._backend)
            logger.error(
                "Backend {} batch invocation failed error_type={} error={}",
                backend_name,
                type(exc).__name__,
                exc,
            )
            error_result = build_backend_error_result(exc, backend_name=backend_name)
            return [dict(error_result) for _ in payloads]


def _resolve_backend_name(backend: Backend) -> str:
    """Resolve the leaf backend class name through wrapper proxies."""

    target = backend
    while isinstance(getattr(target, "_backend", None), Backend):
        next_backend = getattr(target, "_backend")
        if next_backend is target:
            break
        target = next_backend
    return target.__class__.__name__


def _resolve_backend_attr(backend: Backend, attr_name: str) -> Any:
    """Resolve an attribute from the leaf backend through wrapper proxies."""

    target = backend
    while True:
        value = getattr(target, attr_name, _MISSING)
        if value is not _MISSING:
            return value
        next_backend = getattr(target, "_backend", None)
        if not isinstance(next_backend, Backend) or next_backend is target:
            break
        target = next_backend
    raise AttributeError(attr_name)
