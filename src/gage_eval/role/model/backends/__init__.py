"""Backend registry helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict

from gage_eval.role.model.backends.base_backend import Backend
from gage_eval.role.model.runtime import HttpRetryMixin

from .builder import build_backend  # re-export

__all__ = ["wrap_backend", "build_backend"]


def wrap_backend(backend: Backend) -> Backend:
    """Wrap backend with async + retry capabilities based on metadata."""

    wrapped: Backend = _AsyncBackendProxy(backend)
    if getattr(backend, "transport", None) == "http" or hasattr(backend, "http_retry_params"):
        params = getattr(backend, "http_retry_params", {}) or {}
        attempts = int(params.get("attempts", 3))
        interval = float(params.get("interval", 1.0))
        wrapped = _HttpRetryBackendProxy(wrapped, attempts=attempts, interval=interval)
    return wrapped


class _AsyncBackendProxy(Backend):
    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        super().__init__(backend.config)

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        backend_call = getattr(self._backend, "ainvoke", None)
        if backend_call:
            return await backend_call(payload)
        from gage_eval.registry import ensure_async

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
            from gage_eval.registry import ensure_async

            return await ensure_async(self._backend)(request)

        return await self._ainvoke_with_retry(payload, caller=caller)
