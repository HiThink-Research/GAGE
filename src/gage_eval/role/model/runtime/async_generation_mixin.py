"""Async helper mixin used by ModelRoleAdapter backends."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.registry.utils import ensure_async


class AsyncGenerationMixin:
    """Provides an overridable wrapper that executes backend calls asynchronously."""

    async def _ainvoke_backend(self, request: Dict[str, Any]) -> Dict[str, Any]:
        backend_call = getattr(self.backend, "ainvoke", None)
        if backend_call:
            return await backend_call(request)
        return await ensure_async(self.backend)(request)
