"""Sample-scoped router for resolving multiple sandbox providers."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.runtime.invocation import SandboxBinding
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope


class SandboxSessionRouter:
    """Manage sandbox providers for a single sample execution."""

    def __init__(
        self,
        manager: SandboxManager,
        *,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        trace: Optional[ObservabilityTrace] = None,
    ) -> None:
        self._manager = manager
        self._run_id = run_id
        self._task_id = task_id
        self._sample_id = sample_id
        self._trace = trace
        self._providers: Dict[str, SandboxProvider] = {}

    def resolve_config(
        self,
        role_config: Dict[str, object],
        sample_config: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Merge role defaults with sample-scoped overrides."""

        return self._manager.resolve_config(
            dict(role_config or {}),
            dict(sample_config or {}) if sample_config else None,
        )

    def get_provider(self, binding: SandboxBinding) -> Optional[SandboxProvider]:
        """Return a route-scoped provider, creating it lazily when needed."""

        if not binding.enabled or not binding.config:
            return None
        route_key = binding.route_key or _build_route_key(binding)
        provider = self._providers.get(route_key)
        if provider is None:
            provider = SandboxProvider(
                self._manager,
                binding.config,
                SandboxScope(
                    run_id=self._run_id,
                    task_id=self._task_id,
                    sample_id=self._sample_id,
                    arena_id=_resolve_arena_id(binding, self._sample_id),
                ),
                trace=self._trace,
            )
            self._providers[route_key] = provider
        if self._trace is not None:
            payload = {
                "adapter_id": binding.adapter_id,
                "step_type": binding.step_type,
                "source": binding.source,
                "route_key": route_key,
            }
            sandbox_id = (
                binding.config.get("sandbox_id")
                or binding.config.get("template_name")
                or binding.config.get("runtime")
            )
            if sandbox_id:
                payload["sandbox_id"] = str(sandbox_id)
            self._trace.emit(
                "sandbox_route_selected",
                payload,
                sample_id=self._sample_id,
            )
        return provider

    def release_all(self) -> None:
        """Release every provider created for the sample."""

        failures: list[str] = []
        for route_key, provider in list(self._providers.items()):
            try:
                provider.release()
            except Exception as exc:
                failures.append(f"{route_key}: {type(exc).__name__}: {exc}")
            finally:
                self._providers.pop(route_key, None)
        if failures:
            logger.warning(
                "SandboxSessionRouter release completed with {} issue(s): {}",
                len(failures),
                "; ".join(failures),
            )


def _resolve_arena_id(binding: SandboxBinding, sample_id: Optional[str]) -> Optional[str]:
    if binding.step_type != "arena" or not binding.adapter_id or not sample_id:
        return None
    return f"{binding.adapter_id}_{sample_id}"


def _build_route_key(binding: SandboxBinding) -> str:
    payload = json.dumps(binding.config, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    lifecycle = str(binding.config.get("lifecycle") or "per_sample")
    if lifecycle == "per_arena":
        return f"{digest}:{binding.step_type or 'unknown'}:{binding.adapter_id or 'unknown'}"
    return digest
