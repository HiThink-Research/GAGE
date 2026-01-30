"""Helper model adapter implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.registry import registry
from gage_eval.role.adapters.model_role_adapter import ModelRoleAdapter


@registry.asset(
    "roles",
    "helper_model",
    desc="Helper role adapter for prompt augmentation / tool orchestration",
    tags=("role", "helper"),
    role_type="helper_model",
)
class HelperModelAdapter(ModelRoleAdapter):
    """Generic helper adapter (prompt rewrite / embedding / retrieval...)."""

    def __init__(
        self,
        adapter_id: str,
        backend: Any,
        capabilities=(),
        mode: str = "default",
        prompt_renderer=None,
        role_type: str = "helper_model",
        implementation: Optional[str] = None,
        implementation_params: Optional[Dict[str, Any]] = None,
        **params,
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type=role_type or "helper_model",
            capabilities=capabilities,
            backend=backend,
            prompt_renderer=prompt_renderer,
            **params,
        )
        self.mode = mode
        self._implementation = implementation
        self._implementation_params = dict(implementation_params or {})
        self._impl = None
        if implementation:
            try:
                impl_cls = registry.get("helper_impls", implementation)
            except KeyError:
                registry.auto_discover("helper_impls", "gage_eval.role.helper")
                impl_cls = registry.get("helper_impls", implementation)
            self._impl = impl_cls(**self._implementation_params)

    def prepare_backend_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        impl_payload = self._build_impl_payload(payload)
        if self._impl is not None:
            prepare = getattr(self._impl, "prepare_request", None)
            if callable(prepare):
                result = prepare(impl_payload, adapter=self)
                if result is not None:
                    return result
        return self.build_backend_request(impl_payload)

    def handle_backend_response(self, payload: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        impl_payload = self._build_impl_payload(payload)
        if self._impl is not None:
            handler = getattr(self._impl, "handle_response", None)
            if callable(handler):
                result = handler(impl_payload, response, adapter=self)
                if result is not None:
                    return result
        return response

    def build_backend_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build the backend request using the default prompt rendering path."""

        sample = payload.get("sample", {})
        step = payload.get("step", {})
        rendered = self.render_prompt(payload)
        sampling_params = self._compose_sampling_params(
            sample_params=sample.get("sampling_params"),
            runtime_params=payload.get("sampling_params"),
            legacy_params=sample.get("generation_params"),
        )
        return {
            "sample": sample,
            "step": step,
            "mode": step.get("mode") or self.mode,
            "prompt": rendered.prompt or sample.get("prompt") or sample.get("text"),
            "messages": rendered.messages,
            "inputs": sample.get("inputs"),
            "sampling_params": sampling_params,
            "prompt_meta": rendered.metadata or {},
        }

    def _build_impl_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        impl_payload = dict(payload or {})
        impl_payload["params"] = self._merge_params(payload)
        if self._implementation:
            impl_payload.setdefault("implementation", self._implementation)
        impl_payload.setdefault("adapter_id", self.adapter_id)
        return impl_payload

    def _merge_params(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(self._implementation_params)
        if not payload:
            return merged
        step = payload.get("step")
        step_params: Any = None
        if isinstance(step, dict):
            step_params = step.get("params") or step.get("args")
        elif step is not None:
            getter = getattr(step, "get", None)
            if callable(getter):
                try:
                    step_params = getter("params") or getter("args")
                except Exception:
                    step_params = None
            if step_params is None and hasattr(step, "params"):
                step_params = getattr(step, "params")
            if step_params is None and hasattr(step, "args"):
                step_params = getattr(step, "args")
        if isinstance(step_params, dict):
            merged.update(step_params)
        extra_params = payload.get("params")
        if isinstance(extra_params, dict):
            merged.update(extra_params)
        return merged
