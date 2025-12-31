"""Common template for DUT/Judge/Helper role adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderResult, PromptRenderer
from gage_eval.role.model.backends import build_backend, wrap_backend
from gage_eval.registry import ensure_async, run_sync
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState


class ModelRoleAdapter(RoleAdapter):
    """Abstract adapter that bridges RoleAdapter and backend implementations."""

    def __init__(
        self,
        adapter_id: str,
        role_type: str,
        capabilities,
        backend: Any,
        prompt_renderer: Optional[PromptRenderer] = None,
        **params,
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type=role_type,
            capabilities=capabilities,
            resource_requirement=params.pop("resource_requirement", None),
            sandbox_config=params.pop("sandbox_config", None),
        )
        self.backend = self._resolve_backend(backend)
        self.prompt_renderer = prompt_renderer
        self.params = params

    def _resolve_backend(self, backend: Any):
        if backend is None:
            raise ValueError(f"RoleAdapter '{self.adapter_id}' requires a backend configuration or instance")
        if hasattr(backend, "__call__"):
            return backend
        if hasattr(backend, "generate") or hasattr(backend, "generate_batch"):
            return backend
        if isinstance(backend, dict):
            return wrap_backend(build_backend(backend))
        raise TypeError(f"Unsupported backend specification for adapter '{self.adapter_id}': {backend!r}")

    # ------------------------------------------------------------------
    # Template hooks
    # ------------------------------------------------------------------
    def prepare_backend_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def handle_backend_response(self, payload: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        return response
    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _compose_sampling_params(
        self,
        *,
        sample_params: Optional[Dict[str, Any]],
        runtime_params: Optional[Dict[str, Any]],
        legacy_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        dataset_defaults = sample_params or legacy_params or {}
        backend_defaults = getattr(self.backend, "default_params", {}) or {}
        role_defaults = self._role_default_params()

        merged.update(dataset_defaults)
        merged.update(backend_defaults)
        merged.update(role_defaults)
        if runtime_params:
            merged.update(runtime_params)
        return {k: v for k, v in merged.items() if v is not None}

    def _role_default_params(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        candidate = self.params.get("default_params")
        if isinstance(candidate, dict):
            defaults.update(candidate)
        sampling_keys = (
            "max_new_tokens",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "stop",
        )
        for key in sampling_keys:
            if key in self.params and key not in defaults:
                defaults[key] = self.params[key]
        return defaults

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def render_prompt(self, payload: Dict[str, Any]) -> PromptRenderResult:
        if not self.prompt_renderer:
            return PromptRenderResult()
        context = PromptContext(
            sample=payload.get("sample", {}),
            payload=payload,
            history=payload.get("history") or [],
            extras={"adapter_id": self.adapter_id, "role_type": self.role_type},
        )
        return self.prompt_renderer.render(context)

    # ------------------------------------------------------------------
    # RoleAdapter API
    # ------------------------------------------------------------------
    def prepare_request(self, payload: Dict[str, Any], state: RoleAdapterState) -> Any:
        """Producer侧预处理：渲染 prompt + 采样参数，缓存原始 payload 供后续解析。"""

        state.metadata["payload"] = payload
        return self.prepare_backend_request(payload)

    def execute_batch(self, requests: List[Any]) -> List[Any]:
        """默认串行执行 backend，可检测批接口 generate_batch 以减少调用次数。"""

        backend_call = getattr(self.backend, "invoke", None)
        async_backend_call = getattr(self.backend, "ainvoke", None)
        batch_call = getattr(self.backend, "generate_batch", None)
        payloads: List[Dict[str, Any]] = []
        states: List[RoleAdapterState] = []
        for item in requests:
            payload = item.get("_payload") if isinstance(item, dict) else item
            state = item.get("_state") if isinstance(item, dict) else RoleAdapterState()
            payloads.append(payload)
            states.append(state)

        results: List[Any] = []
        if batch_call:
            batch_results = batch_call(payloads)
            if not isinstance(batch_results, list) or len(batch_results) != len(payloads):
                raise ValueError(
                    f"generate_batch returned unexpected result size (got={len(batch_results) if isinstance(batch_results, list) else 'non-list'}, expected={len(payloads)})"
                )
            results = batch_results
        else:
            for payload in payloads:
                if async_backend_call:
                    results.append(run_sync(async_backend_call(payload)))
                elif backend_call:
                    results.append(backend_call(payload))
                else:
                    results.append(run_sync(ensure_async(self.backend)(payload)))

        parsed: List[Any] = []
        for raw, state, original in zip(results, states, [s.metadata.get("payload") for s in states]):
            parsed.append(self.handle_backend_response(original or {}, raw))
        return parsed

    def parse_response(self, response: Any, state: RoleAdapterState) -> Dict[str, Any]:
        """调度器回写：结合 prepare 阶段缓存的原始 payload 做解析。"""

        original = state.metadata.get("payload") if isinstance(state, RoleAdapterState) else {}
        if isinstance(response, dict):
            return response
        return self.handle_backend_response(original or {}, response)

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        request = self.prepare_backend_request(payload)
        backend_call = getattr(self.backend, "ainvoke", None)
        if backend_call:
            response = await backend_call(request)
        else:
            response = await ensure_async(self.backend)(request)
        return self.handle_backend_response(payload, response)
