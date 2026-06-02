"""Factory for GAGE-managed LiteLLM Router instances."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.role.model.backends.litellm.service_profile import _model_to_dict
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig, assert_litellm_capabilities


class LiteLLMRouterFactory:
    """Construct a LiteLLM Router when backend config requests internal routing."""

    def __init__(self, litellm_module: Any) -> None:
        self._litellm = litellm_module

    def build(self, cfg: LiteLLMBackendConfig) -> Any | None:
        route_mode = cfg.resolved_route_mode()
        if route_mode == "direct":
            return None
        if not cfg.model_list:
            raise ValueError("route_mode='router' requires non-empty model_list")

        assert_litellm_capabilities(require_router=True, litellm_module=self._litellm)
        return self._litellm.Router(
            model_list=self._model_list_payload(cfg),
            **self._router_settings_payload(cfg),
        )

    @staticmethod
    def _model_list_payload(cfg: LiteLLMBackendConfig) -> list[Dict[str, Any]]:
        return [_model_to_dict(deployment) for deployment in cfg.model_list]

    @staticmethod
    def _router_settings_payload(cfg: LiteLLMBackendConfig) -> Dict[str, Any]:
        settings = _model_to_dict(cfg.router_settings, exclude_none=True)
        return {key: value for key, value in settings.items() if value is not None}
