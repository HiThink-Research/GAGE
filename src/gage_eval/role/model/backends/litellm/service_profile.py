"""Describe LiteLLM/vLLM routing profile metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from gage_eval.role.model.config.litellm import LiteLLMBackendConfig


def _model_to_dict(value: Any, *, exclude_none: bool = False) -> Dict[str, Any]:
    if isinstance(value, dict):
        data = dict(value)
        if exclude_none:
            return {key: item for key, item in data.items() if item is not None}
        return data
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=exclude_none)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=exclude_none)
    return dict(value)


@dataclass(frozen=True)
class VLLMServiceProfile:
    """Runtime profile for direct, gateway, and internal Router LiteLLM calls."""

    mode: Literal["direct", "router"]
    model: str
    api_base: str | None = None
    deployments: List[Dict[str, Any]] = field(default_factory=list)
    topology: Dict[str, Any] = field(default_factory=dict)
    reasoning_parser: str | None = None
    chat_template_defaults: Dict[str, Any] | None = None
    language_model_only: bool | None = None

    @classmethod
    def from_config(cls, cfg: LiteLLMBackendConfig) -> "VLLMServiceProfile":
        generation_parameters = cfg.generation_parameters.to_dict()
        chat_template_defaults = generation_parameters.get("chat_template_kwargs")
        return cls(
            mode=cfg.resolved_route_mode(),
            model=cfg.model,
            api_base=cfg.api_base,
            deployments=[_model_to_dict(deployment) for deployment in cfg.model_list],
            topology=_model_to_dict(cfg.vllm.topology, exclude_none=True),
            reasoning_parser=cfg.vllm.reasoning_parser,
            chat_template_defaults=chat_template_defaults,
            language_model_only=cfg.vllm.topology.language_model_only,
        )

    def validate_deployment_consistency(self) -> List[str]:
        """Validate known per-deployment vLLM metadata without inventing it."""

        if self.mode != "router" or len(self.deployments) <= 1:
            return []

        parsers = self._deployment_values("reasoning_parser")
        chat_template_defaults = self._deployment_values("chat_template_defaults")
        language_model_only = self._deployment_values("language_model_only")

        self._fail_on_inconsistent_values("reasoning_parser", parsers)
        self._fail_on_inconsistent_values("chat_template_defaults", chat_template_defaults)
        self._fail_on_inconsistent_values("language_model_only", language_model_only)

        top_level_profile_declared = (
            self.reasoning_parser is not None
            or self.chat_template_defaults is not None
            or self.language_model_only is not None
        )
        per_deployment_profile_declared = bool(parsers or chat_template_defaults or language_model_only)
        if not per_deployment_profile_declared and top_level_profile_declared:
            return [
                "per-deployment vLLM metadata not provided; assuming declared profile applies to all deployments"
            ]
        return []

    def _deployment_values(self, field_name: str) -> List[Any]:
        values: List[Any] = []
        for deployment in self.deployments:
            value = self._deployment_metadata_value(deployment, field_name)
            if value is not None:
                values.append(value)
        return values

    @staticmethod
    def _deployment_metadata_value(deployment: Dict[str, Any], field_name: str) -> Any:
        vllm_metadata = deployment.get("vllm")
        if isinstance(vllm_metadata, dict) and vllm_metadata.get(field_name) is not None:
            return vllm_metadata[field_name]
        if field_name == "language_model_only" and isinstance(vllm_metadata, dict):
            topology = vllm_metadata.get("topology")
            if isinstance(topology, dict) and topology.get("language_model_only") is not None:
                return topology["language_model_only"]
        if deployment.get(field_name) is not None:
            return deployment[field_name]
        return None

    @staticmethod
    def _fail_on_inconsistent_values(field_name: str, values: List[Any]) -> None:
        if not values:
            return
        first = values[0]
        if any(value != first for value in values[1:]):
            raise ValueError(f"model_list deployments have inconsistent vLLM {field_name}")
