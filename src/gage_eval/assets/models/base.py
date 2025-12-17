"""Base classes for model assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gage_eval.config.pipeline_config import ModelSpec


@dataclass
class ModelHandle:
    model_id: str
    local_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelHub:
    hub_type = "default"

    def __init__(self, spec: ModelSpec, *, hub_args: Optional[Dict[str, Any]] = None) -> None:
        self.spec = spec
        self.hub_args = hub_args or {}

    def resolve(self) -> ModelHandle:  # pragma: no cover - abstract
        raise NotImplementedError
