"""Local filesystem model hub."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from gage_eval.assets.models.base import ModelHandle, ModelHub
from gage_eval.registry import registry


@registry.asset(
    "model_hubs",
    "local",
    desc="本地文件系统模型",
    tags=("local",),
)
class LocalModelHub(ModelHub):
    hub_type = "local"

    def resolve(self) -> ModelHandle:
        path = self.spec.params.get("path") or self.spec.params.get("local_path")
        if not path:
            raise ValueError(f"Model '{self.spec.model_id}' requires 'path' when using local hub")
        resolved = str(Path(path).expanduser().resolve())
        metadata: Dict[str, str] = {"source": "local"}
        return ModelHandle(model_id=self.spec.model_id, local_path=resolved, metadata=metadata)
