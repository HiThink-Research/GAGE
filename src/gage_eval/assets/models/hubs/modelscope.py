"""ModelScope hub resolver."""

from __future__ import annotations

from typing import Dict

from gage_eval.assets.models.base import ModelHandle, ModelHub
from gage_eval.registry import registry

try:  # pragma: no cover - optional dependency
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
except Exception:  # pragma: no cover
    ms_snapshot_download = None


@registry.asset(
    "model_hubs",
    "modelscope",
    desc="ModelScope 模型仓库",
    tags=("remote", "modelscope"),
)
class ModelScopeModelHub(ModelHub):
    hub_type = "modelscope"

    def resolve(self) -> ModelHandle:
        if ms_snapshot_download is None:
            raise RuntimeError("modelscope is not installed; cannot download models from ModelScope hub")
        model_id = self.spec.params.get("model_id") or self.spec.params.get("remote_id")
        if not model_id:
            raise ValueError(f"Model '{self.spec.model_id}' requires 'model_id' when using ModelScope hub")
        cache_dir = self.hub_args.get("cache_dir")
        revision = self.spec.params.get("revision")
        local_path = ms_snapshot_download(model_id=model_id, cache_dir=cache_dir, revision=revision)
        metadata: Dict[str, str] = {"source": "modelscope", "model_id": model_id}
        if revision:
            metadata["revision"] = revision
        return ModelHandle(model_id=self.spec.model_id, local_path=local_path, metadata=metadata)
