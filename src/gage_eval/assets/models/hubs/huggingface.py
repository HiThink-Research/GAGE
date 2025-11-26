"""HuggingFace snapshot download hub."""

from __future__ import annotations

from typing import Dict

from huggingface_hub import snapshot_download

from gage_eval.assets.models.base import ModelHandle, ModelHub
from gage_eval.registry import registry


@registry.asset(
    "model_hubs",
    "huggingface",
    desc="HuggingFace 模型仓库",
    tags=("remote", "hf"),
)
class HuggingFaceModelHub(ModelHub):
    hub_type = "huggingface"

    def resolve(self) -> ModelHandle:
        repo_id = self.spec.params.get("repo_id") or self.spec.params.get("remote_id")
        if not repo_id:
            raise ValueError(f"Model '{self.spec.model_id}' requires 'repo_id' when using HuggingFace hub")
        revision = self.spec.params.get("revision")
        allow_patterns = self.spec.params.get("allow_patterns")
        ignore_patterns = self.spec.params.get("ignore_patterns")
        cache_dir = self.hub_args.get("cache_dir")
        local_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
        )
        metadata: Dict[str, str] = {"source": "huggingface", "repo_id": repo_id}
        if revision:
            metadata["revision"] = revision
        return ModelHandle(model_id=self.spec.model_id, local_path=local_path, metadata=metadata)
