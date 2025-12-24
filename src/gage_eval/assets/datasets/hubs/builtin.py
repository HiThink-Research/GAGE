"""Builtin DatasetHub implementations."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.hubs.base import DatasetHub, DatasetHubHandle
from gage_eval.registry import registry


@registry.asset(
    "dataset_hubs",
    "inline",
    desc="Inline dataset hub (local/relative paths)",
    tags=("local",),
)
class InlineDatasetHub(DatasetHub):
    """Hub that simply exposes inline file paths."""

    def resolve(self) -> DatasetHubHandle:
        path = self.hub_args.get("path") or self.spec.params.get("path")
        hub_id = path or self.spec.dataset_id
        metadata = {"path": path} if path else {}
        return DatasetHubHandle(hub_id=str(hub_id), metadata=metadata, streaming=self.hub_args.get("streaming", False))


@registry.asset(
    "dataset_hubs",
    "huggingface",
    desc="Remote dataset hub (HuggingFace Hub / ModelScope)",
    tags=("remote", "hf"),
)
class HuggingFaceDatasetHub(DatasetHub):
    """Hub that standardizes HuggingFace/ModelScope identifiers."""

    def resolve(self) -> DatasetHubHandle:
        metadata: Dict[str, Any] = dict(self.hub_args)
        spec_params: Dict[str, Any] = self.spec.params

        hub_id = metadata.get("hub_id") or spec_params.get("hub_id") or spec_params.get("path") or self.spec.dataset_id
        metadata["hub_id"] = hub_id

        raw_source = metadata.get("source") or spec_params.get("source") or "huggingface"
        source = raw_source.lower()
        metadata["source"] = source

        subset = metadata.get("subset")
        if subset is None:
            subset = spec_params.get("subset")
        metadata["subset"] = subset

        split = metadata.get("split")
        if split is None:
            split = spec_params.get("split")
        metadata["split"] = split

        revision = metadata.get("revision")
        if revision is None:
            revision = spec_params.get("revision")
        if revision is not None:
            metadata["revision"] = revision

        trust_remote = metadata.get("trust_remote_code")
        if trust_remote is None:
            trust_remote = spec_params.get("trust_remote_code")
        if trust_remote is not None:
            metadata["trust_remote_code"] = trust_remote

        streaming_flag = metadata.get("streaming") if "streaming" in metadata else spec_params.get("streaming")
        streaming = bool(streaming_flag)

        return DatasetHubHandle(hub_id=str(hub_id), metadata=metadata, streaming=streaming)


@registry.asset(
    "dataset_hubs",
    "modelscope",
    desc="Remote dataset hub (ModelScope)",
    tags=("remote", "modelscope"),
)
class ModelScopeDatasetHub(HuggingFaceDatasetHub):
    """Alias hub that defaults to ModelScope as the source."""

    def resolve(self) -> DatasetHubHandle:
        if "source" not in self.hub_args:
            self.hub_args["source"] = "modelscope"
        return super().resolve()


registry.register(
    "dataset_hubs",
    "ms_hub",
    ModelScopeDatasetHub,
    desc="Remote dataset hub (ModelScope; ms_hub alias)",
    tags=("remote", "modelscope"),
)
