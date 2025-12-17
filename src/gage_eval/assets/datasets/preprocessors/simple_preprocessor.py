"""SimplePreprocessor: legacy adapter -> Sample via convert_llmeval_record."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.utils.legacy import convert_llmeval_record


class SimplePreprocessor(BasePreprocessor):
    """低代码模式：to_legacy -> llm-eval 兼容转换。"""

    name = "simple"

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        merged = dict(self.kwargs)
        merged.update(kwargs)
        dataset_id = merged.pop("dataset_id", None) or record.get("_dataset_id") or "unknown"
        dataset_meta = merged.pop("dataset_metadata", None) or record.get("_dataset_metadata")
        legacy = self.to_legacy(dict(record), **merged)
        return convert_llmeval_record(legacy, dataset_id=dataset_id, dataset_metadata=dataset_meta, **merged)

    def to_legacy(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = ["SimplePreprocessor"]
