"""DefaultPreprocessor: fallback implementation based on SimplePreprocessor."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.assets.datasets.preprocessors.simple_preprocessor import SimplePreprocessor
from gage_eval.assets.datasets.utils.rendering import (
    contains_multimodal,
    render_messages_with_fallback,
    set_render_flags,
)


class DefaultPreprocessor(SimplePreprocessor):
    """默认预处理器：支持外部脚本或 llm-eval 样式转换并提供文本兜底渲染。"""

    module_path: Optional[str] = None

    def __init__(self, tokenizer=None, tokenizer_path: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._handle = None
        self._tokenizer = tokenizer
        self._tokenizer_path = tokenizer_path

    def to_legacy(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        merged = dict(self.kwargs)
        merged.update(kwargs)
        if self._handle:
            result = self._handle.apply(record, **merged)
            if result is not None and "inputs" not in record:
                if isinstance(result, dict):
                    record["inputs"] = result
                elif isinstance(result, str):
                    record["inputs"] = {"prompt": result}
            return record
        return dict(record)

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = super().to_sample(record, **kwargs)
        if self._should_apply_default(sample):
            prompt, source = render_messages_with_fallback(sample.get("messages") or [], self._tokenizer)
            sample["prompt"] = prompt
            sample["inputs"] = {"prompt": prompt}
            set_render_flags(
                sample,
                mode="preprocess",
                source=source,
                rendered_by="preprocess",
                cache_suffix="-chat_template" if source == "model" else "-plain",
            )
            if self._tokenizer_path and "_tokenizer_path" not in sample:
                sample["_tokenizer_path"] = self._tokenizer_path
        return sample

    def _should_apply_default(self, sample: Dict[str, Any]) -> bool:
        messages = sample.get("messages")
        if not isinstance(messages, list) or not messages:
            return False
        return (not contains_multimodal(messages)) and self._tokenizer is not None


__all__ = ["DefaultPreprocessor"]
