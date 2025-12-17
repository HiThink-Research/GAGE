"""Class-based MMMU multimodal preprocessor."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.utils.multimodal import merge_multimodal_inputs
from gage_eval.assets.datasets.utils.rendering import set_render_flags


class MMMUMultimodalPreprocessor(BasePreprocessor):
    """Ensure MMMU samples have multi_modal_data/image filled from messages."""

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = dict(record)
        merge_multimodal_inputs(sample)
        set_render_flags(
            sample,
            mode="preprocess",
            source="manual",
            rendered_by="preprocess",
            cache_suffix="-converted",
            overwrite=False,
        )
        sample.setdefault("inputs", {})
        return sample


__all__ = ["MMMUMultimodalPreprocessor"]
