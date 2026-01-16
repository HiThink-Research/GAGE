"""Mathvista Resource Providers"""

from __future__ import annotations

import random
from typing import Any, Dict
import collections

from gage_eval.assets.datasets.bundles.base import BaseBundle
from gage_eval.assets.datasets.utils.loaders import load_or_cache_dataset
from gage_eval.assets.datasets.bundles.mmlu_pro.utils import format_cot_example, pretty_sample
import os

class MMLUProBundle(BaseBundle):
    """ Provide MMLU-Pro related resources """

    def __init__(self,  **kwargs: Any):
        super().__init__(**kwargs)
        self.loader_params = kwargs.get('hub_params')
        self.hub = kwargs.get('hub')
        self.n_few_shot = kwargs.get("n_few_shot")
        self.dataset = None 
        self.cache_path = None
        #self.few_shots = collections.defaultdict(list)
        self.few_shots = self.load()

    def _normalize_source(self, raw: str) -> str:
        normalized = raw.lower()
        if normalized not in {"huggingface", "modelscope", "local"}:
            raise ValueError(f"Unsupported dataset source: {raw}")
        return normalized

    def load(self) -> None:
        try:
            import datasets  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "`datasets` package is required to load HuggingFace/ModelScope datasets"
            ) from exc

        source = self._normalize_source(self.loader_params.get("source", "huggingface"))
        hub_id = self.loader_params.get("hub_id")
        split = self.loader_params.get("split", "train")
        subset = self.loader_params.get("subset")
        revision = self.loader_params.get("revision")
        trust_remote = self.loader_params.get("trust_remote_code", True)
        load_kwargs = dict(self.loader_params.get("load_kwargs", {}))
        builder_name = self.loader_params.get("builder_name")
        data_files = self.loader_params.get("data_files")
        streaming = False
        dataset, self.cache_path = load_or_cache_dataset(
            datasets_module=datasets,
            hub_id=hub_id,
            source=source,
            split=split,
            subset=subset,
            revision=revision,
            trust_remote=trust_remote,
            load_kwargs=load_kwargs,
            cache_dir=self.loader_params.get("cache_dir"),
            builder_name=builder_name,
            data_files=data_files,
        )
        self.dataset = dataset.map(pretty_sample)
        return self.build_few_shot(self.dataset)

    def normalize_subject(self, sub):
        return sub.replace(' ', '_')
    
    def build_few_shot(self, dataset):
        few_shots = collections.defaultdict(list)
        for sample in dataset:
            category = sample['category']
            key = self.normalize_subject(category)
            single = format_cot_example(sample)
            if len(few_shots[key]) < self.n_few_shot:
                few_shots[key].append(single)
        return few_shots       


    def provide(self, sample: Dict[str, Any], **kwargs: Any) -> Any:  # pragma: no cover - abstract
        category = sample['category']
        key = self.normalize_subject(category)
        sample["few_shot_examples"] = self.few_shots[key]
        return sample        

if __name__ == '__main__':
    t = MMLUProBundle(**{'n_few_shot': 5, 'hub_params': {'source': 'huggingface', 'hub_id': 'TIGER-Lab/MMLU-Pro', 'split': 'validation'}, 'data_path': 'TIGER-Lab/MMLU-Pro'})
    t.load()
    a = t.provide({"category": "math"})
    print("a:", a)