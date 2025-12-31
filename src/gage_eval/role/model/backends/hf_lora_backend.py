"""HuggingFace backend that loads LoRA/PEFT adapters and merges them."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict

from gage_eval.registry import registry
from gage_eval.role.model.backends.hf_backend import HFBackend


@registry.asset(
    "backends",
    "hf_lora",
    desc="HuggingFace Transformers + LoRA/PEFT 适配器合并后端",
    tags=("llm", "local", "peft"),
)
class HFLoRABackend(HFBackend):
    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            import torch
            import transformers
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("hf_lora backend requires transformers, torch and peft") from exc

        base_model = config.get("base_model")
        adapter_path = config.get("adapter_path")
        if not base_model or not adapter_path:
            raise ValueError("hf_lora backend requires 'base_model' and 'adapter_path'")

        cache_root = _resolve_merge_root(config.get("merge_cache_dir"), "lora_merged")
        cache_root.mkdir(parents=True, exist_ok=True)
        merged_dir = cache_root / _slugify(base_model, adapter_path)
        self._merged_dir = merged_dir
        skip_if_exists = config.get("skip_if_exists", True)
        dry_run_merge = bool(config.get("dry_run_merge", False))
        if dry_run_merge and not merged_dir.exists():
            merged_dir.mkdir(parents=True, exist_ok=True)
        elif not dry_run_merge and (config.get("force_merge") or not (skip_if_exists and merged_dir.exists())):
            merged_dir.mkdir(parents=True, exist_ok=True)
            merge_dtype = config.get("merge_dtype", "float16")
            torch_dtype = getattr(torch, merge_dtype, torch.float16)
            base = transformers.AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
            adapter = PeftModel.from_pretrained(base, adapter_path)
            merged = adapter.merge_and_unload()
            merged.save_pretrained(merged_dir)

        if dry_run_merge:
            return

        hf_config = dict(config)
        hf_config["model_name"] = str(merged_dir)
        hf_config.setdefault("tokenizer_name", config.get("tokenizer_name") or adapter_path)
        super().load_model(hf_config)


def _slugify(base: str, adapter: str) -> str:
    key = f"{base}:{adapter}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]


def _resolve_merge_root(merge_cache_dir: Any, subdir: str) -> Path:
    if merge_cache_dir:
        root = Path(merge_cache_dir).expanduser()
    else:
        data_cache = os.environ.get("GAGE_EVAL_DATA_CACHE")
        root = Path(data_cache).expanduser() if data_cache else Path(".cache/gage-eval")
    return root / subdir
