"""HuggingFace backend that applies delta checkpoints to base models."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict

from gage_eval.registry import registry
from gage_eval.role.model.backends.hf_backend import HFBackend


@registry.asset(
    "backends",
    "hf_delta",
    desc="HuggingFace Transformers delta 权重量化后端",
    tags=("llm", "local", "delta"),
)
class HFDeltaBackend(HFBackend):
    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("hf_delta backend requires transformers + torch") from exc

        base_model = config.get("base_model")
        delta_model = config.get("delta_model")
        if not base_model or not delta_model:
            raise ValueError("hf_delta backend requires 'base_model' and 'delta_model'")

        cache_root = _resolve_merge_root(config.get("merge_cache_dir"), "delta_merged")
        cache_root.mkdir(parents=True, exist_ok=True)
        merged_dir = cache_root / _slugify(base_model, delta_model)
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
            delta = transformers.AutoModelForCausalLM.from_pretrained(delta_model, torch_dtype=torch_dtype)
            base_state = base.state_dict()
            delta_state = delta.state_dict()
            for name, param in base_state.items():
                if name in delta_state:
                    param.data += delta_state[name]
            base.save_pretrained(merged_dir)

        if dry_run_merge:
            return

        hf_config = dict(config)
        hf_config["model_name"] = str(merged_dir)
        super().load_model(hf_config)


def _slugify(base: str, delta: str) -> str:
    key = f"{base}:{delta}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]


def _resolve_merge_root(merge_cache_dir: Any, subdir: str) -> Path:
    if merge_cache_dir:
        root = Path(merge_cache_dir).expanduser()
    else:
        data_cache = os.environ.get("GAGE_EVAL_DATA_CACHE")
        root = Path(data_cache).expanduser() if data_cache else Path(".cache/gage-eval")
    return root / subdir
