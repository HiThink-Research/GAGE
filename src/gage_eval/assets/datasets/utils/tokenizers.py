"""Tokenizer utilities shared across loaders/preprocessors."""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional


class TokenizerManager:
    """Thread-safe bounded LRU cache for reusing tokenizers/processors."""

    def __init__(self, max_size: int = 32):
        env_max = os.environ.get("GAGE_EVAL_TOKENIZER_CACHE_MAX")
        if env_max:
            try:
                max_size = int(env_max)
            except ValueError:
                pass
        self._max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, name: str, kwargs: Dict[str, Any], loader_fn: Callable[[], Any]):
        try:
            key = (name, json.dumps(kwargs, sort_keys=True))
        except Exception:
            key = (name, repr(sorted(kwargs.items())))
        with self._lock:
            if key in self._cache:
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
        obj = loader_fn()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            self._cache[key] = obj
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            return obj


_TOKENIZER_MANAGER = TokenizerManager()


def resolve_tokenizer_name(params: Dict[str, Any]) -> Optional[str]:
    """Resolve tokenizer/model path with env fallback."""

    name = params.get("tokenizer_name") or params.get("tokenizer_path") or params.get("model_path")
    if name:
        return name
    return os.environ.get("GAGE_EVAL_MODEL_PATH") or os.environ.get("MODEL_PATH")


def load_tokenizer(args: Dict[str, Any]):
    name = args.get("tokenizer_name") or args.get("tokenizer_path")
    if not name:
        return None
    backend = args.get("tokenizer_backend")
    kwargs = dict(args.get("tokenizer_kwargs", {}))
    if "trust_remote_code" not in kwargs:
        kwargs["trust_remote_code"] = args.get("tokenizer_trust_remote_code", True)
    backends = [backend] if backend else ["auto_tokenizer", "auto_processor"]
    last_exc = None
    for backend_choice in backends:
        try:  # pragma: no cover - heavy dependency
            if backend_choice == "auto_tokenizer":
                from transformers import AutoTokenizer as Loader  # type: ignore
            else:
                from transformers import AutoProcessor as Loader  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers must be installed to use tokenizer-backed preprocessors"
            ) from exc
        try:
            tokenizer = Loader.from_pretrained(name, **kwargs)
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_exc = exc
            continue
        if backend_choice == "auto_processor" and not hasattr(tokenizer, "apply_chat_template"):
            last_exc = RuntimeError(f"Processor for '{name}' lacks apply_chat_template")
            continue
        if max_length := args.get("tokenizer_max_length"):
            setattr(tokenizer, "model_max_length", max_length)
        return tokenizer
    if last_exc:
        raise last_exc
    return None


def get_or_load_tokenizer(params: Dict[str, Any]):
    name = resolve_tokenizer_name(params)
    if not name:
        return None
    kwargs = dict(params.get("tokenizer_kwargs") or {})
    if "tokenizer_backend" not in kwargs and params.get("tokenizer_backend") is not None:
        kwargs["tokenizer_backend"] = params.get("tokenizer_backend")
    kwargs.setdefault("tokenizer_backend", "auto_tokenizer")
    kwargs.setdefault("tokenizer_trust_remote_code", params.get("tokenizer_trust_remote_code", True))
    try:
        return _TOKENIZER_MANAGER.get(
            name,
            kwargs,
            lambda: load_tokenizer({"tokenizer_name": name, **kwargs}),
        )
    except Exception as exc:
        try:
            import logging
            logging.getLogger(__name__).warning("default preprocess failed to load tokenizer '%s': %s", name, exc)
        except Exception:
            pass
        return None


__all__ = ["TokenizerManager", "load_tokenizer", "get_or_load_tokenizer", "resolve_tokenizer_name"]
