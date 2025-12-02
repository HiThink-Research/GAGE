"""Shared helpers for dataset loaders."""

from __future__ import annotations

import ast
import importlib
import inspect
import threading
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.preprocessors import resolve_preprocess
from gage_eval.registry import registry

@dataclass(frozen=True)
class PreprocessContext:
    """Container describing how to invoke a preprocess handle."""

    handle: Any
    kwargs: Dict[str, Any]


class TokenizerManager:
    """线程安全的有界 LRU 缓存，用于 tokenizer/processor 复用。"""

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


def resolve_callable(ref: Optional[Any]) -> Optional[Callable[[Dict[str, Any]], Any]]:
    """Resolve dotted-path references used by doc_to_* hooks."""

    if ref is None:
        return None
    if callable(ref):
        return ref
    if isinstance(ref, str):
        module_path, attr = _split_ref(ref)
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise TypeError(f"Unsupported callable reference: {ref!r}")


def resolve_doc_to_callable(spec: DatasetSpec, field: str) -> Optional[Callable[[Dict[str, Any]], Any]]:
    """Resolve doc_to_* callables and optionally curry keyword arguments."""

    ref = spec.params.get(field)
    if ref is None:
        return None
    func = resolve_callable(ref)
    kwargs = coerce_kwargs(spec.params.get(f"{field}_kwargs"))
    _validate_doc_to_signature(func, field, kwargs)
    if kwargs:
        frozen_kwargs = dict(kwargs)

        def _wrapped(record: Dict[str, Any], *, _func=func, _kwargs=frozen_kwargs):
            return _func(record, **_kwargs)

        return _wrapped
    return func


def build_preprocess_context(spec: DatasetSpec, *, data_path: Optional[str]) -> Optional[PreprocessContext]:
    """Create a preprocess context shared by JSONL/HF loaders."""

    module = spec.params.get("preprocess")
    if not module:
        return None
    kwargs = coerce_kwargs(spec.params.get("preprocess_kwargs"))
    if data_path is not None:
        kwargs.setdefault("data_path", data_path)
    if prompt_type := spec.params.get("prompt_type"):
        kwargs.setdefault("prompt_type", prompt_type)
    tokenizer = spec.params.get("tokenizer_object") or load_tokenizer(spec.params)
    if tokenizer is not None:
        kwargs.setdefault("tokenizer", tokenizer)
    adapter_handle = _resolve_registered_preprocessor(module, kwargs)
    if adapter_handle:
        return PreprocessContext(handle=adapter_handle, kwargs={})
    handle = resolve_preprocess(module)
    return PreprocessContext(handle=handle, kwargs=kwargs)


def inject_default_params(record: Dict[str, Any], default_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure default sampling params exist on the given sample dict."""

    if not default_params:
        return record
    if "sampling_params" not in record:
        record["sampling_params"] = default_params
    if "generation_params" not in record:
        record["generation_params"] = default_params  # backward compatibility
    return record


def apply_preprocess(records: Iterable[Dict[str, Any]], spec: DatasetSpec, *, data_path: Optional[str]) -> Iterable[Dict[str, Any]]:
    ctx = build_preprocess_context(spec, data_path=data_path)
    # 无显式 preprocess 时启用轻量默认渲染（文本场景）
    if not ctx:
        return _apply_default_chat_template(records, spec, data_path=data_path)

    def generator():
        for record in records:
            new_record = dict(record)
            new_record["inputs"] = ctx.handle.apply(new_record, **ctx.kwargs)
            tok_path = spec.params.get("tokenizer_path") or spec.params.get("tokenizer_name") or spec.params.get("model_path")
            if tok_path and "_tokenizer_path" not in new_record:
                new_record["_tokenizer_path"] = tok_path
            yield new_record

    return generator()


def apply_default_params(records: Iterable[Dict[str, Any]], spec: DatasetSpec) -> Iterable[Dict[str, Any]]:
    default_params = spec.params.get("default_params")
    if not default_params:
        return records

    def generator():
        for record in records:
            updated = inject_default_params(dict(record), default_params)
            yield updated

    return generator()


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
            # processor 缺少模板渲染能力，继续尝试其他 Loader
            last_exc = RuntimeError(f"Processor for '{name}' lacks apply_chat_template")
            continue
        if max_length := args.get("tokenizer_max_length"):
            setattr(tokenizer, "model_max_length", max_length)
        return tokenizer
    if last_exc:
        raise last_exc
    return None


def coerce_kwargs(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"Unable to parse preprocess_kwargs: {raw}") from exc
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"preprocess_kwargs string must evaluate to dict: {raw}")
    raise TypeError(f"Unsupported preprocess_kwargs type: {type(raw)!r}")


def _apply_default_chat_template(records: Iterable[Dict[str, Any]], spec: DatasetSpec, *, data_path: Optional[str]):
    """Fallback预处理：无显式 preprocess 时，对文本 messages 使用模型 tokenizer 渲染。"""

    tok_path = _resolve_tokenizer_name(spec.params)
    tokenizer = _get_or_load_tokenizer(spec.params)

    def generator():
        for record in records:
            new_record = dict(record)
            if _should_apply_default(new_record):
                prompt, source = _render_default_prompt(new_record.get("messages") or [], tokenizer)
                new_record["prompt"] = prompt
                new_record["inputs"] = prompt  # 兼容旧流程，后续会在 DataManager 中归一化
                new_record["chat_template_mode"] = "preprocess"
                new_record["template_source"] = source
                new_record["rendered_by"] = "preprocess"
                new_record["cache_suffix"] = "-chat_template" if source == "model" else "-plain"
            if tok_path and "_tokenizer_path" not in new_record:
                new_record["_tokenizer_path"] = tok_path
            yield new_record

    return generator()


def _get_or_load_tokenizer(params: Dict[str, Any]):
    name = _resolve_tokenizer_name(params)
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
        # 兜底：记录日志，返回 None
        try:
            import logging
            logging.getLogger(__name__).warning("default preprocess failed to load tokenizer '%s': %s", name, exc)
        except Exception:
            pass
        return None


def _resolve_tokenizer_name(params: Dict[str, Any]) -> Optional[str]:
    """Resolve tokenizer/model path with env fallback for default preprocess."""

    name = params.get("tokenizer_name") or params.get("tokenizer_path") or params.get("model_path")
    if name:
        return name
    # 环境变量兜底，便于不显式写 tokenizer_path 时仍尝试模型路径
    return os.environ.get("GAGE_EVAL_MODEL_PATH") or os.environ.get("MODEL_PATH")


def _should_apply_default(record: Dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    return not _contains_multimodal(messages)


def _contains_multimodal(messages: list) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for fragment in content:
                if isinstance(fragment, dict) and fragment.get("type") in {"image", "image_url", "audio_url"}:
                    return True
    return False


def _render_default_prompt(messages: list, tokenizer) -> tuple[str, str]:
    def _normalize_messages(msgs: list) -> list:
        normalized = []
        for m in msgs:
            item = dict(m)
            content = item.get("content")
            if isinstance(content, list):
                text_parts = []
                for frag in content:
                    if isinstance(frag, dict) and frag.get("type") == "text":
                        text_parts.append(str(frag.get("text", "")))
                item["content"] = " ".join(text_parts)
            elif content is None:
                item["content"] = ""
            normalized.append(item)
        return normalized

    def _simple_render(msgs: list) -> str:
        segs = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content")
            if isinstance(content, list):
                text_parts = []
                for frag in content:
                    if isinstance(frag, dict) and frag.get("type") == "text":
                        text_parts.append(str(frag.get("text", "")))
                text = " ".join(text_parts)
            else:
                text = str(content) if content is not None else ""
            segs.append(f"{role}: {text}".strip())
        segs.append("assistant:")
        return "\n".join(segs)

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        norm_messages = _normalize_messages(messages)
        try:
            rendered = tokenizer.apply_chat_template(
                norm_messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(rendered, list):
                rendered = rendered[0]
            if rendered:
                return str(rendered), "model"
        except Exception:
            pass
    return _simple_render(messages), "fallback"


def _split_ref(ref: str) -> tuple[str, str]:
    if ":" in ref:
        module_path, attr = ref.rsplit(":", 1)
    elif "." in ref:
        module_path, attr = ref.rsplit(".", 1)
    else:
        raise ValueError(f"Callable reference must include module path: {ref}")
    return module_path, attr


def _validate_doc_to_signature(func: Callable[..., Any], field: str, provided_kwargs: Dict[str, Any]) -> None:
    """Ensure doc_to_* callables expose the expected signature."""

    signature = inspect.signature(func)
    params = list(signature.parameters.values())
    if not params:
        raise TypeError(f"doc_to field '{field}' must accept at least one argument (the sample record)")

    required_kwargs: list[str] = []
    for param in params[1:]:
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            required_kwargs.clear()
            break
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if param.default is inspect._empty:
                required_kwargs.append(param.name)

    missing = [name for name in required_kwargs if name not in provided_kwargs]
    if missing:
        raise TypeError(
            f"doc_to field '{field}' requires keyword arguments {missing}; provide them via {field}_kwargs"
        )


class _PreprocessorAdapter:
    """Adapts DatasetPreprocessor implementations to the legacy handle API."""

    def __init__(self, preprocessor) -> None:
        self._preprocessor = preprocessor

    def apply(self, sample: Dict[str, Any], **kwargs):
        return self._preprocessor.transform(sample, **kwargs)


def _resolve_registered_preprocessor(name: str, kwargs: Dict[str, Any]) -> Optional[_PreprocessorAdapter]:
    if not name:
        return None
    try:
        preprocessor_cls = registry.get("dataset_preprocessors", name)
    except KeyError:
        return None
    preprocessor = preprocessor_cls(**kwargs)
    return _PreprocessorAdapter(preprocessor)
