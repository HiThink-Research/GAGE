"""Shared helpers for dataset loaders."""

from __future__ import annotations

import ast
import importlib
import json
import inspect
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
    if not ctx:
        return records

    def generator():
        for record in records:
            new_record = dict(record)
            new_record["inputs"] = ctx.handle.apply(new_record, **ctx.kwargs)
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
    backend = args.get("tokenizer_backend", "auto_processor")
    kwargs = dict(args.get("tokenizer_kwargs", {}))
    if "trust_remote_code" not in kwargs:
        kwargs["trust_remote_code"] = args.get("tokenizer_trust_remote_code", True)
    try:  # pragma: no cover - heavy dependency
        if backend == "auto_tokenizer":
            from transformers import AutoTokenizer as Loader  # type: ignore
        else:
            from transformers import AutoProcessor as Loader  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers must be installed to use tokenizer-backed preprocessors"
        ) from exc
    tokenizer = Loader.from_pretrained(name, **kwargs)
    if max_length := args.get("tokenizer_max_length"):
        setattr(tokenizer, "model_max_length", max_length)
    return tokenizer


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
