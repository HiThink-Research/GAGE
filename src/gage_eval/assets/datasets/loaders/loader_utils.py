"""Shared helpers for dataset loaders."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, TYPE_CHECKING

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.utils.reflection import resolve_callable as _resolve_callable, coerce_kwargs as _coerce_kwargs
from gage_eval.assets.datasets.utils.tokenizers import (
    _TOKENIZER_MANAGER,
    load_tokenizer,
    get_or_load_tokenizer,
    resolve_tokenizer_name,
)
from gage_eval.registry import registry
from gage_eval.observability.config import get_observability_config

if TYPE_CHECKING:  # pragma: no cover
    from gage_eval.observability.config import ObservabilityConfig
    from gage_eval.observability.trace import ObservabilityTrace
@dataclass(frozen=True)
class PreprocessContext:
    """Container describing how to invoke a preprocess handle."""

    handle: Any
    kwargs: Dict[str, Any]

@dataclass(frozen=True)
class BundleContext:
    """Container describing how to invoke a bundle handle."""

    handle: Any
    kwargs: Dict[str, Any]

class _BundleAdapter:
    """Adapts DatasetPreprocessor implementations to the legacy handle API."""

    def __init__(self, bundle) -> None:
        self._bundle = bundle
        self._bundle.load()

    def apply(self, sample: Dict[str, Any], **kwargs):
        return self._bundle.provide(sample, **kwargs)


class _PreprocessorAdapter:
    """Adapts DatasetPreprocessor implementations to the legacy handle API."""

    def __init__(self, preprocessor) -> None:
        self._preprocessor = preprocessor

    def apply(self, sample: Dict[str, Any], **kwargs):
        return self._preprocessor.transform(sample, **kwargs)


def resolve_callable(ref: Optional[Any]) -> Optional[Callable[[Dict[str, Any]], Any]]:
    """Deprecated shim: import from reflection utils."""
    return _resolve_callable(ref)


def resolve_doc_to_callable(spec: DatasetSpec, field: str) -> Optional[Callable[[Dict[str, Any]], Any]]:
    """Resolve doc_to_* callables and optionally curry keyword arguments."""

    ref = spec.params.get(field)
    if ref is None:
        return None
    func = resolve_callable(ref)
    preprocess_kwargs = coerce_kwargs(spec.params.get("preprocess_kwargs"))
    provided_kwargs = coerce_kwargs(spec.params.get(f"{field}_kwargs"))
    kwargs = _merge_doc_to_kwargs(func, preprocess_kwargs, provided_kwargs)
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
    handle = _resolve_registered_preprocessor(module, kwargs)
    if handle:
        return PreprocessContext(handle=handle, kwargs=dict(kwargs))
    return None

def build_bundle_context(spec: DatasetSpec, *, data_path: Optional[str]) -> Optional[BundleContext]:
    """Create a bundle context shared by JSONL/HF loaders."""

    module = spec.params.get("bundle")
    if not module:
        return None
    kwargs = coerce_kwargs(spec.params.get("bundle_kwargs"))
    if data_path is not None:
        kwargs.setdefault("data_path", data_path)
    handle = _resolve_registered_bundle(module, kwargs)
    if handle:
        return BundleContext(handle=handle, kwargs=dict(kwargs))
    return None

def _resolve_registered_bundle(name: str, kwargs: Dict[str, Any]):
    """Resolve registered class-based bundle by registry name."""
    if not name:
        return None
    try:
        bundle_cls = registry.get("bundles", name)
    except KeyError:
        return None
    bundle = bundle_cls(**kwargs)
    return _BundleAdapter(bundle)

def _resolve_registered_preprocessor(name: str, kwargs: Dict[str, Any]):
    """Resolve registered class-based preprocessor by registry name."""
    if not name:
        return None
    try:
        preprocessor_cls = registry.get("dataset_preprocessors", name)
    except KeyError:
        return None
    preprocessor = preprocessor_cls(**kwargs)
    return _PreprocessorAdapter(preprocessor)


def inject_default_params(record: Dict[str, Any], default_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure default sampling params exist on the given sample dict."""

    if not default_params:
        return record
    if "sampling_params" not in record:
        record["sampling_params"] = default_params
    if "generation_params" not in record:
        record["generation_params"] = default_params  # backward compatibility
    return record


def apply_bundle(
    records: Iterable[Dict[str, Any]],
    spec: DatasetSpec,
    *,
    data_path: Optional[str],
    doc_to_text: Optional[Callable[[Dict[str, Any]], Any]] = None,
    doc_to_visual: Optional[Callable[[Dict[str, Any]], Any]] = None,
    doc_to_audio: Optional[Callable[[Dict[str, Any]], Any]] = None,
    trace: Optional["ObservabilityTrace"] = None,
    observability_config: Optional["ObservabilityConfig"] = None,
) -> Iterable[Dict[str, Any]]:
    config = observability_config or get_observability_config()
    ctx = build_bundle_context(spec, data_path=data_path)
    if not ctx:
        return records

    def generator():
        for record in records:
            record_dict = dict(record)
            record_dict.setdefault("_dataset_id", spec.dataset_id)
            if data_path and "_dataset_metadata" not in record_dict:
                record_dict["_dataset_metadata"] = {"path": data_path}
            new_record = ctx.handle.apply(
                record_dict,
                dataset_id=spec.dataset_id,
                dataset_metadata=record_dict.get("_dataset_metadata"),
                doc_to_text=doc_to_text,
                doc_to_visual=doc_to_visual,
                doc_to_audio=doc_to_audio,
                trace=trace,
                observability_config=config,
                **ctx.kwargs,
            )
            yield new_record

    return generator()

def apply_preprocess(
    records: Iterable[Dict[str, Any]],
    spec: DatasetSpec,
    *,
    data_path: Optional[str],
    doc_to_text: Optional[Callable[[Dict[str, Any]], Any]] = None,
    doc_to_visual: Optional[Callable[[Dict[str, Any]], Any]] = None,
    doc_to_audio: Optional[Callable[[Dict[str, Any]], Any]] = None,
    trace: Optional["ObservabilityTrace"] = None,
    observability_config: Optional["ObservabilityConfig"] = None,
) -> Iterable[Dict[str, Any]]:
    config = observability_config or get_observability_config()
    ctx = build_preprocess_context(spec, data_path=data_path)
    # NOTE: If no explicit preprocess is configured, fall back to DefaultPreprocessor
    # (llm-eval compatible defaults + fallback prompt templating).
    if not ctx:
        return _apply_default_preprocessor(
            records,
            spec,
            data_path=data_path,
            doc_to_text=doc_to_text,
            doc_to_visual=doc_to_visual,
            doc_to_audio=doc_to_audio,
            trace=trace,
            observability_config=config,
        )

    def generator():
        for record in records:
            new_record = dict(record)
            new_record.setdefault("_dataset_id", spec.dataset_id)
            if data_path and "_dataset_metadata" not in new_record:
                new_record["_dataset_metadata"] = {"path": data_path}
            result = ctx.handle.apply(
                new_record,
                dataset_id=spec.dataset_id,
                dataset_metadata=new_record.get("_dataset_metadata"),
                doc_to_text=doc_to_text,
                doc_to_visual=doc_to_visual,
                doc_to_audio=doc_to_audio,
                trace=trace,
                observability_config=config,
                **ctx.kwargs,
            )
            if result is None:
                continue
            yield result

    return generator()


def _apply_default_preprocessor(
    records: Iterable[Dict[str, Any]],
    spec: DatasetSpec,
    *,
    data_path: Optional[str],
    doc_to_text: Optional[Callable[[Dict[str, Any]], Any]],
    doc_to_visual: Optional[Callable[[Dict[str, Any]], Any]],
    doc_to_audio: Optional[Callable[[Dict[str, Any]], Any]],
    trace: Optional["ObservabilityTrace"],
    observability_config: Optional["ObservabilityConfig"],
):
    from gage_eval.assets.datasets.preprocessors.default_preprocessor import DefaultPreprocessor

    tok_path = resolve_tokenizer_name(spec.params)
    tokenizer = get_or_load_tokenizer(spec.params)
    default_pre = DefaultPreprocessor(tokenizer=tokenizer, tokenizer_path=tok_path)

    def generator():
        for record in records:
            new_record = dict(record)
            new_record.setdefault("_dataset_id", spec.dataset_id)
            if data_path and "_dataset_metadata" not in new_record:
                new_record["_dataset_metadata"] = {"path": data_path}
            inputs_val = default_pre.transform(
                new_record,
                dataset_id=spec.dataset_id,
                dataset_metadata={"path": data_path} if data_path else None,
                doc_to_text=doc_to_text,
                doc_to_visual=doc_to_visual,
                doc_to_audio=doc_to_audio,
                trace=trace,
                observability_config=observability_config,
            )
            if inputs_val is None:
                continue
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


def coerce_kwargs(raw: Any) -> Dict[str, Any]:  # shim
    return _coerce_kwargs(raw)


def _merge_doc_to_kwargs(func: Callable[..., Any], preprocess_kwargs: Dict[str, Any], provided_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer preprocess_kwargs for doc_to_* if keys match callable signature."""

    merged: Dict[str, Any] = {}
    if preprocess_kwargs:
        merged.update(_filter_doc_to_kwargs(func, preprocess_kwargs))
    if provided_kwargs:
        merged.update(_filter_doc_to_kwargs(func, provided_kwargs))
    return merged


def _filter_doc_to_kwargs(func: Callable[..., Any], candidates: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by the target callable unless **kwargs is present."""

    if not candidates:
        return {}
    signature = inspect.signature(func)
    params = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    if accepts_var_kwargs:
        return dict(candidates)
    allowed = {
        name
        for name, param in params.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in candidates.items() if k in allowed}


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
