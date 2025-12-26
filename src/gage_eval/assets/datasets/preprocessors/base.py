"""Dataset preprocessor abstractions (template method)."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from gage_eval.assets.datasets.utils.multimodal import merge_multimodal_inputs
from gage_eval.assets.datasets.utils.normalization import normalize_sample, ensure_chat_template_flags
from gage_eval.assets.datasets.validation import validate_sample_schema
from gage_eval.observability.config import get_observability_config

_DOC_TO_KEYS = ("doc_to_text", "doc_to_visual", "doc_to_audio")


class DatasetPreprocessor:
    """Base class for structured preprocessors."""

    name = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def transform(self, sample: Dict[str, Any], **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


class BasePreprocessor(DatasetPreprocessor):
    """Template-method preprocessor for standardizing dataset records.

    This base class centralizes the shared preprocessing pipeline so that
    dataset-specific preprocessors can focus on `to_sample()` only.

    Responsibilities covered here include:
    - optional message role stripping
    - structuring a raw record into the Sample schema via `to_sample()`
    - normalizing `inputs` into a dict form
    - applying optional `doc_to_*` hooks
    - merging multimodal references and de-duplicating them
    - enforcing schema validation and canonical normalization
    """

    def __init__(
        self,
        *,
        on_error: str = "skip",
        ensure_inputs_dict: bool = True,
        roles_to_remove: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.on_error = on_error
        self.ensure_inputs_dict = ensure_inputs_dict
        self.roles_to_remove = tuple(roles_to_remove) if roles_to_remove else ()

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def transform(self, sample: Dict[str, Any], **kwargs: Any) -> Any:
        observability_config = kwargs.pop("observability_config", None)
        trace = kwargs.get("trace")
        cfg = observability_config or (get_observability_config() if trace else None)
        is_debug = _env_flag("GAGE_EVAL_DEBUG_PREPROCESS", default=False)
        debug_sample_rate = _env_float("GAGE_EVAL_DEBUG_PREPROCESS_SAMPLE", default=0.01, minimum=0.0, maximum=1.0)
        debug_log_path = os.environ.get("GAGE_EVAL_DEBUG_PREPROCESS_DIR") or os.path.join(
            os.getcwd(), "runs", "debug_preprocess.jsonl"
        )
        pre_snapshot = _snapshot(sample) if is_debug else None
        sample_id_hint = str(sample.get("id") or sample.get("_dataset_id") or "unknown")
        emit_trace = bool(trace and cfg and cfg.enabled and cfg.should_sample("preprocess", sample_id=sample_id_hint))
        start_time: Optional[float] = time.perf_counter() if emit_trace else None
        if emit_trace:
            trace.emit(
                "preprocess_start",
                {"id": sample_id_hint, "keys": list(sample.keys()), "dataset_id": sample.get("_dataset_id")},
                sample_id=sample_id_hint,
            )
        doc_to_hooks = {k: kwargs.get(k) for k in _DOC_TO_KEYS}
        to_sample_kwargs = {k: v for k, v in kwargs.items() if k not in _DOC_TO_KEYS and k != "trace"}
        try:
            # STEP 1: Optionally strip specific message roles for clean evaluation inputs.
            if self.roles_to_remove:
                _strip_roles(sample, roles_to_remove=self.roles_to_remove)
            # STEP 2: Let the dataset-specific preprocessor structure the record.
            structured_sample = self.to_sample(sample, **to_sample_kwargs)
            # STEP 3: Validate the sample schema early to fail fast.
            validate_sample_schema(structured_sample)
            # STEP 4: Merge multimodal inputs and de-duplicate referenced assets.
            # merge_multimodal_inputs(sample)

            dataset_id = sample.get("_dataset_id") or kwargs.get("dataset_id") or "unknown"
            dataset_meta = sample.get("_dataset_metadata") or kwargs.get("dataset_metadata") or {}
            if emit_trace:
                msgs = structured_sample.messages or []
                trace.emit(
                    "preprocess_structured",
                    {
                        "msg_count": len(msgs) if isinstance(msgs, list) else 0,
                    },
                    sample_id=sample_id_hint,
                )
                cost_ms = (time.perf_counter() - start_time) * 1000 if start_time else 0.0
                trace.emit(
                    "preprocess_done",
                    {"id": sample.get("id") or sample_id_hint, "cost_ms": cost_ms},
                    sample_id=sample_id_hint,
                )
            if is_debug and random.random() < debug_sample_rate:
                _append_debug_record(
                    debug_log_path,
                    {
                        "sample_id": sample.get("id") or sample_id_hint,
                        "stage": "done",
                        "pre": pre_snapshot,
                        "post": _snapshot(structured_sample),
                    },
                )
            return structured_sample
        except Exception as exc:
            if emit_trace:
                trace.emit(
                    "preprocess_error",
                    {
                        "id": sample_id_hint,
                        "dataset_id": sample.get("_dataset_id"),
                        "error": repr(exc),
                        "keys": list(sample.keys()),
                    },
                    sample_id=sample_id_hint,
                )
            if is_debug and random.random() < max(debug_sample_rate, 1e-9):
                _append_debug_record(
                    debug_log_path,
                    {
                        "sample_id": sample.get("id") or sample_id_hint,
                        "stage": "error",
                        "pre": pre_snapshot,
                        "error": repr(exc),
                    },
                )
            if is_debug or self.on_error == "raise":
                raise
            meta = sample.get("metadata") or {}
            meta["preprocess_error"] = str(exc)
            sample["metadata"] = meta
            return None

    def _apply_doc_to(
        self,
        sample: Dict[str, Any],
        *,
        doc_to_text: Optional[Any],
        doc_to_visual: Optional[Any],
        doc_to_audio: Optional[Any],
    ) -> None:
        inputs_val = sample.get("inputs") if isinstance(sample.get("inputs"), dict) else {}
        has_inputs = bool(inputs_val)
        has_messages = isinstance(sample.get("messages"), list) and bool(sample["messages"])
        has_multi_modal = bool(inputs_val.get("multi_modal_data"))
        if doc_to_text and not has_inputs and not has_messages:
            sample["text"] = doc_to_text(sample)
        if doc_to_visual and not has_multi_modal:
            sample["visual"] = doc_to_visual(sample)
        if doc_to_audio and not has_multi_modal:
            sample["audio"] = doc_to_audio(sample)


def _strip_roles(sample: Dict[str, Any], *, roles_to_remove: Sequence[str]) -> None:
    """Remove messages with roles in roles_to_remove and record metadata."""

    roles = {r.lower() for r in roles_to_remove if isinstance(r, str)}
    msgs = sample.get("messages")
    if not isinstance(msgs, list) or not roles:
        return
    kept = []
    removed_roles = set()
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).lower()
        if role in roles:
            removed_roles.add(role)
            continue
        kept.append(m)
    sample["messages"] = kept
    if removed_roles:
        meta = sample.get("metadata") or {}
        prev = set(meta.get("removed_roles") or [])
        meta["removed_roles"] = sorted(prev.union(removed_roles))
        dropped = meta.get("_dropped_fields") or []
        dropped.append({"type": "message_roles", "roles": sorted(removed_roles)})
        meta["_dropped_fields"] = dropped
        sample["metadata"] = meta


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, *, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _snapshot(obj: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return {}


def _append_debug_record(path: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str))
            handle.write("\n")
    except Exception:
        logger.debug("Failed to write debug preprocess record to {}", path)


def _maybe_log_diff(pre_snapshot: Optional[Dict[str, Any]], post: Dict[str, Any], sample_id: str) -> None:
    if pre_snapshot is None:
        return
    before_keys = set(pre_snapshot.keys())
    after_keys = set(post.keys())
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    if added or removed:
        logger.debug(
            "Preprocess debug sample_id={} added_keys={} removed_keys={}",
            sample_id,
            added,
            removed,
        )


__all__ = ["DatasetPreprocessor", "BasePreprocessor"]
