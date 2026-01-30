"""Helpers for mutating and snapshotting the canonical Sample envelope."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence


_MESSAGE_EXTRAS = ("tool_calls", "tool_use", "model_output", "name", "path")


def append_predict_result(sample: Dict[str, Any], model_output: Optional[Dict[str, Any]]) -> None:
    """Append a DUT output entry to `sample["predict_result"]`.

    The original fields are preserved, and a canonical `message` field is added
    if missing. Multi-sample outputs are split into per-candidate entries when
    `_sample_n` is present and `answer` is a list.
    """

    if not isinstance(model_output, dict) or not model_output:
        return

    predict_result = sample.setdefault("predict_result", [])
    if not isinstance(predict_result, list):
        predict_result = sample["predict_result"] = []

    answer = model_output.get("answer")
    if _should_split_predict_result(model_output, answer):
        _append_split_predict_results(predict_result, model_output, answer)
        return

    entry = copy.deepcopy(model_output)
    entry.setdefault("index", len(predict_result))
    if "message" not in entry:
        entry["message"] = _build_message(entry)
    predict_result.append(entry)


def _should_split_predict_result(model_output: Mapping[str, Any], answer: Any) -> bool:
    if not isinstance(answer, list):
        return False
    sample_n = model_output.get("_sample_n")
    if isinstance(sample_n, int):
        return sample_n > 1
    if isinstance(sample_n, str) and sample_n.isdigit():
        return int(sample_n) > 1
    return False


def _append_split_predict_results(
    predict_result: List[Dict[str, Any]],
    model_output: Dict[str, Any],
    answers: List[Any],
) -> None:
    for idx, answer in enumerate(answers):
        entry = copy.deepcopy(model_output)
        entry["answer"] = answer
        entry.pop("message", None)
        entry["index"] = len(predict_result)
        entry["candidate_index"] = idx
        entry["message"] = _build_message(entry)
        predict_result.append(entry)


def latest_predict_result(sample: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the latest predict_result entry if available.

    Args:
        sample: Sample mapping that may contain predict_result.

    Returns:
        The most recent predict_result dict, or None when unavailable.
    """

    if not isinstance(sample, Mapping):
        return None
    predict_result = sample.get("predict_result")
    if isinstance(predict_result, list) and predict_result:
        latest = predict_result[-1]
        if isinstance(latest, Mapping):
            return dict(latest)
    return None


def resolve_model_output(
    sample: Optional[Mapping[str, Any]],
    model_output: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve model_output with predict_result and legacy fallbacks.

    Args:
        sample: Sample mapping that may contain predict_result or model_output.
        model_output: Explicit model_output payload when provided.

    Returns:
        A dict-shaped model_output, preferring explicit payload then predict_result.
    """

    if isinstance(model_output, Mapping) and model_output:
        return dict(model_output)
    latest = latest_predict_result(sample)
    if latest:
        return latest
    if isinstance(sample, Mapping):
        legacy = sample.get("model_output")
        if isinstance(legacy, Mapping) and legacy:
            return dict(legacy)
    return {}


def update_eval_result(sample: Dict[str, Any], judge_output: Optional[Dict[str, Any]]) -> None:
    """Merges judge output into `sample["eval_result"]`."""

    if not isinstance(judge_output, dict) or not judge_output:
        return
    eval_result = sample.setdefault("eval_result", {})
    if not isinstance(eval_result, dict):
        eval_result = sample["eval_result"] = {}
    for key, value in judge_output.items():
        eval_result[key] = copy.deepcopy(value)


def resolve_judge_output(
    sample: Optional[Mapping[str, Any]],
    judge_output: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve judge_output with eval_result and legacy fallbacks.

    Args:
        sample: Sample mapping that may contain eval_result or judge_output.
        judge_output: Explicit judge_output payload when provided.

    Returns:
        A dict-shaped judge_output, preferring explicit payload then eval_result.
    """

    if isinstance(judge_output, Mapping) and judge_output:
        return dict(judge_output)
    if isinstance(sample, Mapping):
        eval_result = sample.get("eval_result")
        if isinstance(eval_result, Mapping) and eval_result:
            return dict(eval_result)
        legacy = sample.get("judge_output")
        if isinstance(legacy, Mapping) and legacy:
            return dict(legacy)
    return {}


def snapshot_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a deep snapshot of a sample for safe persistence."""

    try:
        return json.loads(json.dumps(sample, ensure_ascii=False))
    except (TypeError, ValueError):
        # Fall back to deepcopy. This allows non-JSON-serializable fields and lets
        # the writer handle serialization later.
        return copy.deepcopy(sample)


def _build_message(output: Dict[str, Any]) -> Dict[str, Any]:
    message = output.get("message")
    if isinstance(message, dict):
        return _normalize_message(message)

    messages = output.get("messages")
    if isinstance(messages, Sequence) and messages:
        candidate = messages[-1]
        if isinstance(candidate, dict):
            return _normalize_message(candidate)

    answer = output.get("answer") or output.get("text") or output.get("content")
    if isinstance(answer, dict):
        return _normalize_message({"role": "assistant", "content": answer.get("content") or answer})
    if isinstance(answer, list):
        return _normalize_message({"role": "assistant", "content": answer})
    return {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "" if answer is None else str(answer),
            }
        ],
    }


def _normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    role = message.get("role") or "assistant"
    content = message.get("content")
    normalized = {
        "role": role,
        "content": _normalize_content_list(content),
    }
    for extra in _MESSAGE_EXTRAS:
        if extra in message:
            normalized[extra] = copy.deepcopy(message[extra])
    return normalized


def _normalize_content_list(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        return [_normalize_content(fragment) for fragment in content if fragment is not None]
    if isinstance(content, dict):
        return [_normalize_content(content)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def _normalize_content(fragment: Any) -> Dict[str, Any]:
    if isinstance(fragment, dict):
        fragment_type = fragment.get("type")
        if fragment_type in {"text", "image_url", "audio_url", "video_url", "file_url"}:
            payload = fragment.get(fragment_type) or fragment.get("url")
            if fragment_type == "text":
                text_value = fragment.get("text")
                if text_value is not None:
                    return {"type": "text", "text": str(text_value)}
                return {"type": "text", "text": ""}
            if isinstance(payload, dict):
                return {"type": fragment_type, fragment_type: payload}
            if payload is not None:
                return {"type": fragment_type, fragment_type: {"url": str(payload)}}
        if "text" in fragment and not fragment.get("type"):
            return {"type": "text", "text": str(fragment["text"])}
        try:
            return {"type": "text", "text": json.dumps(fragment, ensure_ascii=False)}
        except (TypeError, ValueError):
            return {"type": "text", "text": str(fragment)}

    if isinstance(fragment, str):
        return {"type": "text", "text": fragment}

    return {"type": "text", "text": str(fragment)}
