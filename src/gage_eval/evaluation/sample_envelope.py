"""Helpers for mutating and snapshotting the canonical Sample envelope."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence


_MESSAGE_EXTRAS = ("tool_calls", "tool_use", "model_output", "name", "path")


def append_predict_result(sample: Dict[str, Any], model_output: Optional[Dict[str, Any]]) -> None:
    """Append a DUT 输出到 sample['predict_result']，保留原始字段并补齐 message。"""

    if not isinstance(model_output, dict) or not model_output:
        return

    predict_result = sample.setdefault("predict_result", [])
    if not isinstance(predict_result, list):
        predict_result = sample["predict_result"] = []

    entry = copy.deepcopy(model_output)
    entry.setdefault("index", len(predict_result))
    if "message" not in entry:
        entry["message"] = _build_message(entry)
    predict_result.append(entry)


def update_eval_result(sample: Dict[str, Any], judge_output: Optional[Dict[str, Any]]) -> None:
    """Merge裁判输出到 sample['eval_result']。"""

    if not isinstance(judge_output, dict) or not judge_output:
        return
    eval_result = sample.setdefault("eval_result", {})
    if not isinstance(eval_result, dict):
        eval_result = sample["eval_result"] = {}
    for key, value in judge_output.items():
        eval_result[key] = copy.deepcopy(value)


def snapshot_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """深拷贝样本，确保写盘时不受后续修改影响。"""

    try:
        return json.loads(json.dumps(sample, ensure_ascii=False))
    except (TypeError, ValueError):
        # Fallback to deepcopy（允许存在非 JSON 兼容字段，写盘时再处理）。
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
