"""Dataclass representation of Sample with adapters to/from dict."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from typing import Any, Dict, List, Optional, Union

SCHEMA_VERSION = "0.0.1"

@dataclass
class MessageContent:
    type: str  # text/image_url/audio_url/video_url/file_url/image/audio/video/file
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None
    audio_url: Optional[Dict[str, Any]] = None
    video_url: Optional[Dict[str, Any]] = None
    file_url: Optional[Dict[str, Any]] = None
    image: Optional[Any] = None
    audio: Optional[Any] = None
    video: Optional[Any] = None
    file: Optional[Any] = None

@dataclass
class Message:
    role: str
    content: List[MessageContent] = field(default_factory=list)

@dataclass
class PredictResult:
    index: int
    message: Message
    raw_response: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None

@dataclass
class Sample:
    schema_version: str
    id: str
    messages: List[Message]
    task_type: Optional[Any] = None
    options: Optional[List[str]] = None
    references: List[Any] = field(default_factory=list)
    label: Optional[str] = None
    few_shot_examples: Optional[List[Any]] = None
    golden_trajectories: Optional[List[Any]] = None
    sandbox: Optional[Dict[str, Any]] = None    
    metadata: Optional[Dict[str, Any]] = None
    data_tag: Optional[Dict[str, Any]] = None
    raw_assets: Optional[Dict[str, Any]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    sampling_params: Optional[Dict[str, Any]] = None
    generation_params: Optional[Dict[str, Any]] = None
    eval_config: Optional[Dict[str, Any]] = None
    unconditioned_input: Optional[Union[str, list[Any]]] = None
    predict_result: List[PredictResult] = field(default_factory=list)
    eval_result: Dict[str, Any] = field(default_factory=dict)


def sample_from_dict(payload: Dict[str, Any]) -> Sample:
    """Lightweight deserializer: dict -> Sample dataclass."""

    def _filter_fields(data: Dict[str, Any], allowed_keys) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if k in allowed_keys}

    def build_message(msg: Dict[str, Any]) -> Message:
        raw_content = msg.get("content") or []
        if isinstance(raw_content, dict):
            raw_content = [raw_content]
        normalized_content: List[MessageContent] = []
        for frag in raw_content:
            if isinstance(frag, dict):
                safe = _filter_fields(frag, MessageContent.__dataclass_fields__.keys())
                normalized_content.append(MessageContent(**safe))
            else:
                normalized_content.append(MessageContent(type="text", text=str(frag)))
        extras = {k: msg.get(k) for k in ("tool_calls", "tool_use", "model_output", "path", "name") if k in msg}
        return Message(role=msg.get("role", "user"), content=normalized_content, **extras)

    messages = [build_message(m) for m in payload.get("messages", []) if isinstance(m, dict)]
    messages = [build_message(m) for m in payload.get("messages", []) if isinstance(m, dict)]
    
    # NOTE: 'choices', 'inputs', 'audit_info' are ignored as strict Sample schema does not support them.

    preds = [
        PredictResult(
            index=pr.get("index", i),
            message=build_message(pr.get("message", {})),
            raw_response=pr.get("raw_response"),
            usage=pr.get("usage"),
            latency_ms=pr.get("latency_ms"),
        )
        for i, pr in enumerate(payload.get("predict_result", []))
        if isinstance(pr, dict)
    ]
    
    return Sample(
        schema_version=SCHEMA_VERSION,
        id=str(payload.get("id")),
        messages=messages,
        metadata=payload.get("metadata") or {},
        data_tag=payload.get("data_tag") or {},
        label=payload.get("label"),
        sampling_params=payload.get("sampling_params") or {},
        generation_params=payload.get("generation_params") or {},
        predict_result=preds,
        eval_result=payload.get("eval_result") or {},
    )


def sample_to_dict(sample: Sample) -> Dict[str, Any]:
    """Serialize Sample dataclass to plain dict."""

    return asdict(sample)


def append_prediction(sample: Sample, model_output: Dict[str, Any]) -> None:
    """Append model_output into predict_result of Sample dataclass."""

    if not isinstance(model_output, dict):
        return
    idx = len(sample.predict_result)
    msg = model_output.get("message") or {
        "role": "assistant",
        "content": [{"type": "text", "text": model_output.get("answer", "")}],
    }
    built_message = sample_from_dict({"messages": [msg]}).messages[0] if isinstance(msg, dict) else Message(
        role="assistant", content=[MessageContent(type="text", text=str(msg))]
    )
    sample.predict_result.append(
        PredictResult(
            index=model_output.get("index", idx),
            message=built_message,
            raw_response=model_output.get("raw_response"),
            usage=model_output.get("usage"),
            latency_ms=model_output.get("latency_ms"),
        )
    )
