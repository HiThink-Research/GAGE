"""Dataclass representation of Sample with adapters to/from dict."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


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
    tool_calls: Optional[Any] = None
    tool_use: Optional[Any] = None
    model_output: Optional[Any] = None
    path: Optional[Any] = None
    name: Optional[str] = None


@dataclass
class Choice:
    index: int
    message: Message
    label: Optional[str] = None


@dataclass
class PredictResult:
    index: int
    message: Message
    raw_response: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None


@dataclass
class Inputs:
    prompt: Optional[str] = None
    input_ids: Optional[List[int]] = None
    multi_modal_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditInfo:
    task_id: Optional[str] = None
    version_id: Optional[str] = None
    query_id: Optional[str] = None
    self_or_open_ai: Optional[str] = None
    check_user: Optional[str] = None
    check_time: Optional[str] = None
    create_at: Optional[str] = None
    create_by: Optional[str] = None
    review_user: Optional[str] = None
    review_time: Optional[str] = None


@dataclass
class Sample:
    id: str
    _dataset_id: str
    messages: List[Message]
    choices: List[Choice] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    data_tag: Dict[str, Any] = field(default_factory=dict)
    label: Optional[Any] = None
    inputs: Inputs = field(default_factory=Inputs)
    _dataset_metadata: Dict[str, Any] = field(default_factory=dict)
    _media_meta: Dict[str, Any] = field(default_factory=dict)
    _tokenizer_path: Optional[str] = None
    chat_template_mode: Optional[str] = None
    rendered_by: Optional[str] = None
    template_source: Optional[str] = None
    cache_suffix: Optional[str] = None
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)
    predict_result: List[PredictResult] = field(default_factory=list)
    eval_result: Dict[str, Any] = field(default_factory=dict)
    audit_info: AuditInfo = field(default_factory=AuditInfo)


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
    choices = [
        Choice(index=c.get("index", i), message=build_message(c.get("message", {})), label=c.get("label"))
        for i, c in enumerate(payload.get("choices", []))
        if isinstance(c, dict)
    ]
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
    inputs_raw = payload.get("inputs") or {}
    inputs = Inputs(
        prompt=inputs_raw.get("prompt"),
        input_ids=inputs_raw.get("input_ids"),
        multi_modal_data=inputs_raw.get("multi_modal_data") or {},
    )
    audit_raw = payload.get("audit_info") or {}
    audit = AuditInfo(**{k: audit_raw.get(k) for k in AuditInfo.__dataclass_fields__})

    return Sample(
        id=str(payload.get("id")),
        _dataset_id=str(payload.get("_dataset_id", "")),
        messages=messages,
        choices=choices,
        metadata=payload.get("metadata") or {},
        data_tag=payload.get("data_tag") or {},
        label=payload.get("label"),
        inputs=inputs,
        _dataset_metadata=payload.get("_dataset_metadata") or {},
        _media_meta=payload.get("_media_meta") or {},
        _tokenizer_path=payload.get("_tokenizer_path"),
        chat_template_mode=payload.get("chat_template_mode"),
        rendered_by=payload.get("rendered_by"),
        template_source=payload.get("template_source"),
        cache_suffix=payload.get("cache_suffix"),
        sampling_params=payload.get("sampling_params") or {},
        generation_params=payload.get("generation_params") or {},
        predict_result=preds,
        eval_result=payload.get("eval_result") or {},
        audit_info=audit,
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
