"""Dataclass representation of Sample with adapters to/from dict."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    tool_calls: Optional[Any] = None
    tool_use: Optional[Any] = None
    model_output: Optional[Any] = None
    path: Optional[str] = None
    name: Optional[str] = None

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
    instruction: Optional[str] = None
    choices: Optional[List[Any]] = None
    prompt: Optional[str] = None
    text: Optional[str] = None
    inputs: Optional[Any] = None
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
    expected_answer: Optional[str] = None
    sampling_params: Optional[Dict[str, Any]] = None
    generation_params: Optional[Dict[str, Any]] = None
    eval_config: Optional[Dict[str, Any]] = None
    unconditioned_input: Optional[Union[str, list[Any]]] = None
    audit_info: Optional[Dict[str, Any]] = None
    chat_template_mode: Optional[str] = None
    rendered_by: Optional[str] = None
    template_source: Optional[str] = None
    cache_suffix: Optional[str] = None
    _media_meta: Optional[Dict[str, Any]] = None
    _tokenizer_path: Optional[str] = None
    predict_result: List[PredictResult] = field(default_factory=list)
    eval_result: Dict[str, Any] = field(default_factory=dict)


def sample_from_dict(payload: Dict[str, Any]) -> Sample:
    """Lightweight deserializer: dict -> Sample dataclass."""

    def _filter_fields(data: Dict[str, Any], allowed_keys) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if k in allowed_keys}

    def _normalize_message_content(raw_content: Any) -> List[MessageContent]:
        if raw_content is None:
            return []
        if isinstance(raw_content, dict):
            raw_fragments = [raw_content]
        elif isinstance(raw_content, list):
            raw_fragments = raw_content
        else:
            raw_fragments = [raw_content]

        normalized_content: List[MessageContent] = []
        for frag in raw_fragments:
            if isinstance(frag, dict):
                safe = _filter_fields(frag, MessageContent.__dataclass_fields__.keys())
                normalized_content.append(MessageContent(**safe))
            else:
                normalized_content.append(MessageContent(type="text", text=str(frag)))
        return normalized_content

    def build_message(msg: Dict[str, Any]) -> Message:
        raw_content = msg.get("content")
        normalized_content = _normalize_message_content(raw_content)
        extras = {k: msg.get(k) for k in ("tool_calls", "tool_use", "model_output", "path", "name") if k in msg}
        return Message(role=msg.get("role", "user"), content=normalized_content, **extras)

    messages = [build_message(m) for m in payload.get("messages", []) if isinstance(m, dict)]

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
        schema_version=str(payload.get("schema_version") or SCHEMA_VERSION),
        id=str(payload.get("id") or payload.get("sample_id") or ""),
        messages=messages,
        instruction=payload.get("instruction") if payload.get("instruction") is not None else None,
        choices=payload.get("choices") if isinstance(payload.get("choices"), list) else None,
        prompt=payload.get("prompt") if payload.get("prompt") is not None else None,
        text=payload.get("text") if payload.get("text") is not None else None,
        inputs=payload.get("inputs"),
        task_type=payload.get("task_type"),
        options=payload.get("options") if isinstance(payload.get("options"), list) else None,
        references=list(payload.get("references") or []),
        few_shot_examples=payload.get("few_shot_examples"),
        golden_trajectories=payload.get("golden_trajectories"),
        sandbox=payload.get("sandbox"),
        metadata=payload.get("metadata") or {},
        data_tag=payload.get("data_tag") or {},
        raw_assets=payload.get("raw_assets"),
        label=payload.get("label"),
        tools=payload.get("tools") if isinstance(payload.get("tools"), list) else None,
        tool_choice=payload.get("tool_choice"),
        expected_answer=(
            str(payload.get("expected_answer"))
            if payload.get("expected_answer") is not None
            else None
        ),
        sampling_params=payload.get("sampling_params") or {},
        generation_params=payload.get("generation_params") or {},
        eval_config=payload.get("eval_config") or {},
        unconditioned_input=payload.get("unconditioned_input"),
        audit_info=payload.get("audit_info") if isinstance(payload.get("audit_info"), dict) else None,
        chat_template_mode=payload.get("chat_template_mode"),
        rendered_by=payload.get("rendered_by"),
        template_source=payload.get("template_source"),
        cache_suffix=payload.get("cache_suffix"),
        _media_meta=payload.get("_media_meta") if isinstance(payload.get("_media_meta"), dict) else None,
        _tokenizer_path=payload.get("_tokenizer_path"),
        predict_result=preds,
        eval_result=payload.get("eval_result") or {},
    )


def sample_to_dict(sample: Sample) -> Dict[str, Any]:
    """Serialize Sample dataclass to plain dict."""

    return _prune_none(asdict(sample))


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


def _prune_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _prune_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_prune_none(item) for item in value]
    return value
