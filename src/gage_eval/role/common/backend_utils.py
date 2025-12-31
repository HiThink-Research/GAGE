"""Shared helpers for backend request handling (request_id, metadata, placeholders)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import io
import re
import uuid
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from gage_eval.role.model.runtime import ChatTemplateMixin
from gage_eval.utils.cleanup import torch_gpu_cleanup
from gage_eval.utils.messages import normalize_messages_for_template, stringify_message_content


from gage_eval.utils.multimodal import load_multimodal_data

__all__ = [
    "resolve_request_id",
    "attach_template_metadata",
    "finalize_backend_result",
    "run_coroutine_threadsafe_with_timeout",
    "normalize_image_placeholders",
    "convert_text_like_output",
    "build_sampling_params_base",
    "_media_dedup_key",
    "extract_images_from_messages",
    "has_multimodal_inputs",
    "extract_audios_from_messages",
    "dedup_media_sources",
    "_merge_multimodal_sources",
    "collect_multimodal_sources",
    "normalize_messages_safe",
    "check_tokenizer_conflict",
    "maybe_tokenize_messages",
    "graceful_loop_shutdown",
    "render_prompt_with_template",
    "prepare_multi_modal_data",
    "load_images",
    "load_multimodal_payload",
    "simple_render_messages",
    "render_with_processor",
]


def resolve_request_id(payload: Dict[str, Any], *, prefix: str) -> str:
    """Resolve a non-empty request_id from common fields or generate one with prefix."""

    candidate = payload.get("request_id")
    sample = payload.get("sample") if isinstance(payload, dict) else None
    if not candidate and isinstance(sample, dict):
        for key in ("request_id", "id", "sample_id", "idx"):
            candidate = sample.get(key)
            if candidate:
                break
    if not candidate and isinstance(payload, dict):
        candidate = payload.get("id") or payload.get("sample_id")

    request_id = str(candidate).strip() if candidate is not None else ""
    if not request_id:
        request_id = f"{prefix}_{uuid.uuid4().hex}"
    return request_id


def attach_template_metadata(
    prepared: Dict[str, Any],
    result: Dict[str, Any],
    *,
    cfg_tokenizer_path: Any = None,
) -> None:
    """Attach chat_template_* metadata and tokenizer path to result if present."""

    meta_keys = ("chat_template_mode", "template_source", "rendered_by", "cache_suffix")
    sample = prepared.get("sample") or {}
    for key in meta_keys:
        value = prepared.get(key)
        if value is None and isinstance(sample, dict):
            value = sample.get(key)
        if value is not None and key not in result:
            result[key] = value
    if "_tokenizer_path" not in result and cfg_tokenizer_path:
        result["_tokenizer_path"] = cfg_tokenizer_path


def finalize_backend_result(
    prepared: Dict[str, Any],
    outputs: List[Any],
    *,
    sample_n: int,
    batch_path: str,
    backend_tag: str,
    cfg_tokenizer_path: Any = None,
) -> Dict[str, Any]:
    """Normalize backend output shape with metadata from prepared inputs."""

    if sample_n > 1:
        result = {"answer": outputs, "_backend": backend_tag, "_sample_n": sample_n, "_batch_path": batch_path}
    else:
        result = {"answer": outputs[0] if outputs else "", "_backend": backend_tag, "_batch_path": batch_path}
    attach_template_metadata(prepared, result, cfg_tokenizer_path=cfg_tokenizer_path)
    if prepared.get("prompt_meta"):
        result["prompt_meta"] = prepared["prompt_meta"]
    if prepared.get("cache_namespace"):
        result["cache_namespace"] = prepared["cache_namespace"]
    return result


def run_coroutine_threadsafe_with_timeout(
    loop: Any,
    coro: Any,
    *,
    timeout: Optional[float],
    request_id: Optional[str] = None,
    abort_fn: Optional[Any] = None,
    logger_prefix: str = "backend",
    timeout_result: Optional[Any] = None,
    timeout_result_fn: Optional[Callable[[], Any]] = None,
) -> Any:
    """Run a coroutine on a loop with timeout handling and optional abort."""

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        if timeout and timeout > 0:
            return future.result(timeout=timeout)
        return future.result()
    except FutureTimeoutError:
        if timeout:
            logger.warning("{} request_id={} timeout after {:.1f}s, aborting", logger_prefix, request_id, timeout)
        else:
            logger.warning("{} request_id={} timeout, aborting", logger_prefix, request_id)
        if abort_fn:
            with contextlib.suppress(Exception):
                result = abort_fn(request_id) if request_id is not None else abort_fn()
                if inspect.iscoroutine(result):
                    asyncio.run_coroutine_threadsafe(result, loop)
        if timeout_result_fn is not None:
            return timeout_result_fn()
        if timeout_result is not None:
            return timeout_result
        raise


def normalize_image_placeholders(prompt: str, image_count: int) -> str:
    """Ensure prompt contains exactly image_count <image> placeholders."""

    marker = "<image>"
    if not prompt:
        return ""
    normalized = re.sub(r"<image\s*\d*>", marker, prompt, flags=re.IGNORECASE)
    if marker not in normalized:
        # Avoid injecting generic placeholders into model-specific templates (e.g., <|image_pad|>).
        return normalized
    parts = normalized.split(marker)
    current = len(parts) - 1
    if image_count <= 0:
        return normalized.replace(marker, "").strip()
    missing = max(0, image_count - current)
    if missing > 0:
        prefix = " ".join([marker] * missing)
        normalized = (prefix + " " + normalized).strip()
        parts = normalized.split(marker)
        current = len(parts) - 1
    if current > image_count:
        normalized = marker.join(parts[: image_count + 1]) + "".join(parts[image_count + 1 :])
    return normalized


def convert_text_like_output(result: Any) -> Any:
    """Extract text/beam-style outputs from vLLM-like responses or passthrough."""

    if isinstance(result, list):
        return [convert_text_like_output(item) for item in result]

    if isinstance(result, dict):
        if result.get("outputs"):
            out = result["outputs"][0]
            if isinstance(out, dict):
                return out.get("text") or out
            return out
        if "text" in result:
            return result["text"]

    if hasattr(result, "outputs"):
        outputs = getattr(result, "outputs") or []
        if outputs:
            first = outputs[0]
            if isinstance(first, dict):
                return first.get("text") or first
            if hasattr(first, "text"):
                return getattr(first, "text")

    return str(result)


def build_sampling_params_base(defaults: Dict[str, Any], runtime: Dict[str, Any], *, max_tokens: int) -> Dict[str, Any]:
    """Merge sampling defaults/runtime and normalize max_new_tokens -> max_tokens."""

    params = dict(defaults or {})
    params.update(runtime or {})
    if "max_tokens" not in params and "max_new_tokens" in params:
        params["max_tokens"] = params.pop("max_new_tokens")
    params.setdefault("max_tokens", max_tokens)
    return params


def _media_dedup_key(src: Any) -> Optional[Any]:
    if isinstance(src, str):
        return src
    try:
        return hash(src)
    except Exception:
        return None


def extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """Extract image sources from message content list."""

    sources: List[Any] = []
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for fragment in content:
            if not isinstance(fragment, dict):
                continue
            frag_type = fragment.get("type")
            if frag_type == "image_url":
                payload = fragment.get("image_url")
            elif frag_type == "image":
                payload = fragment.get("image")
            else:
                continue
            if isinstance(payload, dict):
                value = payload.get("url") or payload.get("data")
            elif payload is None:
                value = fragment.get("url")
            else:
                value = payload
            if value is not None:
                sources.append(value)
    return sources


def has_multimodal_inputs(prepared: Dict[str, Any]) -> bool:
    """Check for multimodal inputs across payload and message content."""

    inputs = prepared.get("inputs") or {}
    sample = prepared.get("sample") or {}
    mm = None
    if isinstance(inputs, dict):
        mm = inputs.get("multi_modal_data")
    if not mm and isinstance(sample, dict):
        mm = sample.get("multi_modal_data")
    if mm:
        return True
    messages = prepared.get("messages") or sample.get("messages") or []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for fragment in content:
                if isinstance(fragment, dict) and fragment.get("type") in {
                    "image",
                    "image_url",
                    "audio",
                    "audio_url",
                    "input_audio",
                    "video",
                    "video_url",
                }:
                    return True
    return False


def extract_audios_from_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """Extract audio sources from message content list."""

    sources: List[Any] = []
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for fragment in content:
            if not isinstance(fragment, dict):
                continue
            frag_type = fragment.get("type")
            payload = None
            if frag_type == "audio_url":
                payload = fragment.get("audio_url")
            elif frag_type == "audio":
                payload = fragment.get("audio")
            elif frag_type == "input_audio":
                payload = fragment.get("input_audio")
            else:
                continue
            if isinstance(payload, dict):
                if frag_type == "input_audio":
                    data = payload.get("data")
                    fmt = payload.get("format") or "wav"
                    if isinstance(data, str):
                        try:
                            binary = base64.b64decode(data)
                        except Exception:
                            continue
                        buf = io.BytesIO(binary)
                        try:
                            buf.name = f"input_audio.{fmt}"
                        except Exception:
                            pass
                        sources.append(buf)
                else:
                    value = payload.get("url")
                    if value is not None:
                        sources.append(value)
            elif payload is None:
                value = fragment.get("url")
                if value is not None:
                    sources.append(value)
            else:
                sources.append(payload)
    return sources


def dedup_media_sources(sources: List[Any]) -> List[Any]:
    """Deduplicate media sources, preserving order."""

    seen = set()
    deduped: List[Any] = []
    for src in sources:
        key = None
        if isinstance(src, str):
            key = src
        else:
            try:
                key = hash(src)
            except Exception:
                key = None
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        deduped.append(src)
    return deduped


def _merge_multimodal_sources(mm_sources: List[Any], message_sources: List[Any]) -> List[Any]:
    """Preserve repeats within each field; dedup only cross-field repeats."""

    merged: List[Any] = []
    seen_message = set()
    for src in message_sources or []:
        key = _media_dedup_key(src)
        merged.append(src)
        if key is not None:
            seen_message.add(key)
    for src in mm_sources or []:
        key = _media_dedup_key(src)
        if key is not None and key in seen_message:
            continue
        merged.append(src)
    return merged


def collect_multimodal_sources(prepared: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Collect raw image/audio sources from inputs/messages/sample without loading."""

    sample = prepared.get("sample") or {}
    raw_inputs = prepared.get("inputs") or sample.get("inputs") or {}
    mm = raw_inputs.get("multi_modal_data") if isinstance(raw_inputs, dict) else None
    if not mm:
        mm = sample.get("multi_modal_data")

    image_sources: List[Any] = []
    audio_sources: List[Any] = []
    if isinstance(mm, dict):
        mm_images = mm.get("image") or mm.get("images")
        if mm_images is not None:
            image_sources.extend(mm_images if isinstance(mm_images, list) else [mm_images])
        audio_raw = mm.get("audio") or mm.get("audios")
        if audio_raw:
            audio_sources.extend(audio_raw if isinstance(audio_raw, list) else [audio_raw])

    messages = prepared.get("messages") or sample.get("messages") or []
    message_images = extract_images_from_messages(messages)
    message_audios = extract_audios_from_messages(messages)

    images = _merge_multimodal_sources(image_sources, message_images)
    audios = _merge_multimodal_sources(audio_sources, message_audios)
    result: Dict[str, List[Any]] = {}
    if images:
        result["image"] = images
    if audios:
        result["audio"] = audios
    return result


def normalize_messages_safe(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten list-based content so chat templates see strings instead of lists."""

    return normalize_messages_for_template(messages)


def check_tokenizer_conflict(prepared: Dict[str, Any], backend_tokenizer_path: Any) -> None:
    """Raise if dataset tokenizer_path conflicts with backend config."""

    dataset_tok = prepared.get("_tokenizer_path") or (prepared.get("sample") or {}).get("_tokenizer_path")
    if dataset_tok and backend_tokenizer_path and str(dataset_tok) != str(backend_tokenizer_path):
        raise ValueError(f"Conflicting tokenizer_path: dataset={dataset_tok} backend={backend_tokenizer_path}")


def maybe_tokenize_messages(
    prepared: Dict[str, Any],
    prompt: str,
    *,
    tokenizer: Any = None,
    processor: Any = None,
    policy: Any = None,
    force_tokenize: bool = False,
    cache_suffix: str = "-chat_template",
    normalize_fn=normalize_messages_safe,
) -> Tuple[str, Any, Dict[str, Any]]:
    """Generate prompt_token_ids using tokenizer/processor if needed.

    Returns: (new_prompt, new_inputs, meta_updates).
    """

    inputs = prepared.get("inputs")
    if isinstance(inputs, dict) and inputs.get("prompt_token_ids"):
        return prompt, inputs, {}
    if not tokenizer and not (processor and hasattr(processor, "apply_chat_template")):
        return prompt, inputs, {}

    if policy and not ChatTemplateMixin.should_render(prepared, policy) and not force_tokenize:
        return prompt, inputs, {}

    if policy and not ChatTemplateMixin.detect_multimodal(prepared) and not force_tokenize:
        return prompt, inputs, {}

    messages = prepared.get("messages")
    if not isinstance(messages, list) or not messages:
        return prompt, inputs, {}

    chat_template_fn = None
    if processor and hasattr(processor, "apply_chat_template"):
        chat_template_fn = processor.apply_chat_template
    elif tokenizer and hasattr(tokenizer, "apply_chat_template"):
        chat_template_fn = tokenizer.apply_chat_template
    if not chat_template_fn:
        return prompt, inputs, {}

    def _strip_non_text(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for m in msgs:
            content = m.get("content")
            if isinstance(content, list):
                texts = []
                for frag in content:
                    if isinstance(frag, dict) and frag.get("type") == "text":
                        texts.append(str(frag.get("text", "")))
                content = [{"type": "text", "text": " ".join(texts)}]
            sanitized.append({"role": m.get("role", "user"), "content": content})
        return sanitized

    try:
        sanitized_messages = messages
        rendered = None
        tokenized = None
        try:
            rendered = chat_template_fn(messages, tokenize=False, add_generation_prompt=True)
            tokenized = chat_template_fn(messages, tokenize=True, add_generation_prompt=True)
        except Exception:
            sanitized_messages = _strip_non_text(messages)
            rendered = chat_template_fn(sanitized_messages, tokenize=False, add_generation_prompt=True)
            tokenized = chat_template_fn(sanitized_messages, tokenize=True, add_generation_prompt=True)

        if isinstance(tokenized, list):
            first = tokenized[0] if tokenized else []
            token_ids = first if isinstance(first, (list, tuple)) else tokenized
        else:
            token_ids = tokenized

        new_prompt = str(rendered) if rendered else prompt
        if not isinstance(inputs, dict):
            inputs = {}
        new_inputs = dict(inputs)
        new_inputs["prompt"] = new_prompt
        if token_ids is not None:
            new_inputs["prompt_token_ids"] = token_ids

        meta = {
            "template_source": "model",
            "rendered_by": "backend",
            "chat_template_mode": "backend",
            "cache_suffix": cache_suffix,
        }
        return new_prompt, new_inputs, meta
    except Exception:
        return prompt, inputs, {}


def graceful_loop_shutdown(loop: Any, loop_thread: Any, model: Any = None) -> None:
    """Best-effort shutdown for background event loop and model cleanup."""

    try:
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if loop_thread and loop_thread.is_alive():
            loop_thread.join(timeout=1.0)
    except Exception:
        logger.warning("backend loop shutdown error", exc_info=True)
    try:
        if model and hasattr(model, "shutdown"):
            model.shutdown()
    except Exception:
        pass
    torch_gpu_cleanup()


def render_prompt_with_template(
    prepared: Dict[str, Any],
    *,
    tokenizer: Any,
    fallback_template: Any,
    policy: Any,
    caps: Any,
    chat_kwargs: Optional[Dict[str, Any]] = None,
    simple_renderer: Optional[Any] = None,
) -> str:
    """Render prompt via chat template, with backend metadata populated."""

    messages = prepared.get("messages") or []
    raw_prompt = prepared.get("prompt") or prepared.get("text") or ""
    chat_kwargs = chat_kwargs or {}
    simple_render = simple_renderer or (lambda msgs: "")

    if not ChatTemplateMixin.should_render(prepared, policy):
        return raw_prompt or simple_render(messages)

    template_source = ChatTemplateMixin.select_template("text", policy, caps)
    template_fn = getattr(tokenizer, "apply_chat_template", None) if tokenizer else None
    fallback_tpl = None if template_source == "model" else fallback_template
    rendered = ChatTemplateMixin.render(
        messages,
        template_fn=template_fn,
        fallback_fn=lambda msgs: simple_render(msgs),
        add_generation_prompt=True,
        chat_template=fallback_tpl,
        **chat_kwargs,
    )
    prepared["chat_template_mode"] = "backend"
    prepared["template_source"] = "model" if template_fn else "fallback"
    prepared["rendered_by"] = "backend"
    return rendered or simple_render(messages) or str(raw_prompt)


def prepare_multi_modal_data(prepared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Collect and PIL-load images/audio paths into a multimodal payload."""

    sources = collect_multimodal_sources(prepared)
    images = load_images(sources.get("image"))
    images = [img for img in images if img is not None]
    audios = [au for au in (sources.get("audio") or []) if au is not None]
    if not images and not audios:
        return None
    result: Dict[str, Any] = {}
    if images:
        result["image"] = images
    if audios:
        result["audio"] = audios
    return result


def load_images(sources) -> List[Any]:
    """Best-effort PIL image loading from paths/base64/objects."""

    if sources is None:
        return []
    if not isinstance(sources, list):
        sources = [sources]
    images: List[Any] = []
    cache: Dict[Any, Any] = {}
    try:  # pragma: no cover - heavy dependency
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed; skipping image loading for multi_modal_data")
        return []

    for src in sources:
        if src is None:
            continue
        key = _media_dedup_key(src)
        if key is not None and key in cache:
            images.append(cache[key])
            continue
        try:
            if isinstance(src, Image.Image):
                loaded = src
            elif isinstance(src, str) and src.startswith("data:"):
                _, b64 = src.split(",", 1)
                binary = base64.b64decode(b64)
                loaded = Image.open(io.BytesIO(binary)).convert("RGB")
            elif isinstance(src, str):
                path = Path(src)
                loaded = Image.open(path).convert("RGB")
            else:
                logger.debug("Unsupported image source type {}; skipping", type(src))
                loaded = None
            if loaded is None:
                continue
            if key is not None:
                cache[key] = loaded
            images.append(loaded)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load image source {}: {}", src, exc)
    return images


def load_multimodal_payload(
    mm: Optional[Dict[str, Any]], processor: Any, *, logger_prefix: str = "backend"
) -> Optional[Dict[str, Any]]:
    """Load multimodal data with processor in a thread pool to avoid loop deadlocks."""

    if not mm:
        return None
    if processor is None:
        logger.warning("{}: processor missing, skipping multi_modal_data load", logger_prefix)
        return mm

    def _safe_load():
        try:
            return load_multimodal_data(processor, mm.get("image"), mm.get("audio"), True)
        except Exception as exc:
            logger.error(
                "{}: load_multimodal_data failed in thread; fallback to raw mm (error={})",
                logger_prefix,
                exc,
            )
            return mm

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_safe_load)
        try:
            return future.result()
        except Exception as exc:
            logger.error("{}: thread pool execution failed: {}", logger_prefix, exc)
            return mm


def simple_render_messages(messages: List[Dict[str, Any]]) -> str:
    """Simple role:content render using stringify_message_content."""

    if not messages:
        return ""
    segments: List[str] = []
    for message in messages:
        role = message.get("role", "user")
        text = stringify_message_content(
            message.get("content"),
            coerce_non_text_fragments=False,
        )
        segments.append(f"{role}: {text}".strip())
    segments.append("assistant:")
    return "\n".join(segments)


def render_with_processor(
    processor: Any, messages: List[Dict[str, Any]], prompt: str, chat_template_kwargs: Optional[Dict[str, Any]] = None,
    *, skip_normalize: bool = False
) -> Optional[str]:
    """Render messages with processor.apply_chat_template, fallback to prompt on failure."""

    apply_template = getattr(processor, "apply_chat_template", None) if processor else None
    if not apply_template:
        return None
    try:
        # NOTE: For multimodal, skip normalize to preserve image_url structure for processor
        # Also sanitize messages to ensure compatibility with HF templates (convert OpenAI image_url to image)
        proc_messages = _sanitize_multimodal_messages(messages) if skip_normalize else normalize_messages_for_template(messages)
        
        rendered = apply_template(
            proc_messages,
            add_generation_prompt=True,
            tokenize=False,
            **(chat_template_kwargs or {}),
        )
        if isinstance(rendered, list):
            rendered = rendered[0]
        return str(rendered) if rendered else prompt
    except Exception as exc:
        logger.debug("processor chat_template failed: {}", exc)
        return None

def _sanitize_multimodal_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sanitize messages for HF Processor chat templates.
    Most local models (Qwen2-VL, LLaVA, etc.) operate on HF standards which use 'image'/'video' type,
    whereas GAGE/OpenAI use 'image_url'/'video_url'. This function bridges the gap.
    
    1. Convert 'image_url' -> 'image', 'video_url' -> 'video', 'audio_url' -> 'audio'.
    2. Flatten nested url dictionaries to simple strings.
    3. Remove keys with None values (sanitize).
    4. Pass through unknown types as-is.
    """
    sanitized = []
    for msg in messages:
        new_msg = {"role": msg.get("role")}
        content = msg.get("content")
        
        if isinstance(content, list):
            new_content = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue
                
                itype = item.get("type")

                # Handle text item
                if itype == "text":
                    new_item = {"type": "text", "text": item.get("text", "")}
                    new_content.append(new_item)
                
                # Handle image types
                elif itype in ["image", "image_url"]:
                    url = item.get("url")
                    if not url and "image_url" in item:
                        val = item["image_url"]
                        url = val.get("url") if isinstance(val, dict) else val
                    if not url:
                        url = item.get("image")
                        
                    if url:
                        new_content.append({"type": "image", "image": url})

                # Handle audio types
                elif itype in ["audio", "audio_url", "input_audio"]:
                    url = item.get("url")
                    if not url and "audio_url" in item:
                        val = item["audio_url"]
                        url = val.get("url") if isinstance(val, dict) else val
                    if not url:
                        url = item.get("audio") or item.get("input_audio")
                        
                    if url:
                        new_content.append({"type": "audio", "audio": url})

                # Handle video types
                elif itype in ["video", "video_url"]:
                    url = item.get("url")
                    if not url and "video_url" in item:
                        val = item["video_url"]
                        url = val.get("url") if isinstance(val, dict) else val
                    if not url:
                        url = item.get("video")
                        
                    if url:
                        new_content.append({"type": "video", "video": url})

                # Fallback: keep other items as-is
                else:
                    new_content.append(item)
            
            new_msg["content"] = new_content
        else:
            new_msg["content"] = content
            
        sanitized.append(new_msg)
    return sanitized


def propagate_metadata_flags(prepared: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """
    Ensure metadata-based flags (chat_template_mode, rendered_by) are propagated from payload.
    This handles cases where normalization misses nested metadata (e.g. sample.metadata).
    """
    meta = prepared.get("sample", {}).get("metadata") or payload.get("metadata") or {}
    if isinstance(meta, dict):
        if "chat_template_mode" in meta and "chat_template_mode" not in prepared:
            prepared["chat_template_mode"] = meta["chat_template_mode"]
        if "rendered_by" in meta and "rendered_by" not in prepared:
            prepared["rendered_by"] = meta["rendered_by"]


def load_hf_tokenizer(config: Dict[str, Any]) -> Any:
    """Best-effort load of HuggingFace tokenizer from config."""
    tok_name = config.get("tokenizer_name") or config.get("tokenizer_path") or config.get("model_path")
    if not tok_name:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            tok_name, trust_remote_code=bool(config.get("trust_remote_code", True))
        )
    except Exception as exc:
        logger.warning("Failed to load tokenizer '{}': {}", tok_name, exc)
        return None


def load_hf_processor(model_id: str, trust_remote_code: bool = True) -> Any:
    """Best-effort load of HuggingFace processor from model_id."""
    if not model_id:
        return None
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        logger.debug("Failed to load processor for '{}': {}", model_id, exc)
        return None


def log_debug_mm_prompt(prompt: str, request_id: str = None) -> None:
    """Log full multimodal prompt if GAGE_EVAL_DEBUG_MM_PROMPT is set."""
    if os.environ.get("GAGE_EVAL_DEBUG_MM_PROMPT"):
        logger.debug("Debug MM Prompt [{}]: {}", request_id or "N/A", prompt)
