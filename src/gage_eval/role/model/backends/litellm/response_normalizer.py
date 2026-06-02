"""Normalize LiteLLM responses into the GAGE backend result contract."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Dict, Iterable

from loguru import logger

from gage_eval.role.model.backends.litellm.policies import ToolCallPolicy
from gage_eval.role.model.reasoning import extract_reasoning_content, strip_thinking_tags


class LiteLLMResponseNormalizer:
    """Normalizes LiteLLM chat completions and stream chunks."""

    _MAX_SUMMARY_DEPTH = 5
    _MAX_SUMMARY_ITEMS = 20
    _MAX_SUMMARY_LIST_ITEMS = 10
    _MAX_SUMMARY_STRING_CHARS = 128
    _SECRET_KEY_NAMES = {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "access_token",
        "refresh_token",
        "client_secret",
    }

    def normalize(self, raw: Any, request_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Normalizes a non-streaming LiteLLM completion response.

        Args:
            raw: LiteLLM response object or JSON-like dictionary.
            request_context: Optional request metadata used for thinking mismatch
                checks and observation summaries.

        Returns:
            A GAGE result dictionary containing answer, raw_response, usage,
            reasoning_content, reasoning_details, tool_calls, finish_reason,
            and metadata.
        """

        context = dict(request_context or {})
        raw_response = self.to_jsonable(raw)
        choice = self._first_choice(raw)
        message = self._choice_message(choice)
        answer = self._content_to_text(self._field(message, "content"))
        reasoning_details = self.to_jsonable(self._field(message, "reasoning_details"))
        reasoning, reasoning_source = self._extract_reasoning(message)
        answer, tag_reasoning = self._strip_answer_thinking_tags(answer)
        if tag_reasoning and not reasoning:
            reasoning = tag_reasoning
            reasoning_source = "think_tags"

        tool_calls = ToolCallPolicy.normalize_tool_calls(self._field(message, "tool_calls"))
        finish_reason = self._field(choice, "finish_reason")
        usage = self.to_jsonable(self._field(raw, "usage"))

        metadata: Dict[str, Any] = {}
        self._preserve_thinking_metadata(metadata, message)
        self._apply_reasoning_metadata(
            metadata,
            reasoning=reasoning,
            reasoning_source=reasoning_source,
            reasoning_details=reasoning_details,
            usage=usage,
            raw_response=raw_response,
            tool_calls=tool_calls,
            request_context=context,
        )
        self._apply_answer_empty_diagnosis(
            metadata,
            answer=answer,
            reasoning=reasoning,
            finish_reason=finish_reason,
        )

        return {
            "answer": answer,
            "raw_response": raw_response,
            "usage": usage,
            "reasoning_content": reasoning,
            "reasoning_details": reasoning_details,
            "tool_calls": tool_calls,
            "finish_reason": str(finish_reason) if finish_reason is not None else None,
            "metadata": metadata,
        }

    def normalize_stream(self, chunks: Iterable[Any], request_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Normalizes sync streaming LiteLLM chunks into one GAGE result."""

        context = dict(request_context or {})
        collected_chunks: list[Any] = []
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_source: str | None = None
        reasoning_details: list[Any] = []
        tool_call_state: Dict[Any, Dict[str, Any]] = {}
        usage: Any = None
        finish_reason: Any = None
        thinking_blocks: list[Any] = []

        for chunk in chunks:
            collected_chunks.append(chunk)
            chunk_usage = self._field(chunk, "usage")
            if chunk_usage is not None:
                usage = chunk_usage

            choice = self._first_choice(chunk)
            if choice is None:
                continue
            chunk_finish_reason = self._field(choice, "finish_reason")
            if chunk_finish_reason is not None:
                finish_reason = chunk_finish_reason

            delta = self._choice_message(choice)
            content = self._content_to_text(self._field(delta, "content"))
            if content:
                answer_parts.append(content)

            reasoning, source = self._extract_reasoning(delta)
            if reasoning:
                reasoning_parts.append(reasoning)
                reasoning_source = reasoning_source or source

            details_delta = self._field(delta, "reasoning_details")
            if details_delta is not None:
                reasoning_details.extend(self._as_list(self.to_jsonable(details_delta)))

            block = self._field(delta, "thinking_blocks")
            if block is not None:
                if isinstance(block, list):
                    thinking_blocks.extend(block)
                else:
                    thinking_blocks.append(block)

            for tool_call_delta in self._as_list(self._field(delta, "tool_calls")):
                ToolCallPolicy.merge_tool_call_delta(tool_call_state, tool_call_delta)

        answer = "".join(answer_parts)
        reasoning = "".join(reasoning_parts) if reasoning_parts else None
        answer, tag_reasoning = self._strip_answer_thinking_tags(answer)
        if tag_reasoning and not reasoning:
            reasoning = tag_reasoning
            reasoning_source = "think_tags"

        tool_calls = ToolCallPolicy.normalize_tool_calls(list(tool_call_state.values()))
        raw_response = self.to_jsonable(collected_chunks)
        usage = self.to_jsonable(usage)
        normalized_reasoning_details = reasoning_details or None
        metadata: Dict[str, Any] = {}
        if thinking_blocks:
            metadata["thinking_blocks"] = self.to_jsonable(thinking_blocks)
        self._apply_reasoning_metadata(
            metadata,
            reasoning=reasoning,
            reasoning_source=reasoning_source,
            reasoning_details=normalized_reasoning_details,
            usage=usage,
            raw_response=raw_response,
            tool_calls=tool_calls,
            request_context=context,
        )
        self._apply_answer_empty_diagnosis(
            metadata,
            answer=answer,
            reasoning=reasoning,
            finish_reason=finish_reason,
        )

        return {
            "answer": answer,
            "raw_response": raw_response,
            "usage": usage,
            "reasoning_content": reasoning,
            "reasoning_details": normalized_reasoning_details,
            "tool_calls": tool_calls,
            "finish_reason": str(finish_reason) if finish_reason is not None else None,
            "metadata": metadata,
        }

    def build_observation_summary(
        self,
        request_kwargs: Dict[str, Any],
        response_or_result: Any,
        request_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Builds a compact, non-sensitive request/response summary."""

        context = dict(request_context or {})
        result = response_or_result if isinstance(response_or_result, dict) else self.normalize(response_or_result, context)
        metadata_raw = result.get("metadata")
        metadata: Dict[str, Any] = metadata_raw if isinstance(metadata_raw, dict) else {}
        api_base = context.get("api_base") or request_kwargs.get("api_base") or request_kwargs.get("base_url")
        usage = result.get("usage")
        latency_ms = context.get("latency_ms", result.get("latency_ms"))
        tool_calls_raw = result.get("tool_calls")
        tool_calls = tool_calls_raw if isinstance(tool_calls_raw, list) else []
        answer = result.get("answer") or ""
        response_model = self._response_model(result.get("raw_response"))

        summary: Dict[str, Any] = {
            "provider": context.get("provider") or request_kwargs.get("custom_llm_provider"),
            "model": context.get("model") or request_kwargs.get("model"),
            "response_model": response_model,
            "api_base_hash": self._hash_text(api_base),
            "route_mode": context.get("route_mode"),
            "topology": self._summarize_topology(context.get("topology")),
            "resolved_thinking_mode": context.get("resolved_thinking_mode")
            or context.get("thinking_mode")
            or metadata.get("thinking_mode"),
            "effective_thinking_mode": metadata.get("thinking_effective"),
            "modality_counts": self._count_modalities(request_kwargs.get("messages")),
            "tool_count": len(self._as_list(request_kwargs.get("tools"))),
            "tool_call_count": len(tool_calls),
            "has_tool_calls": bool(tool_calls),
            "parallel_tool_calls": request_kwargs.get("parallel_tool_calls"),
            "reasoning_effort": request_kwargs.get("reasoning_effort") or context.get("reasoning_effort"),
            "retry_owner": context.get("retry_owner"),
            "outer_retry_attempts": context.get("outer_retry_attempts"),
            "router_num_retries": context.get("router_num_retries"),
            "thinking_inherited_from_server_default": context.get("thinking_inherited_from_server_default"),
            "finish_reason": result.get("finish_reason"),
            "answer_empty_reason": metadata.get("answer_empty_reason"),
            "usage": self.to_jsonable(usage),
            "latency_ms": round(float(latency_ms), 3) if latency_ms is not None else None,
            "answer_chars": len(str(answer)),
            "stream": context.get("stream"),
        }
        compact_summary = {key: value for key, value in summary.items() if value is not None}
        return self._sanitize_observation_summary(compact_summary)

    @classmethod
    def to_jsonable(cls, obj: Any) -> Any:
        """Converts LiteLLM response objects into JSON-safe values."""

        try:
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, dict):
                return {key: cls.to_jsonable(value) for key, value in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [cls.to_jsonable(value) for value in obj]
            if hasattr(obj, "model_dump"):
                return cls.to_jsonable(obj.model_dump())
            if hasattr(obj, "__dict__"):
                return {key: cls.to_jsonable(value) for key, value in vars(obj).items() if not key.startswith("_")}
            return str(obj)
        except Exception:  # pragma: no cover - defensive around third-party response objects
            return str(obj)

    def _extract_reasoning(self, message: Any) -> tuple[str | None, str | None]:
        reasoning = extract_reasoning_content(message)
        if reasoning:
            return reasoning, "reasoning_content"

        reasoning_details = self._reasoning_details_text(self._field(message, "reasoning_details"))
        if reasoning_details:
            return reasoning_details, "reasoning_details"
        return None, None

    def _preserve_thinking_metadata(self, metadata: Dict[str, Any], message: Any) -> None:
        thinking_blocks = self._field(message, "thinking_blocks")
        if thinking_blocks is not None:
            metadata["thinking_blocks"] = self.to_jsonable(thinking_blocks)

    def _apply_reasoning_metadata(
        self,
        metadata: Dict[str, Any],
        *,
        reasoning: str | None,
        reasoning_source: str | None,
        reasoning_details: Any,
        usage: Any,
        raw_response: Any,
        tool_calls: list[Dict[str, Any]],
        request_context: Dict[str, Any],
    ) -> None:
        observed_source = reasoning_source or self._observed_reasoning_source(
            reasoning_details=reasoning_details,
            usage=usage,
            raw_response=raw_response,
        )
        if reasoning_source:
            metadata["reasoning_source"] = reasoning_source
        elif observed_source:
            metadata["reasoning_source"] = observed_source

        expected_mode = self._expected_thinking_mode(request_context)
        if expected_mode:
            metadata["thinking_mode"] = expected_mode

        observed_mode = "enabled" if reasoning or observed_source else "disabled"
        metadata["thinking_effective"] = observed_mode

        if tool_calls and (expected_mode == "enabled" or observed_mode == "enabled") and "thinking_blocks" not in metadata:
            metadata["thinking_blocks_missing"] = True

        mismatch = self._thinking_mismatch(expected_mode, observed_mode, observed_source)
        if not mismatch:
            return

        metadata["thinking_mismatch"] = mismatch
        on_mismatch = self._policy_value(request_context.get("thinking_policy"), "on_mismatch", "warn_and_continue")
        message = f"response_mismatch: expected thinking={expected_mode}, observed thinking={observed_mode}"
        if on_mismatch == "fail_fast":
            raise ValueError(message)
        if on_mismatch == "warn_and_continue":
            logger.warning(message)

    @staticmethod
    def _apply_answer_empty_diagnosis(
        metadata: Dict[str, Any],
        *,
        answer: str,
        reasoning: str | None,
        finish_reason: Any,
    ) -> None:
        if str(answer or "").strip():
            return
        if str(finish_reason or "").lower() != "length":
            return
        if not str(reasoning or "").strip():
            return
        metadata.setdefault("answer_empty_reason", "reasoning_exhausted_completion_budget")

    @classmethod
    def _response_model(cls, raw_response: Any) -> str | None:
        if isinstance(raw_response, dict):
            value = raw_response.get("model")
            return str(value) if value else None
        if isinstance(raw_response, list):
            for chunk in reversed(raw_response):
                value = cls._response_model(chunk)
                if value:
                    return value
        return None

    @staticmethod
    def _thinking_mismatch(
        expected_mode: str | None,
        observed_mode: str,
        reasoning_source: str | None,
    ) -> Dict[str, Any] | None:
        if expected_mode not in {"enabled", "disabled"}:
            return None
        if expected_mode == observed_mode:
            return None
        mismatch: Dict[str, Any] = {
            "expected": expected_mode,
            "observed": observed_mode,
            "error_type": "response_mismatch",
        }
        if reasoning_source:
            mismatch["reasoning_source"] = reasoning_source
        return mismatch

    @classmethod
    def _expected_thinking_mode(cls, request_context: Dict[str, Any]) -> str | None:
        for key in ("expected_thinking_mode", "resolved_thinking_mode", "thinking_mode"):
            value = request_context.get(key)
            normalized = cls._normalize_mode(value)
            if normalized:
                return normalized
        return None

    def _observed_reasoning_source(
        self,
        *,
        reasoning_details: Any,
        usage: Any,
        raw_response: Any,
    ) -> str | None:
        if self._has_reasoning_details(reasoning_details):
            return "reasoning_details"
        if self._has_positive_reasoning_tokens(usage):
            return "usage_reasoning_tokens"
        if self._has_positive_reasoning_tokens(raw_response):
            return "raw_reasoning_tokens"
        return None

    @classmethod
    def _has_reasoning_details(cls, reasoning_details: Any) -> bool:
        if reasoning_details is None:
            return False
        if isinstance(reasoning_details, (list, tuple, dict, str)):
            return bool(reasoning_details)
        return True

    @classmethod
    def _has_positive_reasoning_tokens(cls, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "reasoning_tokens" and cls._is_positive_number(item):
                    return True
                if cls._has_positive_reasoning_tokens(item):
                    return True
            return False
        if isinstance(value, (list, tuple)):
            return any(cls._has_positive_reasoning_tokens(item) for item in value)
        return False

    @staticmethod
    def _is_positive_number(value: Any) -> bool:
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _normalize_mode(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower().replace("-", "_")
        if normalized in {"enabled", "enable", "true", "on"}:
            return "enabled"
        if normalized in {"disabled", "disable", "false", "off"}:
            return "disabled"
        if normalized in {"auto", "default", "unknown"}:
            return normalized
        return None

    @staticmethod
    def _policy_value(policy: Any, key: str, default: Any = None) -> Any:
        if policy is None:
            return default
        if isinstance(policy, dict):
            return policy.get(key, default)
        return getattr(policy, key, default)

    @classmethod
    def _first_choice(cls, response: Any) -> Any:
        choices = cls._field(response, "choices")
        if not choices:
            return None
        try:
            return choices[0]
        except (KeyError, TypeError, IndexError):
            return None

    @classmethod
    def _choice_message(cls, choice: Any) -> Any:
        return cls._field(choice, "message") or cls._field(choice, "delta") or {}

    @classmethod
    def _content_to_text(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(cls._content_piece(part) for part in content)
        return str(content)

    @classmethod
    def _content_piece(cls, part: Any) -> str:
        if isinstance(part, str):
            return part
        text = cls._field(part, "text")
        if text is not None:
            return str(text)
        content = cls._field(part, "content")
        if content is not None and cls._field(part, "type") in {"text", "input_text"}:
            return str(content)
        return ""

    @staticmethod
    def _strip_answer_thinking_tags(answer: str) -> tuple[str, str | None]:
        if not answer:
            return answer, None
        return strip_thinking_tags(answer)

    @classmethod
    def _reasoning_details_text(cls, reasoning_details: Any) -> str | None:
        if reasoning_details is None:
            return None
        parts = cls._readable_reasoning_parts(reasoning_details)
        if not parts:
            return None
        return "\n".join(part for part in parts if part)

    @classmethod
    def _readable_reasoning_parts(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            parts: list[str] = []
            for item in value:
                parts.extend(cls._readable_reasoning_parts(item))
            return parts
        for key in ("text", "content", "reasoning", "summary"):
            item = cls._field(value, key)
            if isinstance(item, str):
                return [item]
            if isinstance(item, (list, tuple)):
                return cls._readable_reasoning_parts(item)
        return []

    @staticmethod
    def _hash_text(value: Any) -> str | None:
        if not value:
            return None
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _count_modalities(cls, messages: Any) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for message in cls._as_list(messages):
            content = cls._field(message, "content")
            if isinstance(content, str):
                counts["text"] += 1
                continue
            for block in cls._as_list(content):
                block_type = cls._field(block, "type")
                if block_type in {"text", "input_text"}:
                    counts["text"] += 1
                elif block_type in {"image", "image_url"}:
                    counts["image"] += 1
                elif block_type in {"audio", "audio_url", "input_audio"}:
                    counts["audio"] += 1
                elif block_type in {"video", "video_url"}:
                    counts["video"] += 1
                elif block_type in {"file", "file_url"}:
                    counts["file"] += 1
                elif block_type == "document":
                    counts["document"] += 1
                elif block_type:
                    counts["other"] += 1
        return dict(counts)

    @classmethod
    def _summarize_topology(cls, topology: Any) -> Any:
        if topology is None:
            return None
        if not isinstance(topology, dict):
            return cls._safe_scalar(topology)

        summary: Dict[str, Any] = {}
        for key, value in topology.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = cls._safe_scalar(value)
            elif isinstance(value, dict):
                summary[key] = {"keys": sorted(str(item) for item in value.keys())}
            elif isinstance(value, list):
                summary[key] = {"count": len(value)}
            else:
                summary[key] = type(value).__name__
        return summary

    @staticmethod
    def _safe_scalar(value: Any) -> Any:
        if isinstance(value, str) and len(value) > 128:
            return {"chars": len(value), "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]}
        return value

    @classmethod
    def _sanitize_observation_summary(cls, summary: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = cls._sanitize_summary_value(summary)
        return sanitized if isinstance(sanitized, dict) else {}

    @classmethod
    def _sanitize_summary_value(cls, value: Any, *, key: str | None = None, depth: int = 0) -> Any:
        if key is not None and cls._is_secret_key(key):
            return "[REDACTED]"
        if isinstance(value, str):
            return cls._sanitize_summary_string(value)
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if depth >= cls._MAX_SUMMARY_DEPTH:
            return cls._container_summary(value)
        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            items = list(value.items())
            for item_key, item_value in items[: cls._MAX_SUMMARY_ITEMS]:
                sanitized[str(item_key)] = cls._sanitize_summary_value(
                    item_value,
                    key=str(item_key),
                    depth=depth + 1,
                )
            if len(items) > cls._MAX_SUMMARY_ITEMS:
                sanitized["_truncated_items"] = len(items) - cls._MAX_SUMMARY_ITEMS
            return sanitized
        if isinstance(value, (list, tuple)):
            sanitized_list = [
                cls._sanitize_summary_value(item, depth=depth + 1)
                for item in list(value)[: cls._MAX_SUMMARY_LIST_ITEMS]
            ]
            if len(value) > cls._MAX_SUMMARY_LIST_ITEMS:
                sanitized_list.append({"_truncated_items": len(value) - cls._MAX_SUMMARY_LIST_ITEMS})
            return sanitized_list
        return str(value)

    @classmethod
    def _sanitize_summary_string(cls, value: str) -> Any:
        if len(value) <= cls._MAX_SUMMARY_STRING_CHARS and not value.startswith("data:"):
            return value
        kind = "data_url" if value.startswith("data:") else "base64" if cls._looks_like_base64(value) else "truncated"
        return {
            "kind": kind,
            "chars": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
        }

    @classmethod
    def _looks_like_base64(cls, value: str) -> bool:
        if len(value) <= cls._MAX_SUMMARY_STRING_CHARS:
            return False
        sample = value[: min(len(value), 512)].strip()
        if not sample:
            return False
        base64_chars = sum(1 for char in sample if char.isalnum() or char in {"+", "/", "="})
        return base64_chars / len(sample) > 0.95

    @classmethod
    def _is_secret_key(cls, key: str) -> bool:
        normalized = key.strip().lower().replace("-", "_")
        return normalized in cls._SECRET_KEY_NAMES or normalized.endswith("_api_key")

    @staticmethod
    def _container_summary(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return {"kind": "dict", "keys": sorted(str(key) for key in value.keys())[:10]}
        if isinstance(value, (list, tuple)):
            return {"kind": "list", "items": len(value)}
        return {"kind": type(value).__name__}

    @staticmethod
    def _field(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]
