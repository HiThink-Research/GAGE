"""HTTP-based agent backend implementation."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from gage_eval.role.agent.backends.base import AgentBackend, normalize_agent_output


class HttpBackend(AgentBackend):
    """Agent backend that sends requests to a remote HTTP service."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._endpoint = config.get("endpoint")
        if not self._endpoint:
            raise ValueError("endpoint is required for HttpBackend")
        self._timeout_s = config.get("timeout_s", 60)
        self._schema = config.get("schema") or "raw"
        self._headers = dict(config.get("headers") or {})
        self._normalize_messages = bool(config.get("normalize_messages", False))

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._schema == "openai_chat" and self._normalize_messages:
            payload["messages"] = _normalize_openai_messages(payload.get("messages"))
        response = requests.post(
            self._endpoint,
            json=payload,
            headers=self._headers,
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        data = _safe_json(response)
        if self._schema == "openai_chat":
            parsed = _parse_openai_chat(data)
            return normalize_agent_output(parsed)
        return normalize_agent_output(data)


def _safe_json(response: requests.Response) -> Dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {"answer": response.text}


def _parse_openai_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "choices" not in payload:
        return payload
    choices = payload.get("choices") or []
    if not choices:
        return payload
    message = (choices[0] or {}).get("message") or {}
    tool_calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    parsed: Dict[str, Any] = {"answer": content}
    if tool_calls:
        parsed["tool_calls"] = tool_calls
    if "usage" in payload:
        parsed["usage"] = payload["usage"]
    return parsed


def _normalize_openai_messages(messages: Any) -> list[Dict[str, Any]]:
    normalized: list[Dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            normalized.append({"role": "user", "content": str(message)})
            continue
        entry = dict(message)
        content = entry.get("content")
        if content is None:
            entry["content"] = ""
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part:
                        parts.append(str(part.get("text") or ""))
                    else:
                        parts.append(json.dumps(part, ensure_ascii=True))
                else:
                    parts.append(str(part))
            entry["content"] = "".join(parts)
        elif not isinstance(content, str):
            entry["content"] = str(content)
        normalized.append(entry)
    return normalized
