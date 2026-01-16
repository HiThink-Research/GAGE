"""MCP-based agent backend implementation (placeholder)."""

from __future__ import annotations

from typing import Any, Dict

import requests

from gage_eval.role.agent.backends.base import AgentBackend, normalize_agent_output


class McpBackend(AgentBackend):
    """Agent backend that speaks MCP over HTTP transports."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._endpoint = config.get("endpoint")
        if not self._endpoint:
            raise ValueError("endpoint is required for McpBackend")
        self._timeout_s = config.get("timeout_s", 60)
        self._headers = dict(config.get("headers") or {})
        self._transport = config.get("transport") or "http_sse"

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            self._endpoint,
            json=payload,
            headers=self._headers,
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        try:
            data = response.json()
        except Exception:
            data = {"answer": response.text}
        data.setdefault("transport", self._transport)
        return normalize_agent_output(data)
