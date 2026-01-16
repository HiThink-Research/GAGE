"""Agent backend base definitions and normalization helpers."""

from __future__ import annotations

from typing import Any, Dict


class AgentBackend:
    """Base interface for agent execution backends."""

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - interface
        """Execute the agent and return the normalized output."""

        raise NotImplementedError

    def shutdown(self) -> None:  # pragma: no cover - optional
        """Release backend resources when the pipeline shuts down."""

        return None


def normalize_agent_output(output: Any) -> Dict[str, Any]:
    """Normalize agent backend outputs to the standard envelope."""

    if isinstance(output, dict):
        normalized = dict(output)
        normalized.setdefault("agent_trace", [])
        if "answer" not in normalized:
            normalized["answer"] = normalized.get("content", "")
        normalized.setdefault("answer", "")
        return normalized
    if isinstance(output, str):
        return {"answer": output, "agent_trace": []}
    if output is None:
        return {"answer": "", "agent_trace": []}
    return {"answer": str(output), "agent_trace": []}
