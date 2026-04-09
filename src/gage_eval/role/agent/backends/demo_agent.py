from __future__ import annotations

from typing import Any


class RuntimeSmokeAgent:
    """Provides a tiny deterministic agent surface for runtime smoke configs."""

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a deterministic answer for config/runtime smoke checks."""

        return {
            "answer": "done",
            "agent_trace": [],
        }

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Expose the standard agent-backend entrypoint used by AgentLoop."""

        return self.run(payload)
