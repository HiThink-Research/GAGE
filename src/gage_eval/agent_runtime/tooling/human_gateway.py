"""Runtime-owned human-in-the-loop gateway for host-only tool calls."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class HumanRequest:
    """Represents a human input request."""

    question: str
    metadata: dict[str, Any]


class HumanGateway:
    """Gateway that retrieves human input for runtime tool calls."""

    def __init__(self, input_provider: Callable[[HumanRequest], str] | None = None) -> None:
        self._input_provider = input_provider or _default_provider

    def request(self, question: str, metadata: dict[str, Any] | None = None) -> str:
        payload = HumanRequest(question=question, metadata=dict(metadata or {}))
        return self._input_provider(payload)


def build_default_human_gateway() -> HumanGateway | None:
    """Build a stdin-backed gateway when CLI stdin mode is enabled."""

    if os.environ.get("GAGE_EVAL_HUMAN_INPUT") != "stdin":
        return None
    return HumanGateway(input_provider=_stdin_provider)


def _default_provider(request: HumanRequest) -> str:
    raise RuntimeError("HumanGateway input provider is not configured")


def _stdin_provider(request: HumanRequest) -> str:
    prompt = request.question.strip() or "Input required"
    sys.stdout.write(f"{prompt}\n> ")
    sys.stdout.flush()
    return sys.stdin.readline().strip()
