"""Human-in-the-loop gateway for host-only tool calls."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class HumanRequest:
    """Represents a human input request."""

    question: str
    metadata: Dict[str, Any]


class HumanGateway:
    """Gateway that retrieves human input for host-only tool calls.

    Args:
        input_provider: Callable that receives a :class:`HumanRequest` and returns the answer.
    """

    def __init__(self, input_provider: Optional[Callable[[HumanRequest], str]] = None) -> None:
        self._input_provider = input_provider or _default_provider

    def request(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Request an answer from the human input provider.

        Args:
            question: Question or prompt to show to the human.
            metadata: Optional metadata attached to the request.

        Returns:
            The human response string.
        """

        payload = HumanRequest(question=question, metadata=metadata or {})
        return self._input_provider(payload)


def _default_provider(request: HumanRequest) -> str:
    raise RuntimeError("HumanGateway input provider is not configured")


def build_default_human_gateway() -> Optional[HumanGateway]:
    """Build a default HumanGateway when CLI stdin is enabled.

    Returns:
        A HumanGateway instance if stdin mode is enabled, otherwise None.
    """

    if os.environ.get("GAGE_EVAL_HUMAN_INPUT") != "stdin":
        return None
    return HumanGateway(input_provider=_stdin_provider)


def _stdin_provider(request: HumanRequest) -> str:
    """Read human input from stdin for CLI workflows."""

    prompt = request.question.strip() or "Input required"
    sys.stdout.write(f"{prompt}\n> ")
    sys.stdout.flush()
    return sys.stdin.readline().strip()
