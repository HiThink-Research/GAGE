"""Runtime role instance borrowed by steps."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger
from gage_eval.observability.trace import ObservabilityTrace


class ConversationHistory:
    """Lightweight helper to maintain turn-taking history for a role."""

    def __init__(self, initial_messages: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        self._messages: List[Dict[str, Any]] = []
        if initial_messages:
            self.replace(initial_messages)

    def replace(self, messages: Sequence[Dict[str, Any]]) -> None:
        self._messages = [self._clone(msg) for msg in messages if isinstance(msg, dict)]

    def extend(self, messages: Optional[Sequence[Dict[str, Any]]]) -> None:
        if not messages:
            return
        for message in messages:
            if isinstance(message, dict):
                self._messages.append(self._clone(message))

    def append(self, message: Dict[str, Any]) -> None:
        if isinstance(message, dict):
            self._messages.append(self._clone(message))

    def snapshot(self) -> List[Dict[str, Any]]:
        return [self._clone(msg) for msg in self._messages]

    def clear(self) -> None:
        self._messages.clear()

    def is_empty(self) -> bool:
        return not self._messages

    def __bool__(self) -> bool:
        return not self.is_empty()

    @staticmethod
    def _clone(message: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(message)


class Role:
    """Encapsulates an adapter (and optional runtime) for a single invocation."""

    def __init__(
        self,
        adapter_id: str,
        adapter,
        runtime: Optional[Any] = None,
        *,
        history: Optional[ConversationHistory] = None,
    ) -> None:
        self.adapter_id = adapter_id
        self._adapter = adapter
        self._runtime = runtime
        self._history = history or ConversationHistory()

    # ------------------------------------------------------------------
    # Conversation history helpers
    # ------------------------------------------------------------------
    def attach_history(self, history: Optional[ConversationHistory]) -> None:
        self._history = history or ConversationHistory()

    def prime_history(self, messages: Optional[Sequence[Dict[str, Any]]], *, overwrite: bool = False) -> None:
        if not messages:
            return
        if overwrite or self._history.is_empty():
            self._history.replace(messages)

    def extend_history(self, messages: Optional[Sequence[Dict[str, Any]]]) -> None:
        self._history.extend(messages)

    def add_message(self, message: Dict[str, Any]) -> None:
        self._history.append(message)

    def reset_history(self) -> None:
        self._history.clear()

    def history_snapshot(self) -> List[Dict[str, Any]]:
        return self._history.snapshot()

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------
    def invoke(self, payload: Dict[str, Any], trace: ObservabilityTrace) -> Dict[str, Any]:
        prepared_payload = dict(payload or {})
        self._apply_history_override(prepared_payload.pop("history_override", None))
        self._ingest_history_delta(prepared_payload.pop("history_delta", None))

        history_view = self.history_snapshot()
        if history_view:
            prepared_payload.setdefault("history", history_view)
            prepared_payload.setdefault("messages", history_view)

        logger.debug(
            "Role invoke adapter={} runtime={} history_tokens={}",
            self.adapter_id,
            bool(self._runtime),
            len(history_view),
        )
        if self._runtime:
            result = self._runtime.execute(prepared_payload, trace)
        else:
            state = self._adapter.clone_for_sample()
            result = self._adapter.invoke(prepared_payload, state)

        self._ingest_history_delta(result.get("history_delta"))
        logger.trace("Role invoke adapter={} produced keys={}", self.adapter_id, list(result.keys()))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_history_override(self, override: Union[Dict[str, Any], Sequence[Dict[str, Any]], None]) -> None:
        if not override:
            return
        if isinstance(override, dict):
            self._history.replace([override])
            return
        self._history.replace(override)

    def _ingest_history_delta(self, delta: Union[Dict[str, Any], Sequence[Dict[str, Any]], None]) -> None:
        if not delta:
            return
        if isinstance(delta, dict):
            self._history.append(delta)
            return
        if isinstance(delta, Sequence):
            self._history.extend(delta)  # type: ignore[arg-type]
