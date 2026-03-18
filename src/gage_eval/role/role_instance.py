"""Runtime role instance borrowed by steps."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.utils.messages import clone_json_like


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
        return clone_json_like(message)


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
    def invoke(self, payload: Optional[Dict[str, Any]], trace: ObservabilityTrace) -> Dict[str, Any]:
        """Invoke the role adapter with sample-scoped history wiring."""

        base_payload = payload or {}

        # STEP 1: Apply request-scoped history mutations before building the adapter payload.
        history_override = base_payload.get("history_override")
        history_delta = base_payload.get("history_delta")
        self._apply_history_override(history_override)
        self._ingest_history_delta(history_delta)

        # STEP 2: Materialize only the payload view required by downstream adapters.
        prepared_payload, history_tokens = self._build_invoke_payload(
            base_payload,
            has_history_override=history_override is not None,
            has_history_delta=history_delta is not None,
        )

        logger.debug(
            "Role invoke adapter={} runtime={} history_tokens={}",
            self.adapter_id,
            bool(self._runtime),
            history_tokens,
        )

        # STEP 3: Delegate execution to the runtime or adapter.
        if self._runtime:
            result = self._runtime.execute(prepared_payload, trace)
        else:
            state = self._adapter.clone_for_sample()
            result = self._adapter.invoke(prepared_payload, state)

        # STEP 4: Persist any response-side history delta back into the session history.
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

    def _build_invoke_payload(
        self,
        base_payload: Dict[str, Any],
        *,
        has_history_override: bool,
        has_history_delta: bool,
    ) -> tuple[Dict[str, Any], int]:
        has_history = "history" in base_payload
        has_messages = "messages" in base_payload
        history_view = base_payload.get("history") if has_history else None
        messages_view = base_payload.get("messages") if has_messages else None
        inject_history = False
        inject_messages = False

        if not has_history and not has_messages:
            if not self._history.is_empty():
                history_view = self.history_snapshot()
                messages_view = history_view
                inject_history = True
                inject_messages = True
        elif not has_history and messages_view is not None:
            history_view = messages_view
            inject_history = True
        elif not has_messages and history_view is not None:
            messages_view = history_view
            inject_messages = True

        needs_copy = has_history_override or has_history_delta or inject_history or inject_messages
        prepared_payload = dict(base_payload) if needs_copy else base_payload
        if needs_copy:
            prepared_payload.pop("history_override", None)
            prepared_payload.pop("history_delta", None)
            if inject_history:
                prepared_payload["history"] = history_view
            if inject_messages:
                prepared_payload["messages"] = messages_view

        history_tokens = self._history_length(prepared_payload.get("messages"))
        if history_tokens == 0:
            history_tokens = self._history_length(prepared_payload.get("history"))
        return prepared_payload, history_tokens

    @staticmethod
    def _history_length(messages: Any) -> int:
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)):
            return len(messages)
        return 0
