"""Runtime role instance borrowed by steps."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from loguru import logger
from gage_eval.observability.trace import ObservabilityTrace


@dataclass(frozen=True)
class HistorySnapshotPolicy:
    """Controls how runtime history is materialized for adapters."""

    copy_mode: str = "shallow"
    window_messages: Optional[int] = None
    preserve_system_messages: bool = True
    fallback_to_deep_on_error: bool = True

    @classmethod
    def parse(cls, value: Any) -> "HistorySnapshotPolicy":
        if isinstance(value, HistorySnapshotPolicy):
            return value
        if not isinstance(value, Mapping):
            return cls()
        copy_mode = str(value.get("copy_mode") or "shallow").strip().lower()
        if copy_mode not in {"shallow", "deep"}:
            raise ValueError(f"Unsupported history copy_mode '{copy_mode}'")
        window_messages = value.get("window_messages")
        if window_messages is not None:
            window_messages = max(0, int(window_messages))
        return cls(
            copy_mode=copy_mode,
            window_messages=window_messages,
            preserve_system_messages=bool(value.get("preserve_system_messages", True)),
            fallback_to_deep_on_error=bool(value.get("fallback_to_deep_on_error", True)),
        )


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

    def snapshot(
        self, policy: Optional[HistorySnapshotPolicy | Mapping[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        resolved_policy = HistorySnapshotPolicy.parse(policy)
        selected = self._select_messages(resolved_policy)
        if resolved_policy.copy_mode == "deep":
            return [self._clone(msg) for msg in selected]
        try:
            return [dict(msg) for msg in selected]
        except Exception:
            if resolved_policy.fallback_to_deep_on_error:
                return [self._clone(msg) for msg in selected]
            raise

    def clear(self) -> None:
        self._messages.clear()

    def is_empty(self) -> bool:
        return not self._messages

    def __bool__(self) -> bool:
        return not self.is_empty()

    @staticmethod
    def _clone(message: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(message)

    def _select_messages(self, policy: HistorySnapshotPolicy) -> List[Dict[str, Any]]:
        if policy.window_messages is None:
            return list(self._messages)
        if policy.window_messages <= 0:
            if not policy.preserve_system_messages:
                return []
            return [
                message
                for message in self._messages
                if str(message.get("role") or "").lower() == "system"
            ]
        if not policy.preserve_system_messages:
            return list(self._messages[-policy.window_messages :])
        non_system: List[Dict[str, Any]] = [
            message
            for message in self._messages
            if str(message.get("role") or "").lower() != "system"
        ]
        retained_non_system = set(map(id, non_system[-policy.window_messages :]))
        selected: List[Dict[str, Any]] = []
        for message in self._messages:
            role = str(message.get("role") or "").lower()
            if role == "system" or id(message) in retained_non_system:
                selected.append(message)
        return selected


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

    def history_snapshot(
        self, policy: Optional[HistorySnapshotPolicy | Mapping[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        return self._history.snapshot(policy)

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------
    def invoke(self, payload: Dict[str, Any], trace: ObservabilityTrace) -> Dict[str, Any]:
        prepared_payload = dict(payload or {})
        self._apply_history_override(prepared_payload.pop("history_override", None))
        self._ingest_history_delta(prepared_payload.pop("history_delta", None))

        history_policy = self._resolve_history_policy()
        history_view = self.history_snapshot(policy=history_policy)
        if history_view:
            # NOTE: Adapters/backends must treat these views as read-only. Use
            # copy_mode=deep when a role needs to mutate nested message content.
            prepared_payload.setdefault("history", history_view)
            prepared_payload.setdefault("messages", history_view)

        logger.debug(
            "Role invoke adapter={} runtime={} history_tokens={} history_copy_mode={} history_window={}",
            self.adapter_id,
            bool(self._runtime),
            len(history_view),
            history_policy.copy_mode,
            history_policy.window_messages,
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

    def _resolve_history_policy(self) -> HistorySnapshotPolicy:
        params = getattr(self._adapter, "params", None)
        if isinstance(params, Mapping):
            return HistorySnapshotPolicy.parse(params.get("history_policy"))
        return HistorySnapshotPolicy()
