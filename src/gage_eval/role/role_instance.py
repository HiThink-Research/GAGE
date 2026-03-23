"""Runtime role instance borrowed by steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

from loguru import logger
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.utils.messages import clone_json_like

if TYPE_CHECKING:  # pragma: no cover
    from gage_eval.role.runtime.invocation import (
        RoleInvocationContext,
        RuntimeRouteDecision,
    )


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
        resolved_policy = (
            HistorySnapshotPolicy(copy_mode="deep")
            if policy is None
            else HistorySnapshotPolicy.parse(policy)
        )
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
        return clone_json_like(message)

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


@dataclass
class RoleInvocationBinding:
    """Transient invocation binding attached while a role lease is active."""

    route_decision: "RuntimeRouteDecision"
    sandbox_provider: Optional[Any] = None
    execution_context: Optional["RoleInvocationContext"] = None


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
        self._invocation_binding: Optional[RoleInvocationBinding] = None

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

    def attach_invocation_binding(
        self,
        route_decision: "RuntimeRouteDecision",
        *,
        sandbox_provider: Optional[Any] = None,
        execution_context: Optional["RoleInvocationContext"] = None,
    ) -> None:
        """Attach route and sandbox metadata for the current lease."""

        self._invocation_binding = RoleInvocationBinding(
            route_decision=route_decision,
            sandbox_provider=sandbox_provider,
            execution_context=execution_context,
        )

    def clear_invocation_binding(self) -> None:
        """Clear the transient invocation binding when the lease is released."""

        self._invocation_binding = None

    def history_snapshot(
        self, policy: Optional[HistorySnapshotPolicy | Mapping[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        try:
            return self._history.snapshot(policy)
        except TypeError:
            return self._history.snapshot()

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------
    def invoke(self, payload: Optional[Dict[str, Any]], trace: ObservabilityTrace) -> Dict[str, Any]:
        base_payload = payload or {}
        history_override = base_payload.get("history_override")
        history_delta = base_payload.get("history_delta")
        self._apply_history_override(history_override)
        self._ingest_history_delta(history_delta)
        history_policy = self._resolve_history_policy()
        prepared_payload, history_tokens = self._build_invoke_payload(
            base_payload,
            has_history_override=history_override is not None,
            has_history_delta=history_delta is not None,
            history_policy=history_policy,
        )
        self._apply_invocation_binding(prepared_payload)
        route_mode = self._invocation_binding.route_decision.runtime_mode if self._invocation_binding else "direct"

        logger.debug(
            "Role invoke adapter={} runtime={} route_mode={} history_tokens={} history_copy_mode={} history_window={}",
            self.adapter_id,
            bool(self._runtime),
            route_mode,
            history_tokens,
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

    def _apply_invocation_binding(self, prepared_payload: Dict[str, Any]) -> None:
        if self._invocation_binding is None:
            return
        route_payload = dict(prepared_payload.get("runtime_route") or {})
        for key, value in self._invocation_binding.route_decision.to_payload().items():
            route_payload.setdefault(key, value)
        prepared_payload["runtime_route"] = route_payload
        if self._invocation_binding.sandbox_provider is not None:
            prepared_payload.setdefault(
                "sandbox_provider", self._invocation_binding.sandbox_provider
            )
        execution_context = self._invocation_binding.execution_context
        if execution_context is not None:
            payload = {
                "run_id": execution_context.run_id,
                "task_id": execution_context.task_id,
                "sample_id": execution_context.sample_id,
                "step_type": execution_context.step_type,
                "adapter_id": execution_context.adapter_id,
            }
            if execution_context.step_slot_id is not None:
                payload["step_slot_id"] = execution_context.step_slot_id
            prepared_payload.setdefault("execution_context", payload)

    def _build_invoke_payload(
        self,
        base_payload: Dict[str, Any],
        *,
        has_history_override: bool,
        has_history_delta: bool,
        history_policy: HistorySnapshotPolicy,
    ) -> tuple[Dict[str, Any], int]:
        has_history = "history" in base_payload
        has_messages = "messages" in base_payload
        history_view = base_payload.get("history") if has_history else None
        messages_view = base_payload.get("messages") if has_messages else None
        inject_history = False
        inject_messages = False

        if not has_history and not has_messages:
            if not self._history.is_empty():
                history_view = self.history_snapshot(policy=history_policy)
                messages_view = history_view
                inject_history = True
                inject_messages = True
        elif not has_history and messages_view is not None:
            history_view = messages_view
            inject_history = True
        elif not has_messages and history_view is not None:
            messages_view = history_view
            inject_messages = True

        needs_copy = (
            has_history_override
            or has_history_delta
            or inject_history
            or inject_messages
            or self._invocation_binding is not None
        )
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
