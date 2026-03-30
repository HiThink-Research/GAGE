"""Execution context and route contracts for role invocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from gage_eval.role.role_instance import ConversationHistory

if TYPE_CHECKING:  # pragma: no cover
    from gage_eval.observability.trace import ObservabilityTrace
    from gage_eval.sandbox.session_router import SandboxSessionRouter


@dataclass
class RoleSessionStore:
    """Store adapter-scoped conversation history for one sample."""

    sample: Dict[str, Any]
    histories: Dict[str, ConversationHistory] = field(default_factory=dict)

    def get_history(self, adapter_id: str) -> ConversationHistory:
        """Returns the adapter-scoped history, creating it lazily."""

        if adapter_id not in self.histories:
            self.histories[adapter_id] = ConversationHistory(
                self._initial_messages(adapter_id)
            )
        return self.histories[adapter_id]

    def _initial_messages(self, adapter_id: str) -> Optional[list[dict[str, Any]]]:
        role_histories = self.sample.get("role_histories")
        if isinstance(role_histories, dict):
            candidate = role_histories.get(adapter_id)
            if isinstance(candidate, list):
                return candidate
        messages = self.sample.get("messages")
        if isinstance(messages, list):
            return messages
        return None


@dataclass(frozen=True)
class SandboxBinding:
    """Describe how one invocation should resolve sandbox access."""

    enabled: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    source: str = "disabled"
    route_key: Optional[str] = None
    step_type: Optional[str] = None
    adapter_id: Optional[str] = None
    step_slot_id: Optional[str] = None


@dataclass(frozen=True)
class RuntimeRouteTemplate:
    """Static route metadata prepared during adapter registration."""

    adapter_id: str
    role_type: str
    runtime_mode: str
    strategy_id: str
    session_mode: str = "explicit_context"
    default_sandbox_config: Dict[str, Any] = field(default_factory=dict)
    supports_sandbox: bool = False


@dataclass(frozen=True)
class RuntimeRouteDecision:
    """Resolved invocation route for one adapter/step/sample tuple."""

    adapter_id: str
    role_type: str
    step_type: str
    runtime_mode: str
    strategy_id: str
    session_mode: str
    route_source: str
    sandbox_binding: SandboxBinding = field(default_factory=SandboxBinding)
    step_slot_id: Optional[str] = None
    observability_tags: Dict[str, str] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-serializable route payload for adapters and traces."""

        payload = {
            "adapter_id": self.adapter_id,
            "role_type": self.role_type,
            "step_type": self.step_type,
            "runtime_mode": self.runtime_mode,
            "strategy_id": self.strategy_id,
            "session_mode": self.session_mode,
            "route_source": self.route_source,
            "sandbox_enabled": self.sandbox_binding.enabled,
        }
        if self.step_slot_id:
            payload["step_slot_id"] = self.step_slot_id
        if self.sandbox_binding.source:
            payload["sandbox_source"] = self.sandbox_binding.source
        if self.sandbox_binding.config:
            sandbox_id = (
                self.sandbox_binding.config.get("sandbox_id")
                or self.sandbox_binding.config.get("template_name")
                or self.sandbox_binding.config.get("runtime")
            )
            if sandbox_id:
                payload["sandbox_id"] = str(sandbox_id)
        if self.observability_tags:
            payload["tags"] = dict(self.observability_tags)
        return payload


@dataclass(frozen=True)
class SampleExecutionContext:
    """Sample-scoped context shared across all step invocations."""

    sample: Dict[str, Any]
    sample_id: str
    run_id: Optional[str] = None
    task_id: Optional[str] = None
    trace: Optional["ObservabilityTrace"] = None
    session_store: RoleSessionStore = field(default_factory=lambda: RoleSessionStore(sample={}))
    sandbox_router: Optional["SandboxSessionRouter"] = None
    sandbox_provider: Optional[Any] = None
    owns_sandbox_provider: bool = False
    route_cache: Dict[str, RuntimeRouteDecision] = field(default_factory=dict)

    def for_invocation(
        self,
        *,
        step_type: str,
        adapter_id: str,
        step_slot_id: Optional[str] = None,
    ) -> "RoleInvocationContext":
        """Derive an invocation-scoped view for one step and adapter."""

        return RoleInvocationContext(
            sample=self.sample,
            sample_id=self.sample_id,
            run_id=self.run_id,
            task_id=self.task_id,
            trace=self.trace,
            session_store=self.session_store,
            sandbox_router=self.sandbox_router,
            default_sandbox_provider=self.sandbox_provider,
            route_cache=self.route_cache,
            step_type=step_type,
            adapter_id=adapter_id,
            step_slot_id=step_slot_id,
        )

    def close(self) -> None:
        """Release sample-scoped runtime resources."""

        if self.sandbox_router is not None:
            self.sandbox_router.release_all()
        if self.owns_sandbox_provider and self.sandbox_provider is not None:
            release = getattr(self.sandbox_provider, "release", None)
            if callable(release):
                release()


@dataclass(frozen=True)
class RoleInvocationContext:
    """Invocation-scoped view derived from a sample execution context."""

    sample: Dict[str, Any]
    sample_id: str
    run_id: Optional[str]
    task_id: Optional[str]
    trace: Optional["ObservabilityTrace"]
    session_store: RoleSessionStore
    sandbox_router: Optional["SandboxSessionRouter"]
    default_sandbox_provider: Optional[Any]
    route_cache: Dict[str, RuntimeRouteDecision]
    step_type: str
    adapter_id: str
    step_slot_id: Optional[str] = None

    def cache_key(self, route_source: str) -> str:
        """Build a stable cache key for the resolved route."""

        return "|".join(
            (
                self.step_type,
                self.adapter_id,
                self.step_slot_id or "-",
                route_source,
            )
        )


class LegacyContextBridge:
    """Bridge legacy ContextVar-backed callers into explicit invocation context."""

    @staticmethod
    def from_sample_context(
        context: Optional[SampleExecutionContext],
        *,
        adapter_id: str,
    ) -> Optional[RoleInvocationContext]:
        if context is None:
            return None
        invocation = context.for_invocation(step_type="legacy", adapter_id=adapter_id)
        if context.trace is not None:
            context.trace.emit(
                "legacy_context_bridge_used",
                {
                    "adapter_id": adapter_id,
                    "step_type": "legacy",
                    "session_mode": "legacy_contextvar",
                },
                sample_id=context.sample_id,
            )
        return invocation
