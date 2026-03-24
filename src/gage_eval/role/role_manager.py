"""RoleManager implementation controlling RoleAdapters and resources."""

from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Dict, Iterator, Optional, Sequence

from loguru import logger
from gage_eval.role.auto_pool import AutoPoolPlanner, PoolShardPlan
from gage_eval.role.role_instance import ConversationHistory, Role
from gage_eval.role.role_pool import RolePool
from gage_eval.role.resource_profile import ResourceProfile
from gage_eval.role.runtime.base_pool import BasePool
from gage_eval.role.runtime.invocation import (
    LegacyContextBridge,
    RoleInvocationContext,
    RuntimeRouteDecision,
    RuntimeRouteTemplate,
    SampleExecutionContext,
    SandboxBinding,
)
from gage_eval.role.runtime.rate_limiter import RateLimiter
from gage_eval.role.runtime.shard_selection import (
    ShardSchedulingConfig,
    ShardSelectionPolicy,
    build_shard_selection_policies,
    normalize_shard_scheduling_config,
)
from gage_eval.role.runtime.strategy import RuntimeStrategyFactory
from gage_eval.role.runtime.sharded_pool import PoolShard, ShardedRolePool


@dataclass(frozen=True)
class PoolAssemblyPlan:
    """Summarize the configured pool contract for one adapter."""

    adapter_id: str
    role_type: str
    planned_capacity: int
    effective_capacity: int
    shard_count: int
    hint_adjusted: bool = False


@dataclass(frozen=True)
class ShutdownIssue:
    """Describe one shutdown failure without aborting the full close path."""

    phase: str
    component_type: str
    component_id: str
    error_type: str
    error_message: str
    duration_ms: float


class RoleManagerShutdownError(RuntimeError):
    """Raised when RoleManager shutdown completed with one or more issues."""

    def __init__(self, issues: Sequence[ShutdownIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__(f"RoleManager shutdown completed with {len(self.issues)} issue(s)")


class RoleManager:
    """High-level facade coordinating ResourceProfile, AutoPool and pools."""

    def __init__(self, resource_profile: ResourceProfile, concurrency_hint: Optional[int] = None) -> None:
        self._resource_profile = resource_profile
        self._auto_pool = AutoPoolPlanner()
        self._adapters: Dict[str, Any] = {}
        self._role_pools: Dict[str, BasePool] = {}
        self._pool_plans: Dict[str, PoolAssemblyPlan] = {}
        self._route_templates: Dict[str, RuntimeRouteTemplate] = {}
        self._runtime_factory = RuntimeStrategyFactory()
        self._session_ctx: ContextVar[Optional[SampleExecutionContext]] = ContextVar(
            "role_manager_session", default=None
        )
        self._concurrency_hint = concurrency_hint if concurrency_hint and concurrency_hint > 0 else None

    def register_role_adapter(self, adapter_id: str, adapter) -> None:
        self._adapters[adapter_id] = adapter

        # STEP 1: Build the shard plan and apply the optional concurrency hint.
        shard_plans = self._auto_pool.plan_instances(self._resource_profile, adapter)
        total_planned = sum(plan.size for plan in shard_plans) or 1
        hint_adjusted = False
        requirement = getattr(adapter, "resource_requirement", {}) or {}
        scheduling_config = normalize_shard_scheduling_config(requirement)
        selection_policy, fallback_policy = build_shard_selection_policies(scheduling_config)
        if not requirement.get("pool_size") and self._concurrency_hint:
            if self._concurrency_hint > total_planned and shard_plans:
                deficit = self._concurrency_hint - total_planned
                shard_plans[0].size += deficit
                adjusted = sum(plan.size for plan in shard_plans)
                hint_adjusted = adjusted != total_planned
                logger.warning(
                    "Pool size smaller than concurrency hint, auto-bumping adapter={} from {} to {}",
                    adapter_id,
                    total_planned,
                    adjusted,
                )
                total_planned = adjusted

        # STEP 2: Build a concrete runtime strategy instead of leaving an empty branch.
        runtime, runtime_mode, strategy_id = self._runtime_factory.build(adapter)
        self._route_templates[adapter_id] = RuntimeRouteTemplate(
            adapter_id=adapter_id,
            role_type=getattr(adapter, "role_type", "unknown"),
            runtime_mode=runtime_mode,
            strategy_id=strategy_id,
            session_mode="explicit_context",
            default_sandbox_config=dict(getattr(adapter, "sandbox_config", {}) or {}),
            supports_sandbox=bool(getattr(adapter, "sandbox_config", {}) or {}),
        )
        if runtime_mode == "native":
            logger.info(
                "Registering adapter '{}' with native runtime strategy '{}'",
                adapter_id,
                strategy_id,
            )
        else:
            logger.info(
                "Registering adapter '{}' with runtime mode={} strategy={}",
                adapter_id,
                runtime_mode,
                strategy_id,
            )

        # STEP 3: Always expose a composite pool to keep capacity, shutdown, and snapshot semantics aligned.
        composite_pool = _build_composite_pool(
            adapter_id,
            adapter,
            runtime,
            shard_plans,
            scheduling_config=scheduling_config,
            selection_policy=selection_policy,
            fallback_policy=fallback_policy,
        )
        effective_capacity = sum(max(1, int(plan.size)) for plan in shard_plans) or 1
        self._role_pools[adapter_id] = composite_pool
        self._pool_plans[adapter_id] = PoolAssemblyPlan(
            adapter_id=adapter_id,
            role_type=getattr(adapter, "role_type", "unknown"),
            planned_capacity=total_planned,
            effective_capacity=effective_capacity,
            shard_count=len(shard_plans),
            hint_adjusted=hint_adjusted,
        )
        logger.info(
            "Registered adapter '{}' (role_type={}, shard_count={}, planned_capacity={}, effective_capacity={}, hint_adjusted={})",
            adapter_id,
            getattr(adapter, "role_type", "unknown"),
            len(shard_plans),
            total_planned,
            effective_capacity,
            hint_adjusted,
        )

    def update_concurrency_hint(self, value: Optional[int]) -> None:
        if value is None:
            return
        hint = max(1, int(value))
        if self._concurrency_hint and self._concurrency_hint >= hint:
            return
        self._concurrency_hint = hint
        logger.info("Updated RoleManager concurrency hint to {}", hint)

    def borrow_role(
        self,
        adapter_id: Optional[str],
        *,
        execution_context: Optional[RoleInvocationContext] = None,
    ):
        if not adapter_id:
            return nullcontext(None)
        pool = self._role_pools.get(adapter_id)
        if pool is None:
            raise KeyError(f"Role '{adapter_id}' is not registered")
        logger.trace("Borrowing role for adapter '{}'", adapter_id)
        lease = pool.acquire()
        invocation_context = execution_context
        if invocation_context is None:
            invocation_context = LegacyContextBridge.from_sample_context(
                self._current_session(),
                adapter_id=adapter_id,
            )
        session_mode = (
            "explicit_context"
            if execution_context is not None
            else ("legacy_contextvar" if invocation_context is not None else "detached")
        )
        route_decision = self._resolve_route_decision(
            adapter_id,
            invocation_context,
            session_mode=session_mode,
        )
        sandbox_provider = self._resolve_invocation_sandbox_provider(
            invocation_context,
            route_decision.sandbox_binding,
        )
        history = self._history_for(adapter_id, invocation_context)

        class _HistoryLease:
            def __init__(
                self,
                lease,
                history: ConversationHistory,
                route_decision: RuntimeRouteDecision,
                sandbox_provider: Optional[Any],
                invocation_context: Optional[RoleInvocationContext],
            ) -> None:
                self._lease = lease
                self._history = history
                self._route_decision = route_decision
                self._sandbox_provider = sandbox_provider
                self._invocation_context = invocation_context
                self._role: Optional[Role] = None

            def __enter__(self):
                role = self._lease.__enter__()
                role.attach_history(self._history)
                role.attach_invocation_binding(
                    self._route_decision,
                    sandbox_provider=self._sandbox_provider,
                    execution_context=self._invocation_context,
                )
                self._role = role
                return role

            def __exit__(self, exc_type, exc, tb):
                if self._role is not None:
                    self._role.clear_invocation_binding()
                return self._lease.__exit__(exc_type, exc, tb)

        return _HistoryLease(
            lease,
            history,
            route_decision,
            sandbox_provider,
            invocation_context,
        )

    def _resolve_invocation_sandbox_provider(
        self,
        invocation_context: Optional[RoleInvocationContext],
        binding: SandboxBinding,
    ) -> Optional[Any]:
        if invocation_context is None:
            return None
        default_provider = invocation_context.default_sandbox_provider
        default_config = (
            dict(getattr(default_provider, "sandbox_config", {}) or {})
            if default_provider is not None
            else {}
        )
        if not binding.enabled or not binding.config:
            return default_provider
        effective_config = dict(binding.config)
        if default_config:
            if invocation_context.sandbox_router is not None:
                effective_config = dict(
                    invocation_context.sandbox_router.resolve_config(
                        default_config,
                        effective_config,
                    )
                )
            else:
                effective_config = _merge_dicts(default_config, effective_config)
        if default_provider is not None and effective_config == default_config:
            return default_provider
        if invocation_context.sandbox_router is None:
            return default_provider
        if effective_config != binding.config:
            binding = SandboxBinding(
                enabled=binding.enabled,
                config=effective_config,
                source=binding.source,
                route_key=_build_binding_route_key(
                    effective_config,
                    step_type=binding.step_type or invocation_context.step_type,
                    adapter_id=binding.adapter_id or invocation_context.adapter_id,
                ),
                step_type=binding.step_type,
                adapter_id=binding.adapter_id,
                step_slot_id=binding.step_slot_id,
            )
        return invocation_context.sandbox_router.get_provider(binding)

    def get_adapter(self, adapter_id: str):
        """Return a registered adapter by id if available."""

        return self._adapters.get(adapter_id)

    def per_sample_session(self, context) -> Iterator["PerSampleSession"]:
        return PerSampleSession(context, self)

    def shutdown(self) -> None:
        logger.info("Shutting down RoleManager ({} adapters)", len(self._adapters))
        issues: list[ShutdownIssue] = []
        self._shutdown_phase(
            phase="adapter_shutdown",
            component_type="adapter",
            components=self._adapters.items(),
            issues=issues,
        )
        self._shutdown_phase(
            phase="pool_shutdown",
            component_type="pool",
            components=self._role_pools.items(),
            issues=issues,
        )
        if issues:
            raise RoleManagerShutdownError(issues)

    def snapshot(self) -> dict[str, Any]:
        """Return the current runtime state for all registered adapters."""

        adapters: list[dict[str, Any]] = []
        for adapter_id in sorted(self._role_pools):
            pool = self._role_pools[adapter_id]
            adapter = self._adapters.get(adapter_id)
            route_template = self._route_templates.get(adapter_id)
            pool_snapshot = pool.snapshot()
            assembly_plan = self._pool_plans.get(adapter_id)
            adapters.append(
                {
                    "adapter_id": adapter_id,
                    "role_type": getattr(adapter, "role_type", "unknown"),
                    "runtime_mode": route_template.runtime_mode if route_template is not None else "native",
                    "runtime_strategy": route_template.strategy_id if route_template is not None else "native_runtime",
                    "session_mode": route_template.session_mode if route_template is not None else "legacy_contextvar",
                    "sandbox_enabled_default": bool(
                        route_template.default_sandbox_config
                    ) if route_template is not None else False,
                    "pool_type": pool_snapshot.get("pool_type", "unknown"),
                    "planned_capacity": (
                        assembly_plan.planned_capacity
                        if assembly_plan is not None
                        else pool_snapshot.get("capacity_total", 0)
                    ),
                    "effective_capacity": (
                        assembly_plan.effective_capacity
                        if assembly_plan is not None
                        else pool_snapshot.get("capacity_total", 0)
                    ),
                    "capacity_total": pool_snapshot.get("capacity_total", pool_snapshot.get("capacity")),
                    "in_use_total": pool_snapshot.get("in_use_total", pool_snapshot.get("in_use", 0)),
                    "available_total": pool_snapshot.get("available_total", pool_snapshot.get("available", 0)),
                    "created_total": pool_snapshot.get("created_total", pool_snapshot.get("created", 0)),
                    "healthy": pool_snapshot.get("healthy", True),
                    "shard_count": (
                        assembly_plan.shard_count
                        if assembly_plan is not None
                        else pool_snapshot.get("shard_count", 0)
                    ),
                    "hint_adjusted": (
                        assembly_plan.hint_adjusted if assembly_plan is not None else False
                    ),
                    "shards": list(pool_snapshot.get("shards", ())),
                    "extensions": dict(pool_snapshot.get("extensions", {})),
                }
            )
        return {
            "snapshot_version": "role_manager.v1",
            "timestamp_ms": int(time.time() * 1000),
            "adapters": adapters,
            "extensions": {},
        }

    # ------------------------------------------------------------------
    # Session & history helpers
    # ------------------------------------------------------------------
    def _activate_session(self, context: SampleExecutionContext) -> Token:
        return self._session_ctx.set(context)

    def _deactivate_session(self, token: Token) -> None:
        self._session_ctx.reset(token)

    def _current_session(self) -> Optional[SampleExecutionContext]:
        return self._session_ctx.get()

    def _history_for(
        self,
        adapter_id: str,
        invocation_context: Optional[RoleInvocationContext] = None,
    ) -> ConversationHistory:
        if invocation_context is not None:
            return invocation_context.session_store.get_history(adapter_id)
        session = self._current_session()
        if session is None:
            return ConversationHistory()
        return session.session_store.get_history(adapter_id)

    def _resolve_route_decision(
        self,
        adapter_id: str,
        invocation_context: Optional[RoleInvocationContext],
        *,
        session_mode: str,
    ) -> RuntimeRouteDecision:
        template = self._route_templates.get(adapter_id)
        adapter = self._adapters.get(adapter_id)
        if template is None:
            _, runtime_mode, strategy_id = self._runtime_factory.build(adapter)
            template = RuntimeRouteTemplate(
                adapter_id=adapter_id,
                role_type=getattr(adapter, "role_type", "unknown"),
                runtime_mode=runtime_mode,
                strategy_id=strategy_id,
                session_mode="legacy_contextvar",
                default_sandbox_config=dict(getattr(adapter, "sandbox_config", {}) or {}),
                supports_sandbox=bool(getattr(adapter, "sandbox_config", {}) or {}),
            )
            self._route_templates[adapter_id] = template
        step_type = invocation_context.step_type if invocation_context is not None else "detached"
        sandbox_binding = self._resolve_sandbox_binding(template, invocation_context)
        route_source_marker = sandbox_binding.route_key or sandbox_binding.source
        cache_key = None
        if invocation_context is not None:
            cache_key = invocation_context.cache_key(
                f"{template.runtime_mode}:{session_mode}:{route_source_marker}"
            )
            cached = invocation_context.route_cache.get(cache_key)
            if cached is not None:
                return cached
        decision = RuntimeRouteDecision(
            adapter_id=adapter_id,
            role_type=template.role_type,
            step_type=step_type,
            runtime_mode=template.runtime_mode,
            strategy_id=template.strategy_id,
            session_mode=session_mode,
            route_source=sandbox_binding.source,
            sandbox_binding=sandbox_binding,
            step_slot_id=invocation_context.step_slot_id if invocation_context is not None else None,
            observability_tags={
                "runtime_mode": template.runtime_mode,
                "session_mode": session_mode,
            },
        )
        if cache_key is not None:
            invocation_context.route_cache[cache_key] = decision
        if invocation_context is not None and invocation_context.trace is not None:
            invocation_context.trace.emit(
                "runtime_route_selected",
                decision.to_payload(),
                sample_id=invocation_context.sample_id,
            )
        return decision

    def _resolve_sandbox_binding(
        self,
        template: RuntimeRouteTemplate,
        invocation_context: Optional[RoleInvocationContext],
    ) -> SandboxBinding:
        if invocation_context is None:
            return SandboxBinding(enabled=False, source="detached")
        sample = invocation_context.sample
        sample_default = sample.get("sandbox")
        if not isinstance(sample_default, dict):
            sample_default = None
        route_override, route_source, disabled = _resolve_sample_route_override(
            sample,
            step_type=invocation_context.step_type,
            adapter_id=invocation_context.adapter_id,
            step_slot_id=invocation_context.step_slot_id,
        )
        if disabled:
            return SandboxBinding(
                enabled=False,
                source=route_source,
                step_type=invocation_context.step_type,
                adapter_id=invocation_context.adapter_id,
                step_slot_id=invocation_context.step_slot_id,
            )
        default_override = _resolve_default_sandbox_override(sample)
        sandbox_router = invocation_context.sandbox_router
        effective: Dict[str, Any] = dict(template.default_sandbox_config or {})
        if sandbox_router is not None and sample_default:
            effective = dict(
                sandbox_router.resolve_config(effective, dict(sample_default))
            )
        elif sample_default:
            effective = _merge_dicts(effective, dict(sample_default))
        if sandbox_router is not None and default_override:
            effective = dict(
                sandbox_router.resolve_config(effective, dict(default_override))
            )
        elif default_override:
            effective = _merge_dicts(effective, dict(default_override))
        if sandbox_router is not None and route_override:
            effective = dict(
                sandbox_router.resolve_config(effective, dict(route_override))
            )
        elif route_override:
            effective = _merge_dicts(effective, dict(route_override))
        if not effective:
            return SandboxBinding(
                enabled=False,
                source="disabled",
                step_type=invocation_context.step_type,
                adapter_id=invocation_context.adapter_id,
                step_slot_id=invocation_context.step_slot_id,
            )
        source = route_source
        if source == "disabled":
            if route_override:
                source = "sample_route_override"
            elif default_override or sample_default:
                source = "sample_default"
            else:
                source = "adapter_default"
        return SandboxBinding(
            enabled=True,
            config=effective,
            source=source,
            route_key=_build_binding_route_key(
                effective,
                step_type=invocation_context.step_type,
                adapter_id=invocation_context.adapter_id,
            ),
            step_type=invocation_context.step_type,
            adapter_id=invocation_context.adapter_id,
            step_slot_id=invocation_context.step_slot_id,
        )

    def _shutdown_phase(
        self,
        *,
        phase: str,
        component_type: str,
        components: Sequence[tuple[str, Any]] | Any,
        issues: list[ShutdownIssue],
    ) -> None:
        for component_id, component in components:
            shutdown_fn = getattr(component, "shutdown", None)
            if not callable(shutdown_fn):
                continue
            start = time.perf_counter()
            try:
                shutdown_fn()
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000.0
                issue = ShutdownIssue(
                    phase=phase,
                    component_type=component_type,
                    component_id=component_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    duration_ms=duration_ms,
                )
                issues.append(issue)
                logger.error(
                    "RoleManager shutdown issue phase={} component_type={} component_id={} error_type={} error={}",
                    phase,
                    component_type,
                    component_id,
                    issue.error_type,
                    issue.error_message,
                )


def _build_rate_limiter(config):
    if not config:
        return None
    capacity = (
        config.get("capacity")
        or config.get("burst")
        or config.get("max_requests")
        or config.get("qps")
        or 0
    )
    interval = config.get("interval") or config.get("per_seconds")
    qps = config.get("qps")
    if capacity and interval:
        return RateLimiter(int(capacity), float(interval))
    if capacity and not interval and qps:
        return RateLimiter(int(capacity), 1.0 / float(qps))
    if qps and not capacity:
        return RateLimiter(1, 1.0 / float(qps))
    return None


def _build_composite_pool(
    adapter_id: str,
    adapter: Any,
    runtime: Optional[object],
    shard_plans: Sequence[PoolShardPlan],
    *,
    scheduling_config: ShardSchedulingConfig,
    selection_policy: ShardSelectionPolicy,
    fallback_policy: ShardSelectionPolicy,
) -> ShardedRolePool:
    shards: list[PoolShard] = []
    for plan in shard_plans:
        shard_pool = RolePool(
            adapter_id=plan.shard_id,
            builder=lambda adapter_id=adapter_id, adapter=adapter, runtime=runtime: Role(adapter_id, adapter, runtime),
            max_size=max(1, int(plan.size)),
        )
        shards.append(
            PoolShard(
                shard_id=plan.shard_id,
                pool=shard_pool,
                rate_limiter=_build_rate_limiter(plan.rate_limit),
                metadata=dict(plan.metadata),
            )
        )
    return ShardedRolePool(
        adapter_id,
        shards,
        scheduling_config=scheduling_config,
        selection_policy=selection_policy,
        fallback_policy=fallback_policy,
    )


class PerSampleSession:
    """Context manager wiring RolePool into step execution."""

    def __init__(self, context, role_manager: RoleManager) -> None:
        self.context = context
        self.role_manager = role_manager
        self._session_token: Optional[Token] = None

    def __enter__(self):
        self._session_token = self.role_manager._activate_session(
            self.context.execution_context
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._session_token is not None:
            self.role_manager._deactivate_session(self._session_token)
            self._session_token = None
        return False

    def execute_support(self) -> None:
        self.context.execute_support()

    def execute_support_step(self, step) -> None:
        self.context.execute_support_step(step)

    def execute_inference(self) -> None:
        self.context.execute_inference()

    def execute_arena(self) -> None:
        self.context.execute_arena()

    def execute_judge(self) -> None:
        self.context.execute_judge()

    def execute_auto_eval(self, sample_id: str) -> None:
        self.context.execute_auto_eval(sample_id)


def _resolve_default_sandbox_override(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sandbox_routes = sample.get("sandbox_routes")
    if not isinstance(sandbox_routes, dict):
        return None
    default_value = sandbox_routes.get("default")
    if isinstance(default_value, dict):
        return dict(default_value)
    return None


def _resolve_sample_route_override(
    sample: Dict[str, Any],
    *,
    step_type: str,
    adapter_id: str,
    step_slot_id: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], str, bool]:
    sandbox_routes = sample.get("sandbox_routes")
    if not isinstance(sandbox_routes, dict):
        return None, "disabled", False
    candidate_keys = []
    if step_slot_id:
        candidate_keys.extend(
            (
                f"{step_type}.{step_slot_id}.{adapter_id}",
                f"{step_type}.{step_slot_id}",
                f"{step_slot_id}.{adapter_id}",
                step_slot_id,
            )
        )
    candidate_keys.extend(
        (
            f"{step_type}.{adapter_id}",
            adapter_id,
            step_type,
        )
    )
    for key in candidate_keys:
        value = sandbox_routes.get(key)
        if not isinstance(value, dict):
            continue
        if value.get("disabled") is True:
            return None, f"sandbox_routes:{key}", True
        return dict(value), f"sandbox_routes:{key}", False
    return None, "disabled", False


def _build_binding_route_key(
    config: Dict[str, Any],
    *,
    step_type: str,
    adapter_id: str,
) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    lifecycle = str(config.get("lifecycle") or "per_sample")
    if lifecycle == "per_arena":
        return f"{digest}:{step_type}:{adapter_id}"
    return digest


def _merge_dicts(
    base: Dict[str, Any],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
