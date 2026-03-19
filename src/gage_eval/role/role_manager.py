"""RoleManager implementation controlling RoleAdapters and resources."""

from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Iterator, Optional, Sequence

from loguru import logger
from gage_eval.role.auto_pool import AutoPoolPlanner, PoolShardPlan
from gage_eval.role.role_instance import ConversationHistory, Role
from gage_eval.role.role_pool import RolePool
from gage_eval.role.resource_profile import ResourceProfile
from gage_eval.role.runtime.base_pool import BasePool
from gage_eval.role.runtime.rate_limiter import RateLimiter
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
        self._session_ctx: ContextVar[Optional["_SampleSessionContext"]] = ContextVar(
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

        # STEP 2: Preserve the current role runtime logging while keeping runtime selection unchanged.
        runtime: Optional[object] = None
        if adapter.role_type in {"dut_model", "judge_model", "helper_model"}:
            backend = getattr(adapter, "backend", None)
            exec_mode = getattr(backend, "execution_mode", "native")
            if exec_mode == "native":
                logger.info(
                    "Registering adapter '{}' in native fast-path (no external runtime, backend invokes directly)",
                    adapter_id,
                )
            else:
                # NOTE: HTTP/remote backends rely on RoleAdapter + Backend internal
                # concurrency and rate-limiting. We do not build an
                # InferenceRuntime/BatchingScheduler for this path.
                logger.info(
                    "Registering adapter '{}' in http/remote path (no InferenceRuntime, lightweight concurrency)",
                    adapter_id,
                )

        # STEP 3: Always expose a composite pool to keep capacity, shutdown, and snapshot semantics aligned.
        composite_pool = _build_composite_pool(adapter_id, adapter, runtime, shard_plans)
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

    def borrow_role(self, adapter_id: Optional[str]):
        if not adapter_id:
            return nullcontext(None)
        pool = self._role_pools.get(adapter_id)
        if pool is None:
            raise KeyError(f"Role '{adapter_id}' is not registered")
        logger.trace("Borrowing role for adapter '{}'", adapter_id)
        lease = pool.acquire()

        history = self._history_for(adapter_id)

        class _HistoryLease:
            def __init__(self, lease):
                self._lease = lease

            def __enter__(self):
                role = self._lease.__enter__()
                role.attach_history(history)
                return role

            def __exit__(self, exc_type, exc, tb):
                return self._lease.__exit__(exc_type, exc, tb)

        return _HistoryLease(lease)

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
            pool_snapshot = pool.snapshot()
            assembly_plan = self._pool_plans.get(adapter_id)
            adapters.append(
                {
                    "adapter_id": adapter_id,
                    "role_type": getattr(adapter, "role_type", "unknown"),
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
    def _activate_session(self, sample: dict) -> Token:
        return self._session_ctx.set(_SampleSessionContext(sample=sample))

    def _deactivate_session(self, token: Token) -> None:
        self._session_ctx.reset(token)

    def _current_session(self) -> Optional["_SampleSessionContext"]:
        return self._session_ctx.get()

    def _history_for(self, adapter_id: str) -> ConversationHistory:
        session = self._current_session()
        if session is None:
            return ConversationHistory()
        return session.get_history(adapter_id)

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
    return ShardedRolePool(adapter_id, shards)


class PerSampleSession:
    """Context manager wiring RolePool into step execution."""

    def __init__(self, context, role_manager: RoleManager) -> None:
        self.context = context
        self.role_manager = role_manager
        self._session_token: Optional[Token] = None

    def __enter__(self):
        self._session_token = self.role_manager._activate_session(self.context.sample)
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


@dataclass
class _SampleSessionContext:
    sample: dict
    histories: Dict[str, ConversationHistory] = field(default_factory=dict)

    def get_history(self, adapter_id: str) -> ConversationHistory:
        if adapter_id not in self.histories:
            self.histories[adapter_id] = ConversationHistory(self._initial_messages(adapter_id))
        return self.histories[adapter_id]

    def _initial_messages(self, adapter_id: str):
        role_histories = self.sample.get("role_histories")
        if isinstance(role_histories, dict):
            candidate = role_histories.get(adapter_id)
            if isinstance(candidate, list):
                return candidate
        messages = self.sample.get("messages")
        if isinstance(messages, list):
            return messages
        return None
