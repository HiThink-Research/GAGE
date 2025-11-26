"""RoleManager implementation controlling RoleAdapters and resources."""

from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

from loguru import logger
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.auto_pool import AutoPoolPlanner
from gage_eval.role.role_pool import RolePool
from gage_eval.role.role_instance import ConversationHistory, Role
from gage_eval.role.resource_profile import ResourceProfile
from gage_eval.role.runtime.sharded_pool import PoolShard, ShardedRolePool
from gage_eval.role.runtime.rate_limiter import RateLimiter
from gage_eval.role.runtime.base_pool import BasePool


class RoleManager:
    """High-level facade coordinating ResourceProfile, AutoPool and pools."""

    def __init__(self, resource_profile: ResourceProfile, concurrency_hint: Optional[int] = None) -> None:
        self._resource_profile = resource_profile
        self._auto_pool = AutoPoolPlanner()
        self._adapters: Dict[str, any] = {}
        self._role_pools: Dict[str, BasePool] = {}
        self._session_ctx: ContextVar[Optional["_SampleSessionContext"]] = ContextVar(
            "role_manager_session", default=None
        )
        self._concurrency_hint = concurrency_hint if concurrency_hint and concurrency_hint > 0 else None

    def register_role_adapter(self, adapter_id: str, adapter) -> None:
        self._adapters[adapter_id] = adapter
        # Translate hardware profile constraints into concrete pool shards (per endpoint/per node).
        shard_plans = self._auto_pool.plan_instances(self._resource_profile, adapter)
        total_planned = sum(plan.size for plan in shard_plans) or 1
        if not adapter.resource_requirement.get("pool_size") and self._concurrency_hint:
            if self._concurrency_hint > total_planned and shard_plans:
                deficit = self._concurrency_hint - total_planned
                shard_plans[0].size += deficit
                adjusted = sum(plan.size for plan in shard_plans)
                logger.warning(
                    "Pool size smaller than concurrency hint, auto-bumping adapter=%s from %s to %s",
                    adapter_id,
                    total_planned,
                    adjusted,
                )
                total_planned = adjusted
        max_workers = sum(plan.size for plan in shard_plans) or 1
        logger.info(
            "Registering adapter '{}' (role_type={}, shards={}, max_workers={})",
            adapter_id,
            getattr(adapter, "role_type", "unknown"),
            len(shard_plans),
            max_workers,
        )
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
                # HTTP / 远程后端：直接依赖 RoleAdapter + Backend 自身的并发与限流策略，
                # 不再构建 InferenceRuntime/BatchingScheduler。
                logger.info(
                    "Registering adapter '{}' in http/remote path (no InferenceRuntime, lightweight concurrency)",
                    adapter_id,
                )

        pool_capacity = adapter.resource_requirement.get("pool_size")
        if not pool_capacity:
            pool_capacity = max_workers
        if len(shard_plans) <= 1:
            plan = shard_plans[0]
            pool_capacity = plan.size or pool_capacity
            self._role_pools[adapter_id] = RolePool(
                adapter_id=adapter_id,
                builder=lambda adapter_id=adapter_id, adapter=adapter, runtime=runtime: Role(adapter_id, adapter, runtime),
                max_size=pool_capacity,
            )
        else:
            shards = []
            for plan in shard_plans:
                rate_limiter = _build_rate_limiter(plan.rate_limit)
                shard_pool = RolePool(
                    adapter_id=f"{adapter_id}:{plan.shard_id}",
                    builder=lambda adapter_id=adapter_id, adapter=adapter, runtime=runtime: Role(adapter_id, adapter, runtime),
                    max_size=plan.size,
                )
                shards.append(
                    PoolShard(
                        shard_id=plan.shard_id,
                        pool=shard_pool,
                        rate_limiter=rate_limiter,
                        metadata=plan.metadata,
                    )
                )
            self._role_pools[adapter_id] = ShardedRolePool(adapter_id, shards)

    def update_concurrency_hint(self, value: Optional[int]) -> None:
        if value is None:
            return
        hint = max(1, int(value))
        if self._concurrency_hint and self._concurrency_hint >= hint:
            return
        self._concurrency_hint = hint
        logger.info("Updated RoleManager concurrency hint to %s", hint)

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

    def per_sample_session(self, context) -> Iterator["PerSampleSession"]:
        return PerSampleSession(context, self)

    def shutdown(self) -> None:
        logger.info("Shutting down RoleManager ({} adapters)", len(self._adapters))
        for adapter in self._adapters.values():
            if hasattr(adapter, "shutdown"):
                adapter.shutdown()
        for role_pool in self._role_pools.values():
            role_pool.shutdown()

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        # 旧实现依赖 InferenceRuntime 采集队列/并发指标；Runtime 已下线，这里先返回空 dict。
        return {}

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
