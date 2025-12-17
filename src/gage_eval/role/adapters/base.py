"""Base RoleAdapter definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Optional, List

from gage_eval.registry import run_sync


@dataclass
class RoleAdapterState:
    """Lightweight state cloned per sample."""

    metadata: Dict[str, Any] = field(default_factory=dict)


class RoleAdapter:
    """Base class for all role adapters (DUT, judge, helper, etc.)."""

    def __init__(
        self,
        adapter_id: str,
        role_type: str,
        capabilities: Sequence[str],
        *,
        resource_requirement: Optional[Dict[str, Any]] = None,
        sandbox_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.adapter_id = adapter_id
        self.role_type = role_type
        self.capabilities = tuple(capabilities)
        # 通用配置显式持有，避免外部动态 setattr
        self.resource_requirement = resource_requirement or {}
        self.sandbox_config = sandbox_config or {}

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def initialize(self) -> None:  # pragma: no cover - stub
        """Perform heavy initialization (load models, open sessions, etc.)."""

    def shutdown(self) -> None:  # pragma: no cover - stub
        """Release heavy resources. Called when the pipeline shuts down."""

    # ------------------------------------------------------------------
    # Sample-level operations
    # ------------------------------------------------------------------
    def clone_for_sample(self) -> RoleAdapterState:
        """Return a sample-local state object.

        Subclasses may override to include session handles or sandbox
        tickets. The default implementation simply returns an empty
        state so that callers never need to check for ``None``.
        """

        return RoleAdapterState()

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def invoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        """Synchronous facade retained while the runtime is refactored to asyncio."""

        return run_sync(self.ainvoke(payload, state))

    # ------------------------------------------------------------------
    # Producer/consumer split hooks
    # ------------------------------------------------------------------
    def prepare_request(self, payload: Dict[str, Any], state: RoleAdapterState) -> Any:
        """CPU 密集型准备逻辑：prompt 渲染、采样参数组装等。

        默认实现直接回传 payload，保证向后兼容。
        """

        return payload

    def execute_batch(self, requests: List[Any]) -> List[Any]:
        """批量执行后端请求。

        默认实现串行调用 invoke，实现类可覆盖为真正的合批调用。
        """

        results: List[Any] = []
        for request in requests:
            state = RoleAdapterState() if not isinstance(request, dict) else request.get("_state", RoleAdapterState())
            results.append(self.invoke(request if not isinstance(request, dict) else request.get("_payload", request), state))
        return results

    def parse_response(self, response: Any, state: RoleAdapterState) -> Dict[str, Any]:
        """回传解析，可结合 state 做上下文拼装。默认原样返回。"""

        return response
