"""Schema for dummy backend (testing)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase


class DummyBackendConfig(BackendConfigBase):
    responses: List[str] = Field(default_factory=lambda: ["dummy response"], description="循环返回的固定文本")
    cycle: bool = Field(default=True, description="耗尽后是否循环使用 responses")
    echo_prompt: bool = Field(default=False, description="若无预设响应时是否回显 prompt")
    metadata: Optional[dict] = Field(default=None, description="写回到结果中的 metadata")
