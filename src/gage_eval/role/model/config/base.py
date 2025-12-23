"""Shared helpers for backend configuration models."""

from __future__ import annotations

from pydantic import BaseModel

try:  # Pydantic v2+
    from pydantic import ConfigDict  # type: ignore
except ImportError:  # pragma: no cover - v1 fallback
    ConfigDict = None


class BackendConfigBase(BaseModel):
    """Base class for backend configuration models.

    设计目标：
    - 关闭 `model_*` 前缀的受保护命名空间，避免与后端参数冲突；
    - 默认允许 extra 字段，从而支持 typed + passthrough 模式：
      - 已声明字段享受类型校验与默认值；
      - 未声明字段保留在模型中，最终仍可透传给具体 Backend。
    """

    if ConfigDict is not None:  # pragma: no branch - runtime evaluated once
        # Pydantic v2 配置：允许额外字段 + 放宽受保护命名空间
        model_config = ConfigDict(  # type: ignore[assignment]
            protected_namespaces=(),
            extra="allow",
        )
    else:  # pragma: no cover - Pydantic v1

        class Config:
            protected_namespaces = ()
            extra = "allow"
