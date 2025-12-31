"""Default observability plugin registrations."""

from __future__ import annotations

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.registry import registry


@registry.asset(
    "observability_plugins",
    "default",
    desc="默认文件/HTTP 观测插件",
    tags=("file", "http"),
)
class DefaultObservabilityPlugin:
    """Constructs the vanilla ObservabilityTrace."""

    def build_trace(self, **kwargs) -> ObservabilityTrace:
        return ObservabilityTrace(**kwargs)
