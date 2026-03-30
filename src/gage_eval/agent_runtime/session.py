"""Runtime session — per-sample execution handle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.resources.bundle import ResourceBundle
from gage_eval.observability.trace import ObservabilityTrace


@dataclass
class AgentRuntimeSession:
    """Holds per-sample runtime inputs and execution side channels."""

    sample: Dict[str, Any]
    trace: ObservabilityTrace
    plan: CompiledRuntimePlan
    resources: ResourceBundle
    artifacts: ArtifactLayout
    metadata: Dict[str, Any] = field(default_factory=dict)
