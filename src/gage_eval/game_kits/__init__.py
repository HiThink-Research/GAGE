"""Game arena kit registry and exports."""

from __future__ import annotations

from gage_eval.game_kits import content as _content  # noqa: F401
from gage_eval.game_kits.contracts import EnvSpec, GameKit, ResolvedRuntimeBinding
from gage_eval.game_kits.observation import ObservationWorkflow
from gage_eval.game_kits.registry import GameKitRegistry, ObservationWorkflowRegistry
from gage_eval.game_kits.runtime_binding import RuntimeBindingResolver

__all__ = [
    "EnvSpec",
    "GameKit",
    "GameKitRegistry",
    "ObservationWorkflow",
    "ObservationWorkflowRegistry",
    "ResolvedRuntimeBinding",
    "RuntimeBindingResolver",
]
