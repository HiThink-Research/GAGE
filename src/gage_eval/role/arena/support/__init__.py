"""Support workflows and unit orchestration for GameArena."""

from __future__ import annotations

from gage_eval.role.arena.support.context import SupportContext
from gage_eval.role.arena.support.hooks import SupportHook, SupportUnit
from gage_eval.role.arena.support.registry import SupportWorkflowRegistry
from gage_eval.role.arena.support.workflow import GameSupportWorkflow
from gage_eval.role.arena.support.units import ContinuousActionShapingUnit

__all__ = [
    "GameSupportWorkflow",
    "ContinuousActionShapingUnit",
    "SupportContext",
    "SupportHook",
    "SupportUnit",
    "SupportWorkflowRegistry",
]
