"""Smart-default profile selection for pipeline payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gage_eval.config.smart_defaults.static_rules as _static_rules  # noqa: F401
from gage_eval.config.smart_defaults.registry import registered_rules
from gage_eval.config.smart_defaults.types import SceneName, SmartDefaultsError, SmartDefaultsProfile

STATIC_PHASES: tuple[str, ...] = ("dataset", "backend", "role_adapter", "custom_steps", "task")
_NOOP_PHASES: tuple[str, ...] = ()
_KNOWN_SCENES = {"static", "agent", "game"}


def select_smart_defaults_profile(payload: dict[str, Any], source_path: Path | None) -> SmartDefaultsProfile:
    """Select the smart-default profile for a raw config payload."""

    kind = str(payload.get("kind") or "PipelineConfig").lower()
    if kind != "pipelineconfig":
        return SmartDefaultsProfile(scene="legacy", phases=_NOOP_PHASES, rules=())

    scene_value = payload.get("scene")
    if scene_value is None:
        return SmartDefaultsProfile(scene="legacy", phases=_NOOP_PHASES, rules=())

    scene = str(scene_value).strip().lower()
    if scene not in _KNOWN_SCENES:
        raise SmartDefaultsError(f"Unknown PipelineConfig scene '{scene_value}'")
    if scene != "static":
        return SmartDefaultsProfile(scene=scene, phases=_NOOP_PHASES, rules=())

    return SmartDefaultsProfile(scene="static", phases=STATIC_PHASES, rules=registered_rules("static"))
