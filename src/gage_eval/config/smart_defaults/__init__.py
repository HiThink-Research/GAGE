"""Smart-default rule framework."""

from __future__ import annotations

from gage_eval.config.smart_defaults.profiles import STATIC_PHASES, select_smart_defaults_profile
from gage_eval.config.smart_defaults.registry import apply_smart_defaults, smart_default
from gage_eval.config.smart_defaults.types import (
    DefaultTrace,
    RuleContext,
    SmartDefaultRule,
    SmartDefaultsError,
    SmartDefaultsProfile,
)

__all__ = [
    "DefaultTrace",
    "RuleContext",
    "SmartDefaultRule",
    "SmartDefaultsError",
    "SmartDefaultsProfile",
    "STATIC_PHASES",
    "apply_smart_defaults",
    "select_smart_defaults_profile",
    "smart_default",
]
