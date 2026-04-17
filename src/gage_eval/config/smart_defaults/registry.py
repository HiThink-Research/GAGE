"""Registration and execution helpers for smart-default rules."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from gage_eval.config.smart_defaults.types import (
    RuleContext,
    SceneName,
    SmartDefaultRule,
    SmartDefaultsError,
    SmartDefaultsProfile,
)

_RULES: list[SmartDefaultRule] = []


def _rule_key(rule: SmartDefaultRule) -> tuple[SceneName, str, str]:
    return (rule.scene, rule.phase, rule.name)


def smart_default(
    *,
    scene: SceneName,
    phase: str,
    priority: int = 0,
    name: str | None = None,
    description: str,
) -> Callable[[Callable[[dict[str, Any], RuleContext], None]], Callable[[dict[str, Any], RuleContext], None]]:
    """Register a smart-default rule for a scene and phase."""

    def decorator(func: Callable[[dict[str, Any], RuleContext], None]) -> Callable[[dict[str, Any], RuleContext], None]:
        rule = SmartDefaultRule(
            scene=scene,
            phase=phase,
            priority=priority,
            name=name or func.__name__,
            description=description,
            apply=func,
        )
        rule_key = _rule_key(rule)
        for index, existing_rule in enumerate(_RULES):
            if _rule_key(existing_rule) == rule_key:
                _RULES[index] = rule
                _RULES[:] = [
                    candidate
                    for candidate_index, candidate in enumerate(_RULES)
                    if candidate_index <= index or _rule_key(candidate) != rule_key
                ]
                return func
        _RULES.append(rule)
        return func

    return decorator


def registered_rules(scene: SceneName | None = None) -> tuple[SmartDefaultRule, ...]:
    """Return registered rules, optionally filtered by scene."""

    rules = _RULES if scene is None else [rule for rule in _RULES if rule.scene == scene]
    return tuple(sorted(rules, key=lambda rule: (rule.priority, rule.name)))


def apply_smart_defaults(payload: dict[str, Any], ctx: RuleContext, profile: SmartDefaultsProfile) -> dict[str, Any]:
    """Apply smart-default rules to a payload copy using the caller's context."""

    expanded = deepcopy(payload)
    rules_by_phase: dict[str, list[SmartDefaultRule]] = {}
    for rule in profile.rules:
        rules_by_phase.setdefault(rule.phase, []).append(rule)

    for phase in profile.phases:
        phase_rules = sorted(rules_by_phase.get(phase, ()), key=lambda rule: (rule.priority, rule.name))
        for rule in phase_rules:
            ctx.current_rule = rule.name
            try:
                rule.apply(expanded, ctx)
            except SmartDefaultsError:
                raise
            except Exception as exc:  # pragma: no cover - defensive wrapper
                raise SmartDefaultsError(str(exc)) from exc
    return expanded
