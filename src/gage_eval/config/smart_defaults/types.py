"""Core types for smart-default rule execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, NoReturn, TypeAlias

SceneName: TypeAlias = Literal["static", "agent", "game", "legacy"]


class SmartDefaultsError(ValueError):
    """Raised when smart defaults cannot materialize a config."""


@dataclass(frozen=True, slots=True)
class DefaultTrace:
    """A single smart-default mutation trace."""

    rule: str
    action: str
    path: str


@dataclass(slots=True)
class RuleContext:
    """Caller-owned execution context shared by smart-default rules."""

    source_path: Path | None
    cli_intent: Any
    scene: str
    traces: list[DefaultTrace] = field(default_factory=list)
    current_rule: str = "<unknown>"

    def fill(self, target: dict[str, Any], *, key: str, value: Any, path: str) -> None:
        """Set a key only when it is absent and record the mutation."""

        if key in target:
            return
        target[key] = value
        self.traces.append(DefaultTrace(rule=self.current_rule, action="fill", path=path))

    def migrate(
        self,
        source: dict[str, Any],
        *,
        source_key: str,
        target: dict[str, Any],
        target_key: str,
        path: str,
    ) -> None:
        """Move a value when present while preserving an explicit target value."""

        if source_key not in source:
            return
        if target_key not in target:
            target[target_key] = source[source_key]
        source.pop(source_key, None)
        self.traces.append(DefaultTrace(rule=self.current_rule, action="migrate", path=path))

    def replace_subtree(self, target: dict[str, Any], *, key: str, value: Any, path: str) -> None:
        """Replace a nested mapping value and record the mutation."""

        target[key] = value
        self.traces.append(DefaultTrace(rule=self.current_rule, action="replace_subtree", path=path))

    def fail(self, message: str, *, path: str | None = None) -> NoReturn:
        """Abort the current rule with a smart-defaults error."""

        location = f" at {path}" if path else ""
        raise SmartDefaultsError(f"{message}{location}")


@dataclass(frozen=True, slots=True)
class SmartDefaultRule:
    """A registered rule that can mutate a smart-default payload."""

    scene: SceneName
    phase: str
    priority: int
    name: str
    description: str
    apply: Callable[[dict[str, Any], RuleContext], None]


@dataclass(frozen=True, slots=True)
class SmartDefaultsProfile:
    """Scene-specific execution profile for smart-default rules."""

    scene: SceneName
    phases: tuple[str, ...]
    rules: tuple[SmartDefaultRule, ...] = ()
