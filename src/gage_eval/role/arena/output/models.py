from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from gage_eval.role.arena.core.types import ArenaSample


@dataclass(frozen=True)
class ArenaOutput:
    sample: ArenaSample
    tick: int
    step: int
    result: object | None = None
    arena_trace: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    header: Mapping[str, object] | None = None
    trace: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    footer: Mapping[str, object] | None = None
    resource_artifacts: Mapping[str, object] | None = None
    game_context: Mapping[str, object] | None = None
    artifacts: Mapping[str, object] | None = None
    output_kind: str = "arena"
