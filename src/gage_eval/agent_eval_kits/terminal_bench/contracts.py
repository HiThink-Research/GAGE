"""Terminal benchmark contracts and shared constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

TERMINAL_BENCH_KIT_ID = "terminal_bench"
TERMINAL_BENCH_REQUIRED_SURFACES = ("terminal", "fs")
TERMINAL_BENCH_DEFAULT_TIMEOUT_SEC = 300


@dataclass(frozen=True)
class BenchmarkKitDefinition:
    """Static metadata for a benchmark kit."""

    kit_id: str
    verifier_kind: Literal["judge_adapter", "native"]
    required_surfaces: tuple[str, ...] = ()
    optional_surfaces: tuple[str, ...] = ()


@dataclass(frozen=True)
class TerminalBenchTaskContext:
    """Normalized task context for terminal benchmark execution."""

    sample_id: str
    instruction: str
    workspace_root: Optional[str] = None
    required_surfaces: tuple[str, ...] = TERMINAL_BENCH_REQUIRED_SURFACES
    metadata: Dict[str, Any] = field(default_factory=dict)
