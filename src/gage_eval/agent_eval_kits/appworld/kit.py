"""AppWorld benchmark kit definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BenchmarkKitDefinition:
    """Static metadata for an AppWorld benchmark kit."""

    kit_id: str
    verifier_kind: Literal["judge_adapter", "native"]
    required_surfaces: tuple[str, ...] = ()
    optional_surfaces: tuple[str, ...] = ()


def build_kit() -> BenchmarkKitDefinition:
    """Build the AppWorld kit definition."""

    return BenchmarkKitDefinition(
        kit_id="appworld",
        verifier_kind="judge_adapter",
        required_surfaces=("terminal", "fs", "env", "api"),
        optional_surfaces=("mcp",),
    )
