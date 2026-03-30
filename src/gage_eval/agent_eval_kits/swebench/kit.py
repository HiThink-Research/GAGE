"""SWE-bench benchmark kit definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BenchmarkKitDefinition:
    """Static benchmark kit metadata."""

    kit_id: str
    verifier_kind: Literal["judge_adapter", "native"]
    required_surfaces: tuple[str, ...] = ()
    optional_surfaces: tuple[str, ...] = ()


def build_kit() -> BenchmarkKitDefinition:
    """Build the SWE-bench kit definition."""
    return BenchmarkKitDefinition(
        kit_id="swebench",
        verifier_kind="judge_adapter",
        required_surfaces=("terminal", "fs"),
        optional_surfaces=(),
    )
