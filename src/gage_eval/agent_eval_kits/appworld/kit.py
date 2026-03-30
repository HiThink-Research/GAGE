"""AppWorld benchmark kit definition — shell only, not yet implemented."""

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
    raise NotImplementedError("appworld.build_kit is not implemented yet")
