"""Resource profile models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence


@dataclass
class NodeResource:
    node_id: str
    gpus: int = 1
    cpus: int = 1
    bandwidth: int = 0
    affinity_tags: Sequence[str] = field(default_factory=tuple)


@dataclass
class ResourceProfile:
    nodes: Sequence[NodeResource]
