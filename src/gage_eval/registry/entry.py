"""Typed registry entry definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


@dataclass(slots=True)
class RegistryEntry:
    """Metadata describing a registered asset."""

    kind: str
    name: str
    impl: str
    desc: str
    version: str = "v1"
    tags: Tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)
