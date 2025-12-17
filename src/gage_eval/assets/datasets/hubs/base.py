"""Dataset hub abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gage_eval.config.pipeline_config import DatasetSpec


@dataclass
class DatasetHubHandle:
    """Materialized reference returned by DatasetHub implementations."""

    hub_id: str
    resource: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    streaming: bool = False


class DatasetHub:
    """Base class for remote/local dataset hubs."""

    hub_type = "default"

    def __init__(self, spec: DatasetSpec, *, hub_args: Optional[Dict[str, Any]] = None) -> None:
        self.spec = spec
        self.hub_args = hub_args or {}

    def resolve(self) -> DatasetHubHandle:  # pragma: no cover - abstract
        raise NotImplementedError
