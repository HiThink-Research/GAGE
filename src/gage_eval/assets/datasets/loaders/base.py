"""DatasetLoader base classes."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.hubs.base import DatasetHubHandle
from gage_eval.assets.datasets.manager import DataSource

if TYPE_CHECKING:  # pragma: no cover
    from gage_eval.observability.trace import ObservabilityTrace

class DatasetLoader:
    """Base loader that turns hub handles into DataSource objects."""

    loader_type = "base"

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec

    def load(
        self,
        hub_handle: Optional[DatasetHubHandle],
        *,
        trace: Optional["ObservabilityTrace"] = None,
    ) -> DataSource:  # pragma: no cover - abstract
        raise NotImplementedError
