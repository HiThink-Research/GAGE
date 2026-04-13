"""DatasetLoader base classes."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.hubs.base import DatasetHubHandle
from gage_eval.assets.datasets.manager import DataSource
from gage_eval.registry import registry

if TYPE_CHECKING:  # pragma: no cover
    from gage_eval.observability.trace import ObservabilityTrace
    from gage_eval.registry.runtime import RegistryLookup

class DatasetLoader:
    """Base loader that turns hub handles into DataSource objects."""

    loader_type = "base"

    def __init__(
        self,
        spec: DatasetSpec,
        *,
        registry_view: Optional["RegistryLookup"] = None,
    ) -> None:
        self.spec = spec
        self.registry_view = registry_view

    @property
    def registry_lookup(self):
        return self.registry_view or registry

    @property
    def allow_asset_lazy_import(self) -> bool:
        return self.registry_view is None

    def load(
        self,
        hub_handle: Optional[DatasetHubHandle],
        *,
        trace: Optional["ObservabilityTrace"] = None,
    ) -> DataSource:  # pragma: no cover - abstract
        raise NotImplementedError
