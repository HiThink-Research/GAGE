from __future__ import annotations

from gage_eval.reporting.evidence.external_harness.base import ExternalHarnessAdapter


class ExternalHarnessAdapterRegistry:
    def __init__(self, adapters: list[ExternalHarnessAdapter] | None = None) -> None:
        self._adapters: dict[str, ExternalHarnessAdapter] = {}
        for adapter in adapters or []:
            self.register(adapter)

    def register(self, adapter: ExternalHarnessAdapter) -> None:
        self._adapters[adapter.harness_id] = adapter

    def get(self, harness_id: str) -> ExternalHarnessAdapter | None:
        return self._adapters.get(harness_id)

    def detect(self, evidence: object) -> ExternalHarnessAdapter | None:
        for adapter in self._adapters.values():
            if adapter.detect(evidence):
                return adapter
        return None

    def all(self) -> list[ExternalHarnessAdapter]:
        return list(self._adapters.values())
