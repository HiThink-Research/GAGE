"""Deterministic dummy backend for smoke tests."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.config.dummy import DummyBackendConfig


@registry.asset(
    "backends",
    "dummy",
    desc="Deterministic dummy backend (tests)",
    tags=("test", "dummy"),
    modalities=("text",),
    config_schema_ref="gage_eval.role.model.config.dummy:DummyBackendConfig",
)
class DummyBackend(EngineBackend):
    def load_model(self, config: Dict[str, Any]):
        self.config = DummyBackendConfig(**config)
        self._responses = list(self.config.responses or ["dummy response"])
        self._index = 0
        self._rng: Optional[random.Random] = None
        if self.config.random:
            self._rng = random.Random(self.config.seed)

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._responses:
            if self._rng:
                answer = self._rng.choice(self._responses)
            else:
                answer = self._responses[self._index]
                if self.config.cycle:
                    self._index = (self._index + 1) % len(self._responses)
                else:
                    self._index = min(self._index + 1, len(self._responses) - 1)
        elif self.config.echo_prompt:
            answer = inputs.get("prompt") or ""
        else:
            answer = ""
        payload = {"answer": answer}
        if self.config.metadata:
            payload["metadata"] = self.config.metadata
        return payload
