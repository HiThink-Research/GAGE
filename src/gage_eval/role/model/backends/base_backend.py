"""Generic engine adaptor base extracted from llm-eval."""

from __future__ import annotations

import time
from typing import Any, Dict

from loguru import logger
from gage_eval.registry import ensure_async, run_sync


class Backend:
    """Async-friendly backend interface consumed by role adapters."""

    kind = "backends"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        # NOTE: Execution mode: `native` means local engine, `http` means remote API.
        self.execution_mode: str = self.config.get("execution_mode", "native")

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous entry point retained for backwards compatibility."""

        return run_sync(self.ainvoke(payload))

    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(payload)


class EngineBackend(Backend):
    """Base class mirroring the llm-eval EngineBackend contract."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        logger.info("Initializing backend {}", self.__class__.__name__)
        self.model = self.load_model(config)

    def load_model(self, config: Dict[str, Any]):  # pragma: no cover
        raise NotImplementedError

    def prepare_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        return payload

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.prepare_inputs(payload)
        start = time.time()
        generate = ensure_async(self.generate)
        result = await generate(inputs)
        result.setdefault("latency_ms", (time.time() - start) * 1000)
        logger.debug(
            "Backend {} finished request latency={:.2f}ms",
            self.__class__.__name__,
            result["latency_ms"],
        )
        return result
