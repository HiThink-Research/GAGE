from __future__ import annotations

import asyncio

import pytest

from gage_eval.role.adapters.dut_model import DUTModelAdapter
from gage_eval.role.model.backends import wrap_backend
from gage_eval.role.model.backends.base_backend import Backend


class _ExplodingBackend(Backend):
    async def ainvoke(self, payload: dict) -> dict:
        raise RuntimeError("boom")


class _FlakyHTTPBackend(Backend):
    def __init__(self, config: dict) -> None:
        self.calls = 0
        self.http_retry_mode = "wrapper"
        self.transport = "http"
        self.http_retry_params = {"attempts": 2, "interval": 0.0}
        super().__init__(config)

    async def ainvoke(self, payload: dict) -> dict:
        self.calls += 1
        if self.calls < 2:
            raise RuntimeError("temporary upstream error")
        return {"answer": "ok"}


class _ExplodingBatchBackend(Backend):
    async def ainvoke(self, payload: dict) -> dict:
        raise AssertionError("batch path should not call ainvoke")

    def generate_batch(self, payloads: list[dict]) -> list[dict]:
        raise RuntimeError("batch boom")


@pytest.mark.fast
def test_wrap_backend_normalizes_exceptions_into_error_payload() -> None:
    wrapped = wrap_backend(_ExplodingBackend({}))

    result = asyncio.run(wrapped.ainvoke({}))

    assert result == {
        "error": "boom",
        "status": None,
        "error_type": "RuntimeError",
        "backend": "_ExplodingBackend",
    }


@pytest.mark.fast
def test_wrap_backend_preserves_http_retries_before_normalizing_errors() -> None:
    backend = _FlakyHTTPBackend({})
    wrapped = wrap_backend(backend)

    result = asyncio.run(wrapped.ainvoke({}))

    assert result["answer"] == "ok"
    assert backend.calls == 2


@pytest.mark.fast
def test_wrap_backend_normalizes_generate_batch_failures() -> None:
    wrapped = wrap_backend(_ExplodingBatchBackend({}))

    result = wrapped.generate_batch([{"prompt": "a"}, {"prompt": "b"}])

    assert result == [
        {
            "error": "batch boom",
            "status": None,
            "error_type": "RuntimeError",
            "backend": "_ExplodingBatchBackend",
        },
        {
            "error": "batch boom",
            "status": None,
            "error_type": "RuntimeError",
            "backend": "_ExplodingBatchBackend",
        },
    ]


@pytest.mark.fast
def test_dut_model_adapter_wraps_backend_instances_with_normalized_error_contract() -> None:
    adapter = DUTModelAdapter(
        adapter_id="dut-adapter",
        role_type="dut_model",
        backend=_ExplodingBackend({}),
        capabilities=(),
    )

    result = adapter.invoke({"sample": {"id": "sample-1", "text": "hello"}}, adapter.clone_for_sample())

    assert result["error"] == "boom"
    assert result["status"] is None
    assert result["error_type"] == "RuntimeError"
    assert result["backend"] == "_ExplodingBackend"
