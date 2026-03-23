from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import gage_eval.role.model.backends.openai_http_backend as openai_http_backend
from gage_eval.role.model.backends.openai_http_backend import OpenAICompatibleHTTPBackend


class _SyncCreate:
    def __init__(self, response: Any) -> None:
        self._response = response

    def create(self, **_: Any) -> Any:
        return self._response


class _AsyncCreate:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def create(self, **_: Any) -> Any:
        return self._response


class _SyncStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __iter__(self):
        return iter(self._events)


class _AsyncStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __aiter__(self):
        async def _iterator():
            for event in self._events:
                yield event

        return _iterator()


def _build_backend() -> OpenAICompatibleHTTPBackend:
    backend = OpenAICompatibleHTTPBackend.__new__(OpenAICompatibleHTTPBackend)
    backend.model_name = "fake-model"
    backend.default_params = {}
    backend.tool_choice_default = None
    backend.stream = False
    backend._reasoning_effort = None
    backend._async_client = None
    backend._async_semaphore = None
    backend.prepare_inputs = lambda payload: payload
    return backend


def _completion_payload(answer: str, *, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "cmpl_123",
        "object": "chat.completion",
        "created": 123,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": answer},
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _stream_chunk(
    content: str,
    *,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "cmpl_123",
        "object": "chat.completion.chunk",
        "created": 123,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "delta": {"role": "assistant", "content": content},
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


@pytest.mark.fast
def test_generate_accepts_completion_like_dict_response() -> None:
    backend = _build_backend()
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=_SyncCreate(_completion_payload("hello"))))

    result = backend.generate({"stream": False})

    assert result["answer"] == "hello"
    assert result["raw_response"]["choices"][0]["message"]["content"] == "hello"
    assert result["usage"] is None


@pytest.mark.fast
def test_generate_collects_sync_stream_into_completion_payload() -> None:
    backend = _build_backend()
    stream = _SyncStream(
        [
            _stream_chunk("hello "),
            _stream_chunk("world", finish_reason="stop", usage={"prompt_tokens": 3, "completion_tokens": 2}),
        ]
    )
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=_SyncCreate(stream)))

    result = backend.generate({"stream": True})

    assert result["answer"] == "hello world"
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert result["raw_response"]["object"] == "chat.completion"
    assert result["raw_response"]["choices"][0]["message"]["content"] == "hello world"
    assert result["raw_response"]["stream_chunk_count"] == 2


@pytest.mark.fast
def test_ainvoke_accepts_completion_like_dict_response() -> None:
    backend = _build_backend()
    backend._async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_AsyncCreate(_completion_payload("hello async", usage={"total_tokens": 5})))
    )

    result = asyncio.run(backend.ainvoke({"stream": False}))

    assert result["answer"] == "hello async"
    assert result["usage"] == {"total_tokens": 5}
    assert result["raw_response"]["choices"][0]["message"]["content"] == "hello async"
    assert result["latency_ms"] >= 0


@pytest.mark.fast
def test_ainvoke_collects_async_stream_into_completion_payload() -> None:
    backend = _build_backend()
    stream = _AsyncStream(
        [
            _stream_chunk("hello "),
            _stream_chunk("async", finish_reason="stop", usage={"prompt_tokens": 4, "completion_tokens": 2}),
        ]
    )
    backend._async_client = SimpleNamespace(chat=SimpleNamespace(completions=_AsyncCreate(stream)))

    result = asyncio.run(backend.ainvoke({"stream": True}))

    assert result["answer"] == "hello async"
    assert result["usage"] == {"prompt_tokens": 4, "completion_tokens": 2}
    assert result["raw_response"]["object"] == "chat.completion"
    assert result["raw_response"]["choices"][0]["message"]["content"] == "hello async"
    assert result["raw_response"]["stream_chunk_count"] == 2


@pytest.mark.fast
def test_generate_warns_and_falls_back_for_invalid_stream_indexes() -> None:
    backend = _build_backend()
    stream = _SyncStream(
        [
            {
                "id": "cmpl_123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "fake-model",
                "choices": [
                    {
                        "index": "bad-index",
                        "delta": {"role": "assistant", "content": "hello"},
                    }
                ],
            }
        ]
    )
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=_SyncCreate(stream)))

    with patch.object(openai_http_backend.logger, "warning") as warning_mock:
        result = backend.generate({"stream": True})

    assert result["answer"] == "hello"
    warning_mock.assert_called_once()
    assert "invalid stream index" in warning_mock.call_args.args[0]
