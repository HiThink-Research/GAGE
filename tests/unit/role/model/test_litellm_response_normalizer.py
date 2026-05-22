from __future__ import annotations

import types

import pytest

from gage_eval.role.model.backends.litellm.response_normalizer import LiteLLMResponseNormalizer

pytestmark = pytest.mark.fast


def test_dict_response_extracts_answer_reasoning_tool_calls_finish_reason_and_usage() -> None:
    normalizer = LiteLLMResponseNormalizer()
    raw = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "final",
                    "reasoning_content": "think",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"q\":\"x\"}",
                            },
                        }
                    ],
                    "thinking_blocks": [{"type": "reasoning", "text": "provider block"}],
                },
            }
        ],
        "usage": {"completion_tokens": 3},
    }

    result = normalizer.normalize(raw, request_context={"thinking_mode": "auto"})

    assert result["answer"] == "final"
    assert result["raw_response"] == raw
    assert result["usage"] == {"completion_tokens": 3}
    assert result["reasoning_content"] == "think"
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "name": "lookup",
            "arguments": {"q": "x"},
            "raw_arguments": "{\"q\":\"x\"}",
        }
    ]
    assert result["metadata"]["thinking_effective"] == "enabled"
    assert result["metadata"]["thinking_blocks"] == [{"type": "reasoning", "text": "provider block"}]


def test_object_response_field_extraction_does_not_depend_on_dict_only() -> None:
    normalizer = LiteLLMResponseNormalizer()
    message = types.SimpleNamespace(
        content=[{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}],
        reasoning_details=[{"type": "text", "text": "object think"}],
        tool_calls=[
            types.SimpleNamespace(
                id="call_obj",
                type="function",
                function=types.SimpleNamespace(name="lookup", arguments="{\"q\":\"object\"}"),
            )
        ],
    )
    raw = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
        usage=types.SimpleNamespace(total_tokens=7),
    )

    result = normalizer.normalize(raw, request_context={"thinking_mode": "auto"})

    assert result["answer"] == "hello"
    assert result["raw_response"]["usage"] == {"total_tokens": 7}
    assert result["usage"] == {"total_tokens": 7}
    assert result["reasoning_content"] == "object think"
    assert result["tool_calls"][0]["arguments"] == {"q": "object"}
    assert result["finish_reason"] == "stop"


def test_reasoning_content_and_reasoning_details_are_both_preserved() -> None:
    normalizer = LiteLLMResponseNormalizer()
    reasoning_details = [{"type": "summary", "text": "detail trace"}]
    raw = {
        "choices": [
            {
                "message": {
                    "content": "final",
                    "reasoning_content": "structured trace",
                    "reasoning_details": reasoning_details,
                }
            }
        ]
    }

    result = normalizer.normalize(raw, request_context={"thinking_mode": "auto"})

    assert result["reasoning_content"] == "structured trace"
    assert result["reasoning_details"] == reasoning_details
    assert result["metadata"]["reasoning_source"] == "reasoning_content"
    assert result["metadata"]["thinking_effective"] == "enabled"


def test_think_tag_fallback_extracts_reasoning_and_cleans_answer() -> None:
    normalizer = LiteLLMResponseNormalizer()

    result = normalizer.normalize(
        {"choices": [{"message": {"content": "<think>hidden</think>\nvisible"}}]},
        request_context={"thinking_mode": "auto"},
    )

    assert result["answer"] == "visible"
    assert result["reasoning_content"] == "hidden"
    assert result["metadata"]["reasoning_source"] == "think_tags"


def test_invalid_tool_arguments_are_preserved_without_raising() -> None:
    normalizer = LiteLLMResponseNormalizer()
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{bad-json"},
                        }
                    ],
                }
            }
        ]
    }

    result = normalizer.normalize(raw, request_context={})

    tool_call = result["tool_calls"][0]
    assert tool_call["name"] == "lookup"
    assert tool_call["arguments"] is None
    assert tool_call["raw_arguments"] == "{bad-json"
    assert tool_call["error"] == "invalid_tool_arguments"


def test_stream_chunks_aggregate_answer_usage_finish_reason_and_tool_delta() -> None:
    normalizer = LiteLLMResponseNormalizer()
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "content": "fin",
                        "reasoning_content": "th",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_stream",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{\"q\""},
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": "al",
                        "reasoning_content": "ink",
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": ":\"x\"}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"completion_tokens": 3},
        },
    ]

    result = normalizer.normalize_stream(chunks, request_context={"thinking_mode": "auto"})

    assert result["answer"] == "final"
    assert result["reasoning_content"] == "think"
    assert result["usage"] == {"completion_tokens": 3}
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"][0]["arguments"] == {"q": "x"}
    assert result["tool_calls"][0]["raw_arguments"] == "{\"q\":\"x\"}"


def test_stream_chunks_preserve_reasoning_details() -> None:
    normalizer = LiteLLMResponseNormalizer()
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "content": "a",
                        "reasoning_details": [{"type": "text", "text": "first"}],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": "b",
                        "reasoning_details": [{"type": "text", "text": "second"}],
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    result = normalizer.normalize_stream(chunks, request_context={"thinking_mode": "auto"})

    assert result["answer"] == "ab"
    assert result["reasoning_content"] == "firstsecond"
    assert result["reasoning_details"] == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert result["metadata"]["thinking_effective"] == "enabled"


def test_thinking_mismatch_metadata_and_fail_fast() -> None:
    normalizer = LiteLLMResponseNormalizer()
    raw = {"choices": [{"message": {"content": "final", "reasoning_content": "think"}}]}

    result = normalizer.normalize(
        raw,
        request_context={
            "thinking_mode": "disabled",
            "thinking_policy": {"on_mismatch": "warn_and_continue"},
        },
    )

    assert result["metadata"]["thinking_effective"] == "enabled"
    assert result["metadata"]["thinking_mismatch"]["expected"] == "disabled"
    assert result["metadata"]["thinking_mismatch"]["observed"] == "enabled"

    with pytest.raises(ValueError, match="response_mismatch"):
        normalizer.normalize(
            raw,
            request_context={
                "thinking_mode": "disabled",
                "thinking_policy": {"on_mismatch": "fail_fast"},
            },
        )


def test_usage_reasoning_tokens_trigger_thinking_mismatch_when_disabled() -> None:
    normalizer = LiteLLMResponseNormalizer()
    raw = {
        "choices": [{"message": {"content": "final"}}],
        "usage": {
            "completion_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }

    result = normalizer.normalize(
        raw,
        request_context={
            "thinking_mode": "disabled",
            "thinking_policy": {"on_mismatch": "warn_and_continue"},
        },
    )

    assert result["metadata"]["thinking_effective"] == "enabled"
    assert result["metadata"]["thinking_mismatch"]["expected"] == "disabled"
    assert result["metadata"]["thinking_mismatch"]["observed"] == "enabled"


def test_unreadable_reasoning_details_still_trigger_thinking_mismatch_when_disabled() -> None:
    normalizer = LiteLLMResponseNormalizer()
    raw = {
        "choices": [
            {
                "message": {
                    "content": "final",
                    "reasoning_details": [{"type": "encrypted", "payload": {"bytes": 128}}],
                }
            }
        ]
    }

    result = normalizer.normalize(
        raw,
        request_context={
            "thinking_mode": "disabled",
            "thinking_policy": {"on_mismatch": "warn_and_continue"},
        },
    )

    assert result["reasoning_content"] is None
    assert result["reasoning_details"] == [{"type": "encrypted", "payload": {"bytes": 128}}]
    assert result["metadata"]["thinking_effective"] == "enabled"
    assert result["metadata"]["thinking_mismatch"]["reasoning_source"] == "reasoning_details"


def test_enabled_thinking_with_tool_calls_records_missing_thinking_blocks_without_reasoning_text() -> None:
    normalizer = LiteLLMResponseNormalizer()
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"},
                        }
                    ],
                }
            }
        ]
    }

    result = normalizer.normalize(raw, request_context={"thinking_mode": "enabled"})

    assert result["tool_calls"][0]["name"] == "lookup"
    assert result["metadata"]["thinking_blocks_missing"] is True


def test_observation_summary_redacts_sensitive_payloads_and_records_counts() -> None:
    normalizer = LiteLLMResponseNormalizer()
    large_args = "{\"secret\":\"" + ("x" * 2000) + "\"}"
    request_kwargs = {
        "model": "hosted_vllm/demo",
        "api_base": "http://127.0.0.1:8000/v1",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "sk-secret-value",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,BBBB"}},
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "arguments": large_args},
            }
        ],
        "parallel_tool_calls": False,
    }
    result = {
        "finish_reason": "stop",
        "usage": {"total_tokens": 10},
        "latency_ms": 12.3456,
        "metadata": {"thinking_effective": "disabled"},
        "tool_calls": [{"id": "call_secret", "name": "lookup", "raw_arguments": large_args}],
    }

    summary = normalizer.build_observation_summary(
        request_kwargs,
        result,
        request_context={
            "provider": "hosted_vllm",
            "route_mode": "router",
            "topology": {"kind": "local_8gpu", "replicas": 8},
            "thinking_mode": "disabled",
            "retry_owner": "gateway",
        },
    )
    rendered = repr(summary)

    assert "sk-secret-value" not in rendered
    assert "base64,AAAA" not in rendered
    assert "call_secret" not in rendered
    assert "x" * 200 not in rendered
    assert summary["provider"] == "hosted_vllm"
    assert summary["model"] == "hosted_vllm/demo"
    assert summary["api_base_hash"]
    assert summary["route_mode"] == "router"
    assert summary["topology"]["kind"] == "local_8gpu"
    assert summary["modality_counts"]["text"] == 1
    assert summary["modality_counts"]["image"] == 1
    assert summary["modality_counts"]["audio"] == 1
    assert summary["tool_count"] == 1
    assert summary["tool_call_count"] == 1
    assert summary["resolved_thinking_mode"] == "disabled"
    assert summary["effective_thinking_mode"] == "disabled"
    assert summary["usage"] == {"total_tokens": 10}
    assert summary["latency_ms"] == 12.346


def test_observation_summary_sanitizes_sensitive_usage_payloads() -> None:
    normalizer = LiteLLMResponseNormalizer()
    secret_key = "sk-usage-secret"
    data_url = "data:image/png;base64," + ("A" * 300)
    debug_payload = "debug-" + ("x" * 300)

    summary = normalizer.build_observation_summary(
        {"model": "hosted_vllm/demo", "api_base": "http://127.0.0.1:8000/v1"},
        {
            "finish_reason": "stop",
            "usage": {
                "total_tokens": 10,
                "api_key": secret_key,
                "debug": debug_payload,
                "nested": {"payload": data_url},
            },
            "metadata": {"thinking_effective": "disabled"},
        },
        request_context={"provider": "hosted_vllm"},
    )
    rendered = repr(summary)

    assert secret_key not in rendered
    assert data_url not in rendered
    assert debug_payload not in rendered
    assert summary["usage"]["total_tokens"] == 10
    assert summary["usage"]["api_key"] == "[REDACTED]"
    assert summary["usage"]["debug"]["chars"] == len(debug_payload)
    assert summary["usage"]["nested"]["payload"]["kind"] == "data_url"
