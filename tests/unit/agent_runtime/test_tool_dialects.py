from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.agent_runtime.tooling.contracts import ToolingError
from gage_eval.agent_runtime.tooling.provider_adapters import OpenAIProviderAdapter, Tau2ToolDialectParser


def test_provider_adapter_implements_five_responsibilities() -> None:
    adapter = OpenAIProviderAdapter()

    assert callable(adapter.serialize_request)
    assert callable(adapter.capture_raw_response)
    assert callable(adapter.extract_tool_calls)
    assert callable(adapter.serialize_tool_result)
    assert callable(adapter.extract_final_answer)


def test_provider_raw_response_parse_error_maps_failure_code() -> None:
    adapter = OpenAIProviderAdapter()

    with pytest.raises(ToolingError) as excinfo:
        adapter.extract_tool_calls("<not-json>", turn_index=1)

    assert excinfo.value.code == "client_execution.tool_protocol_parse_error"


def test_required_tool_missing_maps_missing_call() -> None:
    adapter = OpenAIProviderAdapter()

    with pytest.raises(ToolingError) as excinfo:
        adapter.extract_tool_calls({"choices": [{"message": {"content": "no call"}}]}, turn_index=1, required=True)

    assert excinfo.value.code == "client_execution.tool_protocol_missing_call"


def test_tool_call_without_call_id_maps_missing_call_id() -> None:
    adapter = OpenAIProviderAdapter()

    with pytest.raises(ToolingError) as excinfo:
        adapter.extract_tool_calls(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            },
            turn_index=1,
            require_call_id=True,
        )

    assert excinfo.value.code == "client_execution.tool_protocol_missing_call_id"


def test_openai_adapter_unwraps_openai_http_backend_wrapper() -> None:
    adapter = OpenAIProviderAdapter()

    calls = adapter.extract_tool_calls(
        {
            "answer": "",
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c1", "function": {"name": "lookup", "arguments": '{"q":"refund"}'}}
                            ]
                        }
                    }
                ]
            },
            "usage": {},
        },
        turn_index=1,
    )

    assert [call.name for call in calls] == ["lookup"]


def test_openai_adapter_unwraps_litellm_modelresponse_dict_wrapper() -> None:
    adapter = OpenAIProviderAdapter()

    calls = adapter.extract_tool_calls(
        {
            "answer": "",
            "raw_response": {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c1", "function": {"name": "respond", "arguments": '{"message":"done"}'}}
                            ]
                        }
                    }
                ],
                "_response_ms": 123,
            },
        },
        turn_index=2,
    )

    assert [call.name for call in calls] == ["respond"]


def test_openai_adapter_handles_pydantic_modelresponse_directly() -> None:
    adapter = OpenAIProviderAdapter()
    response = SimpleNamespace(
        model_dump=lambda mode=None: {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "lookup", "arguments": '{"q":"refund"}'}}
                        ]
                    }
                }
            ]
        }
    )

    calls = adapter.extract_tool_calls(response, turn_index=3)

    assert [call.name for call in calls] == ["lookup"]


@pytest.mark.parametrize(
    ("dialect", "message", "expected_name", "expected_arguments"),
    [
        (
            "openai_like",
            {"tool_calls": [{"id": "c1", "function": {"name": "lookup", "arguments": '{"q":"refund"}'}}]},
            "lookup",
            '{"q":"refund"}',
        ),
        (
            "openai_like",
            {"function_call": {"name": "lookup", "arguments": '{"q":"refund"}'}},
            "lookup",
            '{"q":"refund"}',
        ),
        (
            "qwen_xml",
            "<tool_call><arg_key>message</arg_key><arg_value>hello</arg_value></tool_call>",
            "respond",
            '{"message":"hello"}',
        ),
        (
            "qwen_arg_key",
            "<arg_key>message</arg_key><arg_value>hello</arg_value>",
            "respond",
            '{"message":"hello"}',
        ),
        (
            "glm_arg_key",
            "<arg_key>message</arg_key><arg_value>hello</arg_value>",
            "respond",
            '{"message":"hello"}',
        ),
        (
            "minimax",
            '<minimax:tool_call><invoke name="respond"><arg name="message">hello</arg></invoke></minimax:tool_call>',
            "respond",
            '{"message":"hello"}',
        ),
        (
            "gemma",
            'call:respond{message:"hello"}',
            "respond",
            '{"message":"hello"}',
        ),
        (
            "fenced_json",
            '```json\n{"tool_calls": [{"name": "lookup", "arguments": {"q": "refund"}}]}\n```',
            "lookup",
            '{"q":"refund"}',
        ),
        (
            "raw_json",
            '{"tool_calls":[{"name":"lookup","arguments":{"q":"refund"}}]}',
            "lookup",
            '{"q":"refund"}',
        ),
        (
            "function_gemma",
            '<|tool_call>call:respond{message:"hello"}<tool_call|>',
            "respond",
            '{"message":"hello"}',
        ),
        (
            "raw_json",
            '{"name":"lookup","arguments":{"q":"refund"}}',
            "lookup",
            '{"q":"refund"}',
        ),
        (
            "pythonic_call",
            'lookup(q="refund", limit=2)',
            "lookup",
            '{"limit":2,"q":"refund"}',
        ),
    ],
)
def test_tau2patch_tool_dialect_fixtures_normalize_to_tool_call_ir(
    dialect: str,
    message,
    expected_name: str,
    expected_arguments: str,
) -> None:
    calls = Tau2ToolDialectParser().parse(message, dialect=dialect, turn_index=2, call_index=1)

    assert len(calls) == 1
    call = calls[0]
    assert call.call_id == "call_2_1" or call.call_id == "c1"
    assert call.name == expected_name
    assert call.arguments_json == expected_arguments


@pytest.mark.parametrize("message", ["respond{message:\"hello\"}", "call:respond{}"])
def test_gemma_bare_call_forms_normalize_to_tool_call_ir(message: str) -> None:
    calls = Tau2ToolDialectParser().parse(message, dialect="gemma", turn_index=3, call_index=1)

    assert calls[0].name == "respond"


def test_openai_like_dialect_returns_all_provider_tool_calls() -> None:
    calls = Tau2ToolDialectParser().parse(
        {
            "tool_calls": [
                {"id": "c1", "function": {"name": "lookup", "arguments": '{"q":"refund"}'}},
                {"id": "c2", "function": {"name": "respond", "arguments": '{"message":"done"}'}},
            ]
        },
        dialect="openai_like",
        turn_index=4,
    )

    assert [call.call_id for call in calls] == ["c1", "c2"]
    assert [call.name for call in calls] == ["lookup", "respond"]
    assert [call.arguments_json for call in calls] == ['{"q":"refund"}', '{"message":"done"}']


@pytest.mark.parametrize(
    ("dialect", "message"),
    [
        (
            "raw_json",
            '{"tool_calls":[{"name":"lookup","arguments":{"q":"refund"}},{"name":"respond","arguments":{"message":"done"}}]}',
        ),
        (
            "fenced_json",
            '```json\n{"tool_calls":[{"name":"lookup","arguments":{"q":"refund"}},{"name":"respond","arguments":{"message":"done"}}]}\n```',
        ),
    ],
)
def test_json_dialects_return_all_tool_calls(dialect: str, message: str) -> None:
    calls = Tau2ToolDialectParser().parse(message, dialect=dialect, turn_index=5)

    assert [call.call_id for call in calls] == ["call_5_1", "call_5_2"]
    assert [call.name for call in calls] == ["lookup", "respond"]
    assert [call.arguments_json for call in calls] == ['{"q":"refund"}', '{"message":"done"}']


def test_auto_dialect_recognizes_function_gemma_tool_call_tag() -> None:
    calls = Tau2ToolDialectParser().parse(
        '<|tool_call>call:respond{message:"hello"}<tool_call|>',
        dialect="auto",
        turn_index=6,
    )

    assert len(calls) == 1
    assert calls[0].name == "respond"
    assert calls[0].arguments_json == '{"message":"hello"}'


def test_plain_text_fallback_returns_no_tool_call() -> None:
    assert Tau2ToolDialectParser().parse("I can help with that.", dialect="plain_text", turn_index=1) == []
