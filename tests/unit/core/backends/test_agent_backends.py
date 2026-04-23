from __future__ import annotations

import asyncio
import json
import shlex
import sys
import threading

import pytest

from gage_eval.role.agent.backends.class_backend import ClassBackend
from gage_eval.role.agent.backends.cli_backend import CliBackend
from gage_eval.role.agent.backends.http_backend import HttpBackend
from gage_eval.role.agent.backends.model_backend import ModelBackend
from gage_eval.role.agent.backends.mcp_backend import McpBackend


class DummyAgent:
    def run(self, payload):
        return {"answer": payload.get("text", "ok"), "agent_trace": []}


class DummyAgentString:
    def run(self, payload):
        return "raw-answer"


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class DummyModelBackend:
    def invoke(self, payload):
        return {
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "t1",
                                    "type": "function",
                                    "function": {"name": "run", "arguments": "{\"x\": 1}"},
                                }
                            ]
                        }
                    }
                ]
            }
        }


class AsyncDummyModelBackend:
    def __init__(self) -> None:
        self.thread_id: int | None = None

    async def ainvoke(self, payload):
        self.thread_id = threading.get_ident()
        return {
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "t1",
                                    "type": "function",
                                    "function": {"name": "run", "arguments": "{\"x\": 1}"},
                                }
                            ]
                        }
                    }
                ]
            }
        }


def _python_command(script: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


@pytest.mark.fast
def test_class_backend_dict_output():
    backend = ClassBackend(
        {
            "agent_class": "tests.unit.core.backends.test_agent_backends:DummyAgent",
            "method": "run",
        }
    )
    result = backend.invoke({"text": "hello"})
    assert result["answer"] == "hello"
    assert result["agent_trace"] == []


@pytest.mark.fast
def test_class_backend_string_output():
    backend = ClassBackend(
        {
            "agent_class": "tests.unit.core.backends.test_agent_backends:DummyAgentString",
            "method": "run",
        }
    )
    result = backend.invoke({})
    assert result["answer"] == "raw-answer"
    assert result["agent_trace"] == []


@pytest.mark.io
def test_cli_backend_raw_stdout():
    backend = CliBackend({"command": _python_command('print("hello")')})
    result = backend.invoke({})
    assert result["answer"] == "hello"


@pytest.mark.fast
def test_cli_backend_returns_error_payload_on_non_zero_exit() -> None:
    backend = CliBackend(
        {
            "command": _python_command(
                'import sys; print("partial"); sys.stderr.write("boom\\n"); sys.exit(7)'
            )
        }
    )

    result = backend.invoke({})

    assert result["error"] == "boom"
    assert result["status"] == 7
    assert result["error_type"] == "cli_process_error"
    assert result["backend"] == "CliBackend"
    assert result["returncode"] == 7
    assert result["answer"] == ""
    assert result["raw_stdout"] == "partial\n"
    assert result["raw_stderr"] == "boom\n"


@pytest.mark.fast
def test_cli_backend_ignores_output_file_when_process_fails(tmp_path) -> None:
    output_path = tmp_path / "result.json"
    output_path.write_text(json.dumps({"answer": "stale-success"}), encoding="utf-8")
    backend = CliBackend(
        {
            "command": _python_command('import sys; sys.stderr.write("failed\\n"); sys.exit(9)'),
            "output_path": str(output_path),
        }
    )

    result = backend.invoke({})

    assert result["error"] == "failed"
    assert result["status"] == 9
    assert result["returncode"] == 9
    assert result["answer"] == ""


@pytest.mark.fast
def test_http_backend_openai_schema(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": "hi",
                    "tool_calls": [{"id": "t1", "function": {"name": "run", "arguments": "{}"}}],
                }
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }

    def fake_post(*args, **kwargs):
        return DummyResponse(payload)

    monkeypatch.setattr("requests.post", fake_post)
    backend = HttpBackend({"endpoint": "http://example.com", "schema": "openai_chat"})
    result = backend.invoke({"messages": []})
    assert result["answer"] == "hi"
    assert result["tool_calls"][0]["id"] == "t1"
    assert result["usage"]["input_tokens"] == 1


@pytest.mark.fast
def test_mcp_backend_basic(monkeypatch):
    def fake_post(*args, **kwargs):
        return DummyResponse({"answer": "ok"})

    monkeypatch.setattr("requests.post", fake_post)
    backend = McpBackend({"endpoint": "http://example.com", "transport": "http_sse"})
    result = backend.invoke({})
    assert result["answer"] == "ok"
    assert result["transport"] == "http_sse"


@pytest.mark.fast
def test_model_backend_extracts_tool_calls():
    backend = ModelBackend({"backend": DummyModelBackend()})
    result = backend.invoke({"messages": []})
    assert result["tool_calls"][0]["function"]["name"] == "run"


@pytest.mark.fast
def test_model_backend_extracts_tagged_text_tool_call():
    class TaggedToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<tool_call>\n"
                    '{"name": "respond", "arguments": {"message": "hello"}}\n'
                    "</tool_call>"
                )
            }

    backend = ModelBackend({"backend": TaggedToolCallBackend()})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert "raw_answer" in result
    assert result["tool_calls"][0]["type"] == "function"
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {"message": "hello"}


@pytest.mark.fast
def test_model_backend_extracts_tool_calls_json_array_marker():
    class ArrayToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "[TOOL_CALLS]\n"
                    '[{"name": "set_network_mode_preference", "arguments": {"mode": "5G"}}]'
                )
            }

    backend = ModelBackend({"backend": ArrayToolCallBackend()})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "set_network_mode_preference"
    assert result["tool_calls"][0]["function"]["arguments"] == {"mode": "5G"}


@pytest.mark.fast
def test_model_backend_extracts_qwen_xml_function_call():
    class XmlToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<tool_call>\n"
                    "<function=respond>\n"
                    "<parameter=message>Done, your mobile data is working.</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                )
            }

    backend = ModelBackend({"backend": XmlToolCallBackend()})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "Done, your mobile data is working."
    }


@pytest.mark.fast
def test_model_backend_extracts_qwen_json_parameter_values():
    class QwenToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<tool_call>\n"
                    "<function=update_preferences>\n"
                    "<parameter=enabled>\ntrue\n</parameter>\n"
                    '<parameter=filters>\n{"plan": "premium", "tags": ["5g", "roaming"]}\n</parameter>\n'
                    "</function>\n"
                    "</tool_call>"
                )
            }

    backend = ModelBackend({"backend": QwenToolCallBackend(), "tool_format": "qwen"})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "update_preferences"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "enabled": True,
        "filters": {"plan": "premium", "tags": ["5g", "roaming"]},
    }


@pytest.mark.fast
def test_model_backend_infers_generic_qwen3_tool_format():
    class Qwen3ToolCallBackend:
        config = {"model": "Qwen/Qwen3-235B-A22B"}

        def invoke(self, payload):
            return {
                "answer": (
                    "<tool_call>\n"
                    "<function=respond>\n"
                    "<parameter=message>ok</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                )
            }

    backend = ModelBackend({"backend": Qwen3ToolCallBackend()})
    result = backend.invoke({"messages": []})

    assert backend._tool_call_format == "qwen"
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {"message": "ok"}


@pytest.mark.fast
def test_model_backend_infers_qwen_tool_format_for_qwen2_and_bare_qwen_names():
    class Qwen2Backend:
        config = {"model": "Qwen/Qwen2.5-7B-Instruct"}

        def invoke(self, payload):
            return {"answer": "done"}

    class BareQwenBackend:
        config = {"model_path": "/mnt/model/qwen-omni"}

        def invoke(self, payload):
            return {"answer": "done"}

    assert ModelBackend({"backend": Qwen2Backend()})._tool_call_format == "qwen"
    assert ModelBackend({"backend": BareQwenBackend()})._tool_call_format == "qwen"


@pytest.mark.fast
def test_model_backend_extracts_minimax_tool_call():
    class MiniMaxToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<minimax:tool_call>\n"
                    '<invoke name="respond">'
                    '<parameter name="message">Done, your mobile data is working.</parameter>'
                    '<parameter name="urgent">false</parameter>'
                    "</invoke>\n"
                    "</minimax:tool_call>"
                )
            }

    backend = ModelBackend({"backend": MiniMaxToolCallBackend()})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "Done, your mobile data is working.",
        "urgent": False,
    }


@pytest.mark.fast
def test_model_backend_extracts_minimax_json_parameter_values():
    class MiniMaxToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<minimax:tool_call>\n"
                    '<invoke name="set_filters">'
                    '<parameter name="filters">{"network": "5G", "limits": [1, 2]}</parameter>'
                    "</invoke>\n"
                    "</minimax:tool_call>"
                )
            }

    backend = ModelBackend({"backend": MiniMaxToolCallBackend(), "tool_format": "minimax"})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "set_filters"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "filters": {"network": "5G", "limits": [1, 2]}
    }


@pytest.mark.fast
def test_model_backend_extracts_minimax_truncated_tool_call():
    class MiniMaxToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<think>Need a tool.</think>\n"
                    "<minimax:tool_call>\n"
                    '<invoke\n  name="respond"\n>'
                    '<parameter\n name="message"\n>Done, your mobile data is working.</parameter>'
                    "</invoke>"
                )
            }

    backend = ModelBackend({"backend": MiniMaxToolCallBackend(), "tool_format": "minimax"})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "Done, your mobile data is working."
    }


@pytest.mark.fast
def test_model_backend_extracts_minimax_bare_invoke_with_loose_attributes():
    class MiniMaxToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<think>Need a tool.</think>\n"
                    "<invoke name='respond'>"
                    "<parameter name=message>Hello from MiniMax.</parameter>"
                    "<parameter name='urgent'>false</parameter>"
                    "</invoke>"
                )
            }

    backend = ModelBackend({"backend": MiniMaxToolCallBackend(), "tool_format": "minimax-m2.5"})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "Hello from MiniMax.",
        "urgent": False,
    }


@pytest.mark.fast
def test_model_backend_extracts_gemma4_tool_call():
    class Gemma4ToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<|channel>thought\n"
                    "Need a tool.\n"
                    "<channel|><|tool_call>"
                    'call:get_current_weather{location:<|"|>London<|"|>,units:<|"|>celsius<|"|>,days:2}'
                    "<tool_call|><|tool_response>"
                )
            }

    backend = ModelBackend({"backend": Gemma4ToolCallBackend()})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "get_current_weather"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "location": "London",
        "units": "celsius",
        "days": 2,
    }


@pytest.mark.fast
def test_model_backend_extracts_gemma4_bare_tool_call():
    class Gemma4ToolCallBackend:
        config = {"model_path": "/mnt/models/gemma_4_27b_it"}

        def invoke(self, payload):
            return {
                "answer": (
                    "call:respond{message:Hello Emma Kim! I can help with that, "
                    "and I will first check your booking.}"
                )
            }

    backend = ModelBackend({"backend": Gemma4ToolCallBackend()})
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "respond", "parameters": {}},
                }
            ],
        }
    )

    assert backend._tool_call_format == "gemma4"
    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "Hello Emma Kim! I can help with that, and I will first check your booking."
    }


@pytest.mark.fast
def test_model_backend_extracts_gemma4_nested_tool_call_arguments():
    class Gemma4ToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<|tool_call>"
                    "call:search_records{"
                    'query:<|"|>alpha,beta {literal}<|"|>,'
                    "filters:{enabled:true,tags:[<|\"|>x,y<|\"|>,<|\"|>z<|\"|>]},"
                    "limit:3"
                    "}"
                    "<tool_call|><|tool_response>"
                )
            }

    backend = ModelBackend({"backend": Gemma4ToolCallBackend(), "tool_format": "gemma4"})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "search_records"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "query": "alpha,beta {literal}",
        "filters": {"enabled": True, "tags": ["x,y", "z"]},
        "limit": 3,
    }


@pytest.mark.fast
def test_model_backend_infers_functiongemma_tool_format():
    class FunctionGemmaBackend:
        config = {"model": "google/functiongemma-270m-it"}

        def invoke(self, payload):
            return {"answer": "call:respond{message:Hello.}"}

    backend = ModelBackend({"backend": FunctionGemmaBackend()})
    result = backend.invoke({"messages": []})

    assert backend._tool_call_format == "gemma4"
    assert result["tool_calls"][0]["function"]["name"] == "respond"


@pytest.mark.fast
def test_model_backend_extracts_named_xml_function_call():
    class NamedXmlToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<function_calls>\n"
                    '<function name="toggle_airplane_mode">\n'
                    '<parameter name="enabled">false</parameter>\n'
                    "</function>\n"
                    "</function_calls>"
                )
            }

    backend = ModelBackend({"backend": NamedXmlToolCallBackend()})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "toggle_airplane_mode"
    assert result["tool_calls"][0]["function"]["arguments"] == {"enabled": False}


@pytest.mark.fast
def test_model_backend_extracts_pythonic_function_call():
    class PythonicToolCallBackend:
        def invoke(self, payload):
            return {"answer": "<function_calls>\nrespond(message='All set')\n</function_calls>"}

    backend = ModelBackend({"backend": PythonicToolCallBackend()})
    result = backend.invoke({"messages": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {"message": "All set"}


@pytest.mark.fast
def test_model_backend_extracts_bare_pythonic_function_call_after_think_tail():
    class PythonicToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<think>I should ask the user for their phone number.</think>\n\n"
                    "respond(message='Please provide your phone number.')"
                )
            }

    backend = ModelBackend({"backend": PythonicToolCallBackend()})
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "respond", "parameters": {}},
                }
            ],
        }
    )

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "Please provide your phone number."
    }


@pytest.mark.fast
@pytest.mark.parametrize(
    "answer",
    [
        (
            "<think>Need to refuel data.</think>\n"
            "<tool_call>\n"
            '{"name":"refuel_data", "arguments":{"gb_amount":2.0}}\n'
            "</tool_call>"
        ),
        (
            "<think>Need to refuel data.</think>\n"
            "```json\n"
            '{"name":"refuel_data", "arguments":{"gb_amount":2.0}}\n'
            "```"
        ),
    ],
)
def test_model_backend_does_not_duplicate_wrapped_json_after_think_tail(answer: str):
    class WrappedJsonToolCallBackend:
        def invoke(self, payload):
            return {"answer": answer}

    backend = ModelBackend({"backend": WrappedJsonToolCallBackend()})
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "refuel_data", "parameters": {}},
                }
            ],
        }
    )

    assert result["answer"] == ""
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "refuel_data"
    assert result["tool_calls"][0]["function"]["arguments"] == {"gb_amount": 2.0}


def test_model_backend_filters_tool_calls_by_tools_schema_if_present():
    class XmlToolCallBackend:
        def __init__(self) -> None:
            self._answer = (
                "<tool_call>\n"
                "<function=respond>\n"
                "<parameter=message>Done, your mobile data is working.</parameter>\n"
                "</function>\n"
                "</tool_call>"
            )

        def invoke(self, payload):
            return {"answer": self._answer}

    backend = ModelBackend(
        {
            "backend": XmlToolCallBackend(),
            "sampling_params": {},
        }
    )
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "respond", "parameters": {}},
                }
            ],
        }
    )

    assert result["answer"] == ""
    assert result["raw_answer"] == (
        "<tool_call>\n"
        "<function=respond>\n"
        "<parameter=message>Done, your mobile data is working.</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "Done, your mobile data is working."
    }


def test_model_backend_rejects_tool_calls_not_in_tools_schema_from_text_parser():
    class XmlToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<tool_call>\n"
                    "<function=respond>\n"
                    "<parameter=message>Done, your mobile data is working.</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                )
            }

    backend = ModelBackend({"backend": XmlToolCallBackend(), "sampling_params": {}})
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "other_tool", "parameters": {}},
                }
            ],
        }
    )

    assert result["answer"].startswith("<tool_call>")
    assert "raw_answer" not in result
    assert "tool_calls" not in result
    assert result["invalid_tool_call_names"] == ["respond"]
    assert result["filtered_tool_calls"][0]["function"]["name"] == "respond"


def test_model_backend_reports_filtered_gemma4_tool_call_names():
    class Gemma4ToolCallBackend:
        def invoke(self, payload):
            return {"answer": "call:run_speed_test{}"}

    backend = ModelBackend({"backend": Gemma4ToolCallBackend(), "tool_format": "gemma4"})
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "respond", "parameters": {}},
                }
            ],
        }
    )

    assert result["answer"] == "call:run_speed_test{}"
    assert "tool_calls" not in result
    assert result["invalid_tool_call_names"] == ["run_speed_test"]
    assert result["filtered_tool_calls"][0]["function"]["name"] == "run_speed_test"


@pytest.mark.fast
def test_model_backend_wraps_gemma4_plain_text_response_as_respond_tool():
    class Gemma4PlainTextBackend:
        config = {"model_path": "/mnt/models/gemma-4-27b-it"}

        def invoke(self, payload):
            return {
                "answer": (
                    "<think>I have enough information to answer.</think>\n\n"
                    "Your mobile data has been restored. Please try loading a webpage now."
                )
            }

    backend = ModelBackend(
        {
            "backend": Gemma4PlainTextBackend(),
            "plain_text_response_tool": "respond",
            "plain_text_response_formats": ["gemma4"],
        }
    )
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "respond", "parameters": {}},
                }
            ],
        }
    )

    assert backend._tool_call_format == "gemma4"
    assert result["answer"] == ""
    assert result["plain_text_response_wrapped"] is True
    assert result["raw_answer"].startswith("<think>")
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "Your mobile data has been restored. Please try loading a webpage now."
    }


@pytest.mark.fast
def test_model_backend_wraps_qwen_plain_text_response_when_enabled():
    class QwenPlainTextBackend:
        config = {"model_path": "/mnt/models/qwen3_6_35B"}

        def invoke(self, payload):
            return {
                "answer": (
                    "<think>I should answer the user.</think>\n\n"
                    "Please run a speed test and tell me the result."
                )
            }

    backend = ModelBackend(
        {
            "backend": QwenPlainTextBackend(),
            "plain_text_response_tool": "respond",
            "plain_text_response_formats": ["qwen"],
        }
    )
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "respond", "parameters": {}},
                }
            ],
        }
    )

    assert backend._tool_call_format == "qwen"
    assert result["answer"] == ""
    assert result["plain_text_response_wrapped"] is True
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "Please run a speed test and tell me the result."
    }


@pytest.mark.fast
def test_model_backend_wraps_qwen_plain_text_response_after_bare_think_close():
    class QwenBareClosePlainTextBackend:
        config = {"model_path": "/mnt/models/qwen3_6_35B"}

        def invoke(self, payload):
            return {
                "answer": (
                    "I should ask the customer for the cancellation reason.\n"
                    "</think>\n\n"
                    "What is the reason for cancellation?"
                )
            }

    backend = ModelBackend(
        {
            "backend": QwenBareClosePlainTextBackend(),
            "plain_text_response_tool": "respond",
            "plain_text_response_formats": ["qwen"],
        }
    )
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "respond", "parameters": {}},
                }
            ],
        }
    )

    assert result["plain_text_response_wrapped"] is True
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "What is the reason for cancellation?"
    }


@pytest.mark.fast
@pytest.mark.parametrize(
    "configured_format",
    ["auto", "${TAU2_AGENT_TOOL_FORMAT:-}"],
)
def test_model_backend_wraps_qwen_plain_text_when_config_format_is_not_specific(
    configured_format,
):
    class QwenPlainTextBackend:
        config = {"model_path": "/mnt/models/Qwen3.6-35B-A3B"}

        def invoke(self, payload):
            return {
                "answer": (
                    "</think>\n\n"
                    "Please confirm whether you want me to proceed with this exchange."
                )
            }

    backend = ModelBackend(
        {
            "backend": QwenPlainTextBackend(),
            "tool_call_format": configured_format,
            "plain_text_response_tool": "${TAU2_PLAIN_TEXT_RESPONSE_TOOL:-respond}",
            "plain_text_response_formats": ["qwen"],
        }
    )
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "respond", "parameters": {}},
                }
            ],
        }
    )

    assert backend._tool_call_format == "qwen"
    assert result["answer"] == ""
    assert result["plain_text_response_wrapped"] is True
    assert result["tool_calls"][0]["function"] == {
        "name": "respond",
        "arguments": {
            "message": "Please confirm whether you want me to proceed with this exchange."
        },
    }


@pytest.mark.fast
def test_model_backend_strips_qwen_channel_suffix_from_tool_call_name():
    class QwenChannelSuffixBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<tool_call>\n"
                    "<function=respond<|channel|>commentary>\n"
                    "<parameter=message>Hello</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                )
            }

    backend = ModelBackend({"backend": QwenChannelSuffixBackend(), "tool_format": "qwen"})
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "respond", "parameters": {}},
                }
            ],
        }
    )

    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert "invalid_tool_call_names" not in result


@pytest.mark.fast
def test_model_backend_reports_malformed_gemma4_tool_call_separately():
    class Gemma4MalformedToolCallBackend:
        def invoke(self, payload):
            return {"answer": "call:respond{message:Hello"}

    backend = ModelBackend(
        {
            "backend": Gemma4MalformedToolCallBackend(),
            "tool_format": "gemma4",
            "plain_text_response_tool": "respond",
            "plain_text_response_formats": ["gemma4"],
        }
    )
    result = backend.invoke(
        {
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "respond", "parameters": {}},
                }
            ],
        }
    )

    assert result["answer"] == "call:respond{message:Hello"
    assert "tool_calls" not in result
    assert result["tool_call_parse_error_type"] == "gemma4_tool_call_format_error"
    assert "Gemma tool-call text was present" in result["tool_call_parse_error"]
    assert "plain_text_response_wrapped" not in result


def test_model_backend_no_tool_filtering_when_tools_empty():
    class XmlToolCallBackend:
        def invoke(self, payload):
            return {
                "answer": (
                    "<tool_call>\n"
                    "<function=respond>\n"
                    "<parameter=message>Done, your mobile data is working.</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                )
            }

    backend = ModelBackend({"backend": XmlToolCallBackend(), "sampling_params": {}})
    result = backend.invoke({"messages": [], "tools": []})

    assert result["answer"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "respond"
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "message": "Done, your mobile data is working."
    }


@pytest.mark.fast
def test_model_backend_ainvoke_reuses_running_thread() -> None:
    wrapped_backend = AsyncDummyModelBackend()
    backend = ModelBackend({"backend": wrapped_backend})

    async def _run() -> None:
        caller_thread_id = threading.get_ident()
        result = await backend.ainvoke({"messages": []})

        assert result["tool_calls"][0]["function"]["name"] == "run"
        assert wrapped_backend.thread_id == caller_thread_id

    asyncio.run(_run())


@pytest.mark.fast
def test_model_backend_invoke_fails_fast_in_active_loop() -> None:
    backend = ModelBackend({"backend": AsyncDummyModelBackend()})

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="active event loop"):
            backend.invoke({"messages": []})

    asyncio.run(_run())
