import json

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
    backend = CliBackend({"command": "python -c 'print(\"hello\")'"})
    result = backend.invoke({})
    assert result["answer"] == "hello"


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
