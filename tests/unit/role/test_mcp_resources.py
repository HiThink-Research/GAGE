import pytest

from gage_eval.mcp import McpClient


@pytest.mark.fast
def test_mcp_resources_and_sampling():
    calls = []

    def requester(method, payload):
        calls.append((method, payload))
        if method == "list_resources":
            return {"resources": [{"uri": "prompt://demo", "name": "demo"}]}
        if method == "read_resource":
            return {"content": "hello"}
        if method == "sample":
            return {"choices": [{"text": "ok"}]}
        return {}

    client = McpClient(
        mcp_client_id="mcp_demo",
        endpoint="http://example.com",
        params={"requester": requester},
    )
    resources = client.list_resources()
    assert resources[0]["uri"] == "prompt://demo"
    content = client.read_resource("prompt://demo")
    assert content["content"] == "hello"
    response = client.sample({"prompt": "ping"})
    assert response["choices"][0]["text"] == "ok"
    assert calls[0][0] == "list_resources"
