import pytest

from gage_eval.mcp import McpClient
from gage_eval.role.agent.tool_router import ToolRouter


@pytest.mark.fast
def test_tool_router_meta_tool_routes_to_mcp() -> None:
    executed = {}

    def executor(name, arguments):
        executed["name"] = name
        executed["arguments"] = arguments
        return {"status": "ok"}

    client = McpClient(
        mcp_client_id="mcp_main",
        transport="stub",
        endpoint="http://example.com",
        params={"executor": executor, "tools": []},
    )
    router = ToolRouter(mcp_clients={"mcp_main": client})
    tool_call = {"function": {"name": "call_spotify", "arguments": {"endpoint": "search", "params": {"q": "hi"}}}}
    tool_registry = {
        "call_spotify": {
            "x-gage": {
                "meta_tool": True,
                "app_name": "spotify",
                "allowed_endpoints": ["search"],
                "mcp_client_id": "mcp_main",
            }
        }
    }

    result = router.execute(tool_call, None, tool_registry=tool_registry)
    assert result["status"] == "success"
    assert executed["name"] == "spotify__search"
    assert executed["arguments"] == {"q": "hi"}
    assert result["resolved_tool"] == "spotify__search"


@pytest.mark.fast
def test_tool_router_meta_tool_blocks_unknown_endpoint() -> None:
    client = McpClient(
        mcp_client_id="mcp_main",
        transport="stub",
        endpoint="http://example.com",
        params={"executor": lambda *args, **kwargs: {}},
    )
    router = ToolRouter(mcp_clients={"mcp_main": client})
    tool_call = {"function": {"name": "call_spotify", "arguments": {"endpoint": "remove"}}}
    tool_registry = {
        "call_spotify": {
            "x-gage": {
                "meta_tool": True,
                "app_name": "spotify",
                "allowed_endpoints": ["search"],
                "mcp_client_id": "mcp_main",
            }
        }
    }

    result = router.execute(tool_call, None, tool_registry=tool_registry)
    assert result["status"] == "error"
    assert "invalid_endpoint" in result["output"]["error"]
    assert "resolved_tool" not in result


@pytest.mark.fast
def test_tool_router_meta_tool_flattens_params() -> None:
    executed = {}

    def executor(name, arguments):
        executed["name"] = name
        executed["arguments"] = arguments
        return {"status": "ok"}

    client = McpClient(
        mcp_client_id="mcp_main",
        transport="stub",
        endpoint="http://example.com",
        params={"executor": executor, "tools": []},
    )
    router = ToolRouter(mcp_clients={"mcp_main": client})
    tool_call = {"function": {"name": "call_spotify", "arguments": {"endpoint": "login", "username": "u", "password": "p"}}}
    tool_registry = {
        "call_spotify": {
            "x-gage": {
                "meta_tool": True,
                "app_name": "spotify",
                "allowed_endpoints": ["login"],
                "mcp_client_id": "mcp_main",
            }
        }
    }

    result = router.execute(tool_call, None, tool_registry=tool_registry)
    assert result["status"] == "success"
    assert executed["name"] == "spotify__login"
    assert executed["arguments"] == {"username": "u", "password": "p"}
    assert result["resolved_tool"] == "spotify__login"
