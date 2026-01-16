import pytest

from gage_eval.role.agent.human_gateway import HumanGateway, HumanRequest
from gage_eval.role.agent.tool_router import ToolRouter


@pytest.mark.fast
def test_human_gateway_request():
    def provider(request: HumanRequest) -> str:
        return f"ok:{request.question}"

    gateway = HumanGateway(input_provider=provider)
    result = gateway.request("approve?")
    assert result == "ok:approve?"


@pytest.mark.fast
def test_tool_router_host_execution():
    gateway = HumanGateway(input_provider=lambda req: "yes")
    router = ToolRouter(human_gateway=gateway)
    tool_call = {"function": {"name": "human_input", "arguments": {"question": "need input"}}}
    tool_registry = {"human_input": {"x-gage": {"execution": "host"}}}
    result = router.execute(tool_call, None, tool_registry=tool_registry)
    assert result["status"] == "success"
    assert result["output"]["response"] == "yes"
