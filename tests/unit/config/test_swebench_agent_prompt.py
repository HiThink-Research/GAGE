from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.mark.io
def test_swebench_agent_prompt_requires_submit_patch_tool() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "swebench_pro"
        / "swebench_pro_smoke_agent.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert "prompts" not in payload
    assert payload["benchmarks"][0]["kit_id"] == "swebench"
    assert payload["agents"][0]["scheduler"]["type"] == "framework_loop"

    # SWE-bench prompt and submit contract are owned by the v2 kit, not by
    # config-local legacy prompt templates.
    from gage_eval.agent_eval_kits.swebench.tools import build_swebench_tools

    tool_names = {tool["function"]["name"] for tool in build_swebench_tools({})}
    assert "submit_patch_tool" in tool_names
