from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.mark.io
def test_swebench_agent_prompt_requires_submit_patch_tool() -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "swebench_pro_smoke_agent.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    prompts = payload.get("prompts") or []
    prompt = next(item for item in prompts if item.get("prompt_id") == "swebench_pro_patch_prompt")
    template = prompt.get("template") or ""

    assert "submit_patch_tool" in template
    assert "final answer MUST be the unified diff text only" in template
