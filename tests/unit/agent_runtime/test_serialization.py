from __future__ import annotations

from pathlib import Path

from gage_eval.agent_runtime.serialization import to_json_compatible


class _RuntimeHelper:
    pass


def test_to_json_compatible_normalizes_runtime_helpers() -> None:
    payload = to_json_compatible(
        {
            "helper": _RuntimeHelper(),
            "path": Path("/tmp/runtime.json"),
            "error": ValueError("boom"),
        }
    )

    assert payload["helper"]["object_type"].endswith("._RuntimeHelper")
    assert payload["path"] == "/tmp/runtime.json"
    assert payload["error"]["message"] == "boom"
