from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def test_to_json_compatible_handles_pydantic_v2_basemodel() -> None:
    from pydantic import BaseModel

    class Message(BaseModel):
        role: str
        content: str

    assert to_json_compatible(Message(role="assistant", content="hi")) == {
        "role": "assistant",
        "content": "hi",
    }


def test_to_json_compatible_uses_descriptor_for_live_dataclass_handles() -> None:
    class _SocketBackedClient:
        def __deepcopy__(self, memo: dict[int, Any]) -> "_SocketBackedClient":
            raise TypeError("cannot pickle 'socket' object")

    @dataclass
    class _LeaseLike:
        lease_id: str
        environment: _SocketBackedClient

        def to_descriptor(self) -> dict[str, Any]:
            return {
                "lease_id": self.lease_id,
                "provider": "docker",
                "environment_descriptor": {"env_id": "env-1"},
            }

    payload = to_json_compatible(
        {
            "environment_lease": _LeaseLike(
                lease_id="lease-1",
                environment=_SocketBackedClient(),
            )
        }
    )

    assert payload == {
        "environment_lease": {
            "lease_id": "lease-1",
            "provider": "docker",
            "environment_descriptor": {"env_id": "env-1"},
        }
    }
