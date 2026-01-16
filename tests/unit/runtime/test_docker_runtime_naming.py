from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.sandbox.docker_runtime import DockerSandbox


@pytest.mark.fast
def test_docker_runtime_name_suffix_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_uuid = SimpleNamespace(hex="deadbeefcafebabe")
    monkeypatch.setattr("gage_eval.sandbox.docker_runtime.uuid.uuid4", lambda: fixed_uuid)

    sandbox = DockerSandbox(
        runtime_configs={
            "container_name_prefix": "gage-sandbox",
            "container_name_suffix": "run:1 sample/1",
        }
    )
    sandbox._config = {}

    name = sandbox._resolve_container_name()

    assert name == "gage-sandbox-run-1-sample-1-deadbeef"
