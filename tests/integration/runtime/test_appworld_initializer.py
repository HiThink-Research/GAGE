from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from gage_eval.sandbox.integrations.appworld.initializer import AppWorldInitializer


class StubInitializer:
    def __init__(self, **config: Any) -> None:
        self._config = dict(config)
        self.started = False
        self.parallelizable_across = "all"

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    @property
    def configs(self) -> List[Dict[str, Any]]:
        return [dict(self._config)]

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


def stub_factory(**config: Any) -> StubInitializer:
    return StubInitializer(**config)


@pytest.mark.fast
def test_appworld_initializer_maps_endpoints() -> None:
    initializer = AppWorldInitializer(
        env_endpoint="http://127.0.0.1:8000",
        apis_endpoint="http://127.0.0.1:9000",
        mcp_endpoint="http://127.0.0.1:5001",
        start_servers=True,
        initializer_factory=stub_factory,
        experiment_name="demo",
    )

    with initializer as running:
        config = running.config
        assert config["remote_environment_url"] == "http://127.0.0.1:8000"
        assert config["remote_apis_url"] == "http://127.0.0.1:9000"
        assert config["remote_mcp_url"] == "http://127.0.0.1:5001"
        assert running.parallelizable_across == "all"
