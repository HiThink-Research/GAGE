import pytest

from gage_eval.sandbox.docker_runtime import normalize_runtime_configs


@pytest.mark.fast
def test_docker_network_alias_bridge_host():
    normalized = normalize_runtime_configs({"network_mode": "bridge_host"})
    assert normalized["network_mode"] == "bridge"
    assert "host.docker.internal:host-gateway" in normalized["extra_hosts"]


@pytest.mark.fast
def test_docker_network_host_mode():
    normalized = normalize_runtime_configs({"network_mode": "host"})
    assert normalized["network_mode"] == "host"
    assert "extra_hosts" not in normalized
