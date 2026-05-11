from __future__ import annotations

import pytest

from gage_eval.external_harness_kits.errors import ExternalHarnessError
from gage_eval.external_harness_kits.harbor.environment import build_harbor_environment_binding


@pytest.mark.fast
def test_docker_provider_minimal_config_translates_without_socket_mount() -> None:
    binding = build_harbor_environment_binding(
        {
            "env_id": "harbor_trial_docker",
            "provider": "docker",
            "resources": {"cpus": 2, "memory_mb": 4096, "storage_mb": 20480},
        }
    )

    assert binding.env_id == "harbor_trial_docker"
    assert binding.provider == "docker"
    assert binding.harbor_environment == {
        "type": "docker",
        "delete": False,
        "override_cpus": 2,
        "override_memory_mb": 4096,
        "override_storage_mb": 20480,
    }
    assert "/var/run/docker.sock" not in str(binding.harbor_environment)
    assert binding.preflight_notes == []


@pytest.mark.fast
def test_docker_provider_delete_can_be_enabled_by_environment_override() -> None:
    binding = build_harbor_environment_binding(
        {"env_id": "harbor_trial_docker", "provider": "docker"},
        environment_override={"delete": True},
    )

    assert binding.harbor_environment == {"type": "docker", "delete": True}


@pytest.mark.fast
def test_e2b_provider_minimal_config_translates_with_deferred_live_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    binding = build_harbor_environment_binding(
        {
            "env_id": "harbor_trial_e2b",
            "provider": "e2b",
            "provider_config": {"kwargs": {"template_id": "tb2-template"}},
        }
    )

    assert binding.harbor_environment == {
        "type": "e2b",
        "kwargs": {"template_id": "tb2-template"},
    }
    assert any("deferred live check" in note for note in binding.preflight_notes)
    assert any("E2B token" in note for note in binding.preflight_notes)


@pytest.mark.fast
def test_e2b_provider_missing_template_gets_deferred_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    binding = build_harbor_environment_binding({"env_id": "e2b", "provider": "e2b"})

    assert binding.harbor_environment == {"type": "e2b"}
    assert any("template" in note for note in binding.preflight_notes)


@pytest.mark.fast
def test_local_process_provider_fails_fast_with_invalid_environment_provider() -> None:
    with pytest.raises(ExternalHarnessError) as exc_info:
        build_harbor_environment_binding({"env_id": "host", "provider": "local_process"})

    assert exc_info.value.code == "external_harness.config.invalid_environment_provider"
    assert "not a Harbor trial sandbox provider" in str(exc_info.value)


@pytest.mark.fast
def test_provider_override_type_mismatch_uses_config_phase_code() -> None:
    with pytest.raises(ExternalHarnessError) as exc_info:
        build_harbor_environment_binding(
            {"env_id": "docker_env", "provider": "docker"},
            environment_override={"type": "e2b"},
            validation_phase="config",
        )

    assert exc_info.value.code == "external_harness.config.provider_mismatch"


@pytest.mark.fast
def test_provider_override_type_mismatch_uses_environment_phase_code() -> None:
    with pytest.raises(ExternalHarnessError) as exc_info:
        build_harbor_environment_binding(
            {"env_id": "docker_env", "provider": "docker"},
            environment_override={"type": "e2b"},
            validation_phase="environment",
        )

    assert exc_info.value.code == "external_harness.environment.provider_mismatch"


@pytest.mark.fast
def test_environment_override_import_path_is_forbidden() -> None:
    with pytest.raises(ExternalHarnessError) as exc_info:
        build_harbor_environment_binding(
            {"env_id": "docker_env", "provider": "docker"},
            environment_override={"import_path": "custom.module:Environment"},
        )

    assert exc_info.value.code == "external_harness.config.invalid_environment_override"


@pytest.mark.fast
def test_environment_override_env_rejects_secret_like_keys() -> None:
    with pytest.raises(ExternalHarnessError) as exc_info:
        build_harbor_environment_binding(
            {"env_id": "docker_env", "provider": "docker"},
            environment_override={"env": {"OPENAI_API_KEY": "sk-test"}},
        )

    assert exc_info.value.code == "external_harness.config.invalid_environment_override"


@pytest.mark.fast
def test_e2b_override_template_id_prevents_false_missing_template_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    binding = build_harbor_environment_binding(
        {"env_id": "e2b", "provider": "e2b"},
        environment_override={"kwargs": {"template_id": "override-template"}},
    )

    assert binding.harbor_environment == {
        "type": "e2b",
        "kwargs": {"template_id": "override-template"},
    }
    assert not any("template" in note for note in binding.preflight_notes)
    assert any("E2B token" in note for note in binding.preflight_notes)


@pytest.mark.fast
def test_docker_preflight_does_not_require_yaml_socket_mount() -> None:
    binding = build_harbor_environment_binding(
        {
            "env_id": "docker_env",
            "provider": "docker",
            "provider_config": {
                "docker_image": "ubuntu:24.04",
                "docker_socket": "/var/run/docker.sock",
            },
        }
    )

    assert binding.harbor_environment == {
        "type": "docker",
        "delete": False,
        "kwargs": {"docker_image": "ubuntu:24.04"},
    }
    assert "/var/run/docker.sock" not in str(binding.harbor_environment)
