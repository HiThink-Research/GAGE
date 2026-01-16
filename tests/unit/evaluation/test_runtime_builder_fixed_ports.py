from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import RoleAdapterSpec
from gage_eval.evaluation import runtime_builder


@pytest.mark.fast
def test_fixed_ports_force_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []

    def fake_warning(message: str, *args: object, **kwargs: object) -> None:
        warnings.append(message.format(*args))

    monkeypatch.setattr(runtime_builder.logger, "warning", fake_warning)

    role_bindings = {
        "toolchain_main": RoleAdapterSpec(
            adapter_id="toolchain_main",
            role_type="toolchain",
            sandbox={"sandbox_id": "appworld_local"},
        )
    }
    sandbox_profiles = {
        "appworld_local": {
            "sandbox_id": "appworld_local",
            "runtime_configs": {"ports": ["8000:8000", "9000:9000"]},
        }
    }

    resolved = runtime_builder._apply_fixed_port_guard(
        scope="task 'demo'",
        concurrency=4,
        role_bindings=role_bindings,
        sandbox_profiles=sandbox_profiles,
    )

    assert resolved == 1
    assert warnings


@pytest.mark.fast
def test_dynamic_ports_keep_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []

    def fake_warning(message: str, *args: object, **kwargs: object) -> None:
        warnings.append(message.format(*args))

    monkeypatch.setattr(runtime_builder.logger, "warning", fake_warning)

    role_bindings = {
        "toolchain_main": RoleAdapterSpec(
            adapter_id="toolchain_main",
            role_type="toolchain",
            sandbox={"runtime_configs": {"ports": ["0:8000"]}},
        )
    }

    resolved = runtime_builder._apply_fixed_port_guard(
        scope="task 'demo'",
        concurrency=3,
        role_bindings=role_bindings,
        sandbox_profiles={},
    )

    assert resolved == 3
    assert warnings == []
