from __future__ import annotations

import pytest

from gage_eval.agent_runtime.resolver import AgentRuntimeResolver
from gage_eval.agent_runtime.schedulers.framework_loop import FrameworkLoopScheduler
from gage_eval.agent_runtime.schedulers.installed_client import InstalledClientScheduler
from gage_eval.agent_runtime.spec import AgentRuntimeSpec


def _build_runtime_spec(*, runtime_id: str, scheduler: str) -> AgentRuntimeSpec:
    return AgentRuntimeSpec(
        agent_runtime_id=runtime_id,
        scheduler=scheduler,
        benchmark_kit_id="swebench",
    )


@pytest.mark.fast
def test_resolver_resolves_installed_client() -> None:
    resolver = AgentRuntimeResolver(
        [_build_runtime_spec(runtime_id="rt-installed", scheduler="installed_client")]
    )

    plan = resolver.resolve("rt-installed")

    assert plan.scheduler_type == "installed_client"
    assert plan.benchmark_kit_id == "swebench"


@pytest.mark.fast
def test_resolver_resolves_framework_loop() -> None:
    resolver = AgentRuntimeResolver(
        [_build_runtime_spec(runtime_id="rt-framework", scheduler="framework_loop")]
    )

    plan = resolver.resolve("rt-framework")

    assert plan.scheduler_type == "framework_loop"


@pytest.mark.fast
def test_resolver_rejects_unknown_runtime_id() -> None:
    resolver = AgentRuntimeResolver([])

    with pytest.raises(KeyError):
        resolver.resolve("missing")


@pytest.mark.fast
def test_resolver_builds_installed_scheduler() -> None:
    resolver = AgentRuntimeResolver(
        [_build_runtime_spec(runtime_id="rt-installed", scheduler="installed_client")]
    )

    scheduler = resolver.build_scheduler(resolver.resolve("rt-installed"))

    assert isinstance(scheduler, InstalledClientScheduler)


@pytest.mark.fast
def test_resolver_builds_framework_scheduler() -> None:
    resolver = AgentRuntimeResolver(
        [_build_runtime_spec(runtime_id="rt-framework", scheduler="framework_loop")]
    )

    scheduler = resolver.build_scheduler(resolver.resolve("rt-framework"))

    assert isinstance(scheduler, FrameworkLoopScheduler)
