from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.common.backend_utils import graceful_loop_shutdown
from gage_eval.role.model.backends.vllm_backend import VLLMBackend


class DummyModel:
    """Lightweight placeholder to avoid real vLLM engine initialization in tests."""


def make_backend() -> VLLMBackend:
    """Create a backend instance with heavyweight dependencies patched out."""

    config = {"model_path": "dummy-model", "tokenizer_path": "dummy-tokenizer"}
    with (
        patch("gage_eval.role.model.backends.vllm_backend.VLLMBackend.load_model", return_value=DummyModel()),
        patch("gage_eval.role.model.backends.vllm_backend.VLLMBackend._init_tokenizer", return_value=MagicMock()),
        patch("gage_eval.role.model.backends.vllm_backend.VLLMBackend._load_auto_processor", return_value=MagicMock()),
    ):
        return VLLMBackend(config)


class _FakeLoop:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self._running = True

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False

    def call_soon_threadsafe(self, func) -> None:
        self._calls.append("loop.stop")
        func()


class _FakeThread:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        self._calls.append(f"thread.join:{timeout}")


class _FakeModel:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def shutdown(self) -> None:
        self._calls.append("model.shutdown")


class VLLMShutdownTests(unittest.TestCase):
    def test_graceful_loop_shutdown_stops_model_before_loop(self) -> None:
        calls: list[str] = []
        loop = _FakeLoop(calls)
        loop_thread = _FakeThread(calls)
        model = _FakeModel(calls)

        with patch("gage_eval.role.common.backend_utils.torch_gpu_cleanup", side_effect=lambda: calls.append("gpu.cleanup")):
            graceful_loop_shutdown(loop, loop_thread, model)

        self.assertEqual(
            calls,
            ["model.shutdown", "loop.stop", "thread.join:1.0", "gpu.cleanup"],
        )

    def test_vllm_backend_shutdown_is_idempotent(self) -> None:
        backend = make_backend()

        with patch("gage_eval.role.model.backends.vllm_backend.graceful_loop_shutdown") as shutdown_mock:
            backend.shutdown()
            backend.shutdown()

        shutdown_mock.assert_called_once_with(backend._loop, backend._loop_thread, backend.model)
        self.assertTrue(backend._shutdown_started)
        self.assertTrue(backend._shutdown_completed)


if __name__ == "__main__":
    unittest.main()
