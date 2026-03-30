from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[4] / "src"
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


class _FakeAsyncLLMModel:
    def __init__(self, output_handler: asyncio.Task[None]) -> None:
        self.output_handler = output_handler
        self.output_handler_seen_at_shutdown = object()

    def shutdown(self) -> None:
        self.output_handler_seen_at_shutdown = self.output_handler


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
        original_loop = backend._loop
        original_loop_thread = backend._loop_thread
        original_model = backend.model

        with patch("gage_eval.role.model.backends.vllm_backend.graceful_loop_shutdown") as shutdown_mock:
            backend.shutdown()
            backend.shutdown()

        shutdown_mock.assert_called_once_with(original_loop, original_loop_thread, original_model)
        self.assertTrue(backend._shutdown_started)
        self.assertTrue(backend._shutdown_completed)
        self.assertIsNone(backend.model)
        self.assertIsNone(backend._loop)
        self.assertIsNone(backend._loop_thread)
        self.assertIsNone(backend._processor)
        self.assertIsNone(backend._tokenizer)
        self.assertIsNone(backend._engine_runtime)
        self.assertIsNone(backend._engine_mm_support)

    def test_vllm_backend_shutdown_unregisters_cleanup_callback(self) -> None:
        backend = make_backend()
        unregister = MagicMock()
        backend._cleanup_unregister = unregister

        with patch("gage_eval.role.model.backends.vllm_backend.graceful_loop_shutdown"):
            backend.shutdown()
            backend.shutdown()

        unregister.assert_called_once_with()

    def test_vllm_backend_shutdown_quiesces_output_handler_before_engine_shutdown(self) -> None:
        backend = make_backend()
        loop = asyncio.new_event_loop()

        async def _wait_forever() -> None:
            await asyncio.sleep(10)

        handler = loop.create_task(_wait_forever())
        model = _FakeAsyncLLMModel(handler)
        backend.model = model
        backend._loop = loop
        backend._loop_thread = None

        with (
            patch(
                "gage_eval.role.model.backends.vllm_backend.run_coroutine_threadsafe_with_timeout",
                side_effect=lambda target_loop, coro, **kwargs: target_loop.run_until_complete(coro),
            ),
            patch(
                "gage_eval.role.model.backends.vllm_backend.graceful_loop_shutdown",
                side_effect=lambda loop_arg, loop_thread_arg, model_arg: model_arg.shutdown(),
            ),
        ):
            backend.shutdown()

        self.assertTrue(handler.cancelled())
        self.assertIsNone(model.output_handler_seen_at_shutdown)
        loop.close()

    def test_vllm_backend_starts_background_loop_lazily_once(self) -> None:
        backend = make_backend()
        fake_loop = object()
        fake_thread = MagicMock()
        unregister = MagicMock()

        with (
            patch("gage_eval.role.model.backends.vllm_backend.asyncio.new_event_loop", return_value=fake_loop),
            patch("gage_eval.role.model.backends.vllm_backend.threading.Thread", return_value=fake_thread),
            patch("gage_eval.role.model.backends.vllm_backend.install_signal_cleanup", return_value=unregister),
        ):
            resolved_first = backend._ensure_background_loop()
            resolved_second = backend._ensure_background_loop()

        self.assertIs(resolved_first, fake_loop)
        self.assertIs(resolved_second, fake_loop)
        fake_thread.start.assert_called_once_with()
        unregister.assert_not_called()
        self.assertIs(backend._loop, fake_loop)
        self.assertIs(backend._loop_thread, fake_thread)

    def test_vllm_backend_shutdown_runs_on_init_failure(self) -> None:
        with (
            patch("gage_eval.role.model.backends.vllm_backend.graceful_loop_shutdown") as shutdown_mock,
            patch("gage_eval.role.model.backends.vllm_backend.VLLMBackend.load_model", side_effect=RuntimeError("boom")),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            VLLMBackend({"model_path": "dummy-model"})

        shutdown_mock.assert_called_once_with(None, None, None)


if __name__ == "__main__":
    unittest.main()
