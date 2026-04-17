from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[4] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.vllm_backend import VLLMBackend


def _make_backend() -> VLLMBackend:
    config = {"model_path": "dummy-model", "tokenizer_path": "dummy-tokenizer"}
    with (
        patch.object(VLLMBackend, "load_model", return_value=MagicMock()),
        patch.object(VLLMBackend, "_init_tokenizer", return_value=MagicMock()),
        patch.object(VLLMBackend, "_load_auto_processor", return_value=MagicMock()),
    ):
        return VLLMBackend(config)


def test_shutdown_closes_engine_core_and_renderer_and_clears_model() -> None:
    backend = _make_backend()
    backend.model = MagicMock()
    backend.model.output_handler = MagicMock(done=MagicMock(return_value=True))
    engine_core_mock = backend.model.engine_core
    renderer_mock = backend.model.renderer

    with patch("gage_eval.role.model.backends.vllm_backend.graceful_loop_shutdown") as mock_graceful:
        backend.shutdown()

    engine_core_mock.shutdown.assert_called_once()
    renderer_mock.shutdown.assert_called_once()
    assert backend.model is None
    mock_graceful.assert_called_once_with(backend._loop, backend._loop_thread, None)


def test_ensure_background_loop_raises_when_shutdown_flag_is_set() -> None:
    backend = _make_backend()
    backend._loop.call_soon_threadsafe(backend._loop.stop)
    backend._loop_thread.join(timeout=2.0)
    backend._shutdown_started = True

    with pytest.raises(RuntimeError, match="unavailable after shutdown"):
        backend._ensure_background_loop()
