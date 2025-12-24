"""Minimal process-isolated vLLM worker for the native backend.

The parent process communicates with the worker process via `multiprocessing.Pipe`
to reduce the blast radius of crashes and avoid thread-safety pitfalls.

Protocol:
- Request: `(req_id: int, payload: dict)`
- Response: `(req_id: int, result: dict)` (or a dict with an `"error"` key)

Limitations:
- Only supports synchronous single-prompt generation. This is a minimal viable
  isolation mode intended for robustness rather than maximum throughput.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from typing import Any, Dict, Tuple

from gage_eval.utils.cleanup import install_signal_cleanup, torch_gpu_cleanup


def _worker_loop(conn, config: Dict[str, Any]) -> None:
    """Runs the isolated vLLM worker loop.

    Args:
        conn: A `multiprocessing.Connection` created from `mp.Pipe()`.
        config: Backend config dict containing `model_path` and optional vLLM args.
    """

    def _cleanup(llm=None):
        try:
            if llm and hasattr(llm, "shutdown"):
                llm.shutdown()
        except Exception:
            pass
        torch_gpu_cleanup()
        try:
            conn.close()
        except Exception:
            pass

    # STEP 1: Import heavy dependencies inside the worker process.
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as exc:
        conn.send((-1, {"error": f"import vllm failed: {exc}"}))
        return
    try:
        # STEP 2: Parse config and initialize the vLLM engine.
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for vLLM worker")
        max_tokens = int(config.get("max_tokens", 512))
        sampling_defaults = config.get("sampling_params") or {}
        model_kwargs: Dict[str, Any] = {}
        for key in ("tensor_parallel_size", "max_model_len", "swap_space"):
            if config.get(key) is not None:
                model_kwargs[key] = int(config[key])
        if config.get("gpu_memory_utilization") is not None:
            try:
                model_kwargs["gpu_memory_utilization"] = float(config.get("gpu_memory_utilization"))
            except Exception:
                pass
        model_kwargs.setdefault("trust_remote_code", True)
        if config.get("enforce_eager") is not None:
            model_kwargs["enforce_eager"] = bool(config.get("enforce_eager"))
        llm = LLM(model=model_path, **model_kwargs)
        install_signal_cleanup(lambda: _cleanup(llm))
        conn.send((0, {"status": "ready"}))

        # STEP 3: Serve requests until the parent sends a sentinel (None).
        while True:
            msg = conn.recv()
            if msg is None:
                break
            req_id, payload = msg
            try:
                prompt = payload.get("prompt") or payload.get("text") or ""
                sampling = dict(sampling_defaults)
                sampling.update(payload.get("sampling_params") or {})
                params = SamplingParams(**{**sampling, "max_tokens": sampling.get("max_tokens", max_tokens)})
                outputs = llm.generate([prompt], sampling_params=params)
                if outputs and outputs[0].outputs:
                    text = outputs[0].outputs[0].text
                else:
                    text = ""
                conn.send((req_id, {"answer": text.strip()}))
            except Exception:
                conn.send((req_id, {"error": traceback.format_exc()}))
    except Exception:
        conn.send((-1, {"error": traceback.format_exc()}))
    finally:
        # STEP 4: Best-effort cleanup to release GPU memory and close the pipe.
        _cleanup(locals().get("llm"))


class VLLMIsolatedWorker:
    """Wrapper for a process-isolated vLLM engine."""

    def __init__(self, config: Dict[str, Any]) -> None:
        parent_conn, child_conn = mp.Pipe()
        self._conn = parent_conn
        ctx = mp.get_context("spawn")
        self._proc = ctx.Process(target=_worker_loop, args=(child_conn, config), daemon=True)
        self._proc.start()
        code, payload = parent_conn.recv()
        if code != 0:
            raise RuntimeError(f"Failed to start vLLM worker: {payload}")

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a completion for a single request payload."""

        req_id = os.getpid()  # lightweight req id
        self._conn.send((req_id, payload))
        rid, resp = self._conn.recv()
        if rid != req_id:
            raise RuntimeError(f"Mismatched response id: {rid} != {req_id}")
        return resp

    def shutdown(self) -> None:
        """Stops the worker process and releases resources."""

        try:
            self._conn.send(None)
        except Exception:
            pass
        try:
            self._proc.join(timeout=1.0)
        except Exception:
            pass
