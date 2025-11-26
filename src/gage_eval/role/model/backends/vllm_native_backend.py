"""VLLM backend that executes models in-process for local benchmarking."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend


@registry.asset(
    "backends",
    "vllm_native",
    desc="vLLM 原生推理后端，直接在进程内加载模型。",
    tags=("llm", "local", "native"),
    modalities=("text",),
)
class VLLMNativeBackend(EngineBackend):
    """Backend that leverages vLLM's in-process LLM runner."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._sampling_defaults = config.get("sampling_params") or {}
        self._model_kwargs = self._build_model_kwargs(config)
        self._max_tokens = int(config.get("max_tokens", 512))
        # 强制线程安全：同步 vLLM 接口非线程安全，统一加锁
        self._lock = threading.Lock()
        self._vllm_version = self._detect_version()
        if self._vllm_version:
            logger.info("Detected vLLM version %s (native backend)", self._vllm_version)
        # Feature detection: 是否支持 list[SamplingParams] 形式的批量 generate
        self._list_sampling_supported = True
        self._isolated = bool(config.get("process_isolation") or os.environ.get("GAGE_EVAL_VLLM_ISOLATED"))
        cfg = dict(config)
        cfg.setdefault("execution_mode", "native")
        super().__init__(cfg)

    def load_model(self, config: Dict[str, Any]):
        self._ensure_spawn_start_method()
        if self._isolated:
            logger.info("VLLMNativeBackend starting isolated process worker")
            return VLLMIsolatedWorker(config)
        try:  # pragma: no cover - heavy dependency
            from vllm import LLM
            import torch
            OOM_ERRORS = (torch.cuda.OutOfMemoryError, ValueError)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("vllm is not installed; cannot use VLLMNativeBackend") from exc

        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("VLLMNativeBackend requires 'model_path' in the backend config")
        logger.info("Loading vLLM model {} (kwargs={})", model_path, self._model_kwargs)
        attempts = [
            ("orig", self._model_kwargs.get("gpu_memory_utilization")),
            ("fallback0.7", 0.7),
            ("fallback0.5", 0.5),
        ]
        last_exc: Optional[Exception] = None
        for tag, util in attempts:
            try:
                if util is not None:
                    self._model_kwargs["gpu_memory_utilization"] = util
                return LLM(model=model_path, **self._model_kwargs)
            except OOM_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    "VLLMNativeBackend load_model OOM (tag=%s util=%s): %s, retrying next tier",
                    tag,
                    util,
                    exc,
                )
                continue
        raise RuntimeError("vllm load_model failed after OOM retries") from last_exc

    def prepare_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("prompt") or payload.get("text") or ""
        messages = payload.get("messages") or []
        rendered_prompt = self._render_messages(messages, prompt)
        sampling = dict(self._sampling_defaults)
        sampling.update(payload.get("sampling_params") or {})
        if not rendered_prompt.strip():
            # Soft skip: allow pipeline继续，不让空提示打断进程
            sample = payload.get("sample") or {}
            sample_id = sample.get("idx") or sample.get("id") or sample.get("sample_id") or "<unknown>"
            logger.warning("VLLMNativeBackend: empty prompt/messages for sample %s, returning empty answer", sample_id)
            return {"_skip": True, "_skip_reason": "empty_prompt", "sample_id": sample_id}
        return {
            "prompt": rendered_prompt,
            "sampling_params": sampling,
        }

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """单请求兼容，用于无 Scheduler 场景。"""

        if inputs.get("_skip"):
            return {"answer": "", "_batch_path": "skipped", "skip_reason": inputs.get("_skip_reason")}
        prompt = inputs.get("prompt") or ""
        sampling_params = self._build_sampling_params(inputs.get("sampling_params") or {})
        if self._isolated:
            return self.model.generate({"prompt": prompt, "sampling_params": sampling_params})
        with self._lock:
            outputs = self.model.generate([prompt], sampling_params=sampling_params)
        return self._extract_first(outputs, batch_path="native_batch")

    def generate_batch(self, inputs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量生成，支持 prompts/sampling_params 列表；不支持时退化为单次调用。"""

        prepared_inputs = [self.prepare_inputs(item) for item in inputs_list]
        prompts = [item.get("prompt") or "" for item in prepared_inputs]
        params_list = [self._build_sampling_params(item.get("sampling_params") or {}) for item in prepared_inputs]
        skip_mask = [bool(item.get("_skip")) for item in prepared_inputs]
        try:
            # 过滤可执行的条目，空 prompt 的跳过并补空结果
            exec_indices = [idx for idx, skip in enumerate(skip_mask) if not skip]
            exec_prompts = [prompts[idx] for idx in exec_indices]
            exec_params = [params_list[idx] for idx in exec_indices]

            results: List[Dict[str, Any]] = [
                {"answer": "", "_batch_path": "skipped", "skip_reason": "empty_prompt"} for _ in inputs_list
            ]
            if exec_prompts:
                if self._list_sampling_supported:
                    try:
                        with self._lock:
                            outputs = self.model.generate(exec_prompts, sampling_params=exec_params)
                    except TypeError as exc:
                        # 部分 vLLM 版本不支持 list[SamplingParams]，降级并打点
                        self._list_sampling_supported = False
                        logger.warning(
                            "VLLMNativeBackend: list[SamplingParams] unsupported, "
                            "falling back to per-prompt sampling (error=%s)",
                            exc,
                        )
                        outputs = None
                else:
                    outputs = None

                if outputs is not None:
                    if len(outputs) != len(exec_prompts):
                        raise ValueError(f"vLLM returned {len(outputs)} results for {len(exec_prompts)} prompts")
                    for out, original_idx in zip(outputs, exec_indices):
                        results[original_idx] = self._extract_first([out], batch_path="native_batch")
                else:
                    # 降级路径：对每条执行一次 generate，保留 _batch_path=fallback_serial
                    for original_idx, (prompt, params) in zip(exec_indices, zip(exec_prompts, exec_params)):
                        with self._lock:
                            single = self.model.generate([prompt], sampling_params=params)
                        results[original_idx] = self._extract_first(single, batch_path="fallback_serial")
            return results
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("vLLM generate_batch fallback to per-request due to: {}", exc)
            results: List[Dict[str, Any]] = []
            for idx, (prompt, params, skip) in enumerate(zip(prompts, params_list, skip_mask)):
                if skip:
                    results.append({"answer": "", "_batch_path": "skipped", "skip_reason": "empty_prompt"})
                    continue
                with self._lock:
                    single = self.model.generate([prompt], sampling_params=params)
                results.append(self._extract_first(single, batch_path="fallback_serial"))
            return results

    def _build_sampling_params(self, runtime_params: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            from vllm import SamplingParams
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("vllm is not installed; cannot build SamplingParams") from exc

        params = dict(runtime_params)
        params.setdefault("max_tokens", self._max_tokens)
        params.setdefault("temperature", 0.0)
        return SamplingParams(**params)

    @staticmethod
    def _extract_first(outputs, *, batch_path: str = "native_batch") -> Dict[str, Any]:
        if not outputs:
            return {"answer": ""}
        entry = outputs[0]
        if not getattr(entry, "outputs", None):
            return {"answer": ""}
        text = entry.outputs[0].text
        return {"answer": text.strip(), "_batch_path": batch_path}

    @staticmethod
    def _render_messages(messages: List[Dict[str, Any]], fallback: str) -> str:
        if not messages:
            return fallback
        segments: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content")
            if isinstance(content, list):
                text_parts = []
                for fragment in content:
                    if isinstance(fragment, dict) and fragment.get("type") == "text":
                        text_parts.append(str(fragment.get("text", "")))
                text = " ".join(text_parts)
            else:
                text = str(content) if content is not None else ""
            segments.append(f"{role}: {text}".strip())
        segments.append("assistant:")
        return "\n".join(segments)

    @staticmethod
    def _build_model_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
        """Compose kwargs passed to vLLM.LLM while honoring optional overrides."""

        kwargs: Dict[str, Any] = {}
        numeric_fields = ("tensor_parallel_size", "max_model_len", "swap_space")
        boolean_fields = ("trust_remote_code", "enforce_eager")
        passthrough_fields = ("device",)

        for field in numeric_fields:
            value = config.get(field)
            if value is not None:
                kwargs[field] = int(value)
        # gpu_memory_utilization 允许浮点配置
        if config.get("gpu_memory_utilization") is not None:
            try:
                kwargs["gpu_memory_utilization"] = float(config.get("gpu_memory_utilization"))
            except Exception:
                pass

        for field in boolean_fields:
            if field in config:
                kwargs[field] = bool(config[field])

        for field in passthrough_fields:
            value = config.get(field)
            if value:
                kwargs[field] = value

        # Default sensible values if not provided.
        kwargs.setdefault("tensor_parallel_size", 1)
        kwargs.setdefault("trust_remote_code", True)

        return kwargs

    @staticmethod
    def _ensure_spawn_start_method() -> None:
        """Force spawn to avoid CUDA re-init in forked vLLM workers."""

        try:
            import torch.multiprocessing as mp  # type: ignore
        except Exception:  # pragma: no cover - torch missing
            import multiprocessing as mp  # type: ignore

        current = mp.get_start_method(allow_none=True)
        if current != "spawn":
            logger.info("vLLMNativeBackend switching multiprocessing start method to 'spawn' (was %s)", current)
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError as exc:  # pragma: no cover - defensive
                logger.warning("vLLMNativeBackend failed to set spawn start method: {}", exc)
        # Ensure vLLM respects spawn even if upstream defaults differ.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    @staticmethod
    def _detect_version() -> Optional[str]:
        try:  # pragma: no cover - optional dependency
            import vllm  # type: ignore
        except Exception:
            logger.warning("vLLM not installed; version detection skipped (native backend)")
            return None
        ver = getattr(vllm, "__version__", None)
        if isinstance(ver, str):
            return ver
        try:
            from importlib.metadata import version as meta_version

            return meta_version("vllm")
        except Exception:
            return None
from gage_eval.role.model.backends.vllm_worker import VLLMIsolatedWorker
