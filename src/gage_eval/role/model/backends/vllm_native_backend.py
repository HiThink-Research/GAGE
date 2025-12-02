"""VLLM backend that executes models in-process for local benchmarking."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.runtime import BackendCapabilities, ChatTemplateMixin, ChatTemplatePolicy
from gage_eval.utils.chat_templates import get_fallback_template


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
            logger.info("Detected vLLM version {} (native backend)", self._vllm_version)
        # Feature detection: 是否支持 list[SamplingParams] 形式的批量 generate
        self._list_sampling_supported = True
        self._isolated = bool(config.get("process_isolation") or os.environ.get("GAGE_EVAL_VLLM_ISOLATED"))
        self._chat_template_mode = str(config.get("use_chat_template", "auto"))
        self._chat_template_policy = ChatTemplatePolicy(mode=self._chat_template_mode)
        self._fallback_template = get_fallback_template("text")
        self._tokenizer = None
        self._cfg_tokenizer_path = config.get("tokenizer_path") or config.get("tokenizer_name")
        cfg = dict(config)
        cfg.setdefault("execution_mode", "native")
        super().__init__(cfg)

    def load_model(self, config: Dict[str, Any]):
        self._ensure_spawn_start_method()
        if self._isolated:
            logger.info("VLLMNativeBackend starting isolated process worker")
            return VLLMIsolatedWorker(config)
        self._tokenizer = self._init_tokenizer(config)
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
        self._check_tokenizer_conflict(payload)
        prompt = payload.get("prompt") or payload.get("text") or ""
        messages = payload.get("messages") or []
        policy = self._chat_template_policy
        caps = BackendCapabilities(supports_mm=False, has_processor_chat_template=False)
        rendered_prompt = ""

        if ChatTemplateMixin.should_render(payload, policy):
            template_source = ChatTemplateMixin.select_template("text", policy, caps)
            template_fn = getattr(self._tokenizer, "apply_chat_template", None) if self._tokenizer else None
            fallback_tpl = None if template_source == "model" else self._fallback_template
            rendered_prompt = ChatTemplateMixin.render(
                messages,
                template_fn=template_fn,
                fallback_fn=lambda msgs: self._fallback_render(msgs, fallback_tpl),
                add_generation_prompt=True,
                chat_template=fallback_tpl,
            )
            payload["chat_template_mode"] = "backend"
            payload["template_source"] = "model" if template_fn else "fallback"
            payload["rendered_by"] = "backend"
        else:
            rendered_prompt = prompt or self._simple_render(messages)

        sampling = dict(self._sampling_defaults)
        sampling.update(payload.get("sampling_params") or {})
        if not rendered_prompt.strip():
            sample = payload.get("sample") or {}
            sample_id = sample.get("idx") or sample.get("id") or sample.get("sample_id") or "<unknown>"
            logger.warning("VLLMNativeBackend: empty prompt/messages for sample {}, returning empty answer", sample_id)
            return {"_skip": True, "_skip_reason": "empty_prompt", "sample_id": sample_id}
        caps = BackendCapabilities(supports_mm=False, has_processor_chat_template=False)
        return {
            "prompt": rendered_prompt,
            "sampling_params": sampling,
            "cache_suffix": ChatTemplateMixin.get_cache_suffix("text", self._chat_template_policy, caps),
        }

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """单请求兼容，用于无 Scheduler 场景。"""

        if inputs.get("_skip"):
            return {"answer": "", "_batch_path": "skipped", "skip_reason": inputs.get("_skip_reason")}
        prompt = inputs.get("prompt") or ""
        sampling_params = self._build_sampling_params(inputs.get("sampling_params") or {})
        if self._isolated:
            result = self.model.generate({"prompt": prompt, "sampling_params": sampling_params})
            self._attach_template_metadata(inputs, result)
            return result
        with self._lock:
            outputs = self.model.generate([prompt], sampling_params=sampling_params)
        result = self._extract_first(outputs, batch_path="native_batch")
        self._attach_template_metadata(inputs, result)
        return result

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
        # 回填模板元数据
        for prepared, result in zip(prepared_inputs, results):
            self._attach_template_metadata(prepared, result)
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

    def _attach_template_metadata(self, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
        meta_keys = ("chat_template_mode", "template_source", "rendered_by", "cache_suffix")
        sample = payload.get("sample") or {}
        for key in meta_keys:
            value = payload.get(key)
            if value is None:
                value = sample.get(key)
            if value is not None and key not in result:
                result[key] = value
        if "_tokenizer_path" not in result and getattr(self, "_cfg_tokenizer_path", None):
            result["_tokenizer_path"] = self._cfg_tokenizer_path

    @staticmethod
    def _simple_render(messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""
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

    def _fallback_render(self, messages: List[Dict[str, Any]], tpl: Optional[str]) -> str:
        return self._simple_render(messages)

    def _init_tokenizer(self, config: Dict[str, Any]):
        tok_name = config.get("tokenizer_name") or config.get("tokenizer_path") or config.get("model_path")
        if not tok_name:
            return None
        try:  # pragma: no cover - optional dependency
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(tok_name, trust_remote_code=bool(config.get("trust_remote_code", True)))
        except Exception as exc:
            logger.warning("Failed to load tokenizer '{}' for chat_template fallback: {}", tok_name, exc)
            return None

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

    def _check_tokenizer_conflict(self, payload: Dict[str, Any]) -> None:
        dataset_tok = payload.get("_tokenizer_path") or (payload.get("sample") or {}).get("_tokenizer_path")
        backend_tok = self._cfg_tokenizer_path
        if dataset_tok and backend_tok and str(dataset_tok) != str(backend_tok):
            raise ValueError(f"Conflicting tokenizer_path: dataset={dataset_tok} backend={backend_tok}")

    @staticmethod
    def _ensure_spawn_start_method() -> None:
        """Force spawn to avoid CUDA re-init in forked vLLM workers."""

        try:
            import torch.multiprocessing as mp  # type: ignore
        except Exception:  # pragma: no cover - torch missing
            import multiprocessing as mp  # type: ignore

        current = mp.get_start_method(allow_none=True)
        if current != "spawn":
            logger.info("vLLMNativeBackend switching multiprocessing start method to 'spawn' (was {})", current)
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
