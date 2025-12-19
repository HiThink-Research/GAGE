"""vLLM AsyncLLMEngine backend（对齐 TextGenerationMixin 请求字段，支持 sample_n 与多模态）。"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.backends.vllm_request import normalize_vllm_payload, resolve_sample_n
from gage_eval.role.model.runtime import BackendCapabilities, ChatTemplateMixin, ChatTemplatePolicy
from gage_eval.utils.chat_templates import get_fallback_template
from gage_eval.utils.cleanup import install_signal_cleanup, torch_gpu_cleanup


def _ensure_spawn_start_method() -> None:
    """Force spawn to avoid CUDA re-init issues in forked vLLM workers."""

    try:
        import torch.multiprocessing as mp  # type: ignore
    except Exception:  # pragma: no cover - torch missing
        import multiprocessing as mp  # type: ignore

    current = mp.get_start_method(allow_none=True)
    if current != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
        except Exception:
            logger.warning("vLLMBackend failed to set multiprocessing start method to spawn")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@registry.asset(
    "backends",
    "vllm",
    desc="vLLM 本地推理后端（AsyncLLMEngine，文本/多模态）",
    tags=("llm", "local", "serving"),
    modalities=("text", "vision"),
)
class VLLMBackend(EngineBackend, ChatTemplateMixin):
    """Backend that proxies requests to vLLM AsyncLLMEngine with unified request shaping."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._default_sampling = config.get("sampling_params") or {}
        self._max_tokens = int(config.get("max_tokens", 512))
        self._request_timeout = float(config.get("request_timeout", 300))
        self._cfg_tokenizer_path = config.get("tokenizer_path") or config.get("tokenizer_name")
        self._chat_template_mode = str(config.get("use_chat_template", "auto"))
        self._chat_template_policy = ChatTemplatePolicy(mode=self._chat_template_mode)
        self._fallback_template = get_fallback_template("text")
        self._tokenizer = None
        self._mm_supported = True
        self._mm_strategy = "inputs"  # inputs | disabled
        self._strict_mm = bool(
            config.get("strict_multi_modal")
            or os.environ.get("GAGE_EVAL_VLLM_STRICT_MM", "").lower() in {"1", "true", "yes", "on"}
        )
        if config.get("force_prompt_only") or os.environ.get("GAGE_EVAL_VLLM_PROMPT_ONLY", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            self._mm_supported = False
            self._mm_strategy = "disabled"

        _ensure_spawn_start_method()
        self._vllm_version = self._detect_version()
        if self._vllm_version:
            logger.info("Detected vLLM version {}", self._vllm_version)
            try:
                from packaging import version

                if version.parse(self._vllm_version) < version.parse("0.8.3"):
                    self._mm_supported = False
                    self._mm_strategy = "disabled"
                    logger.warning("vLLM < 0.8.3 detected; defaulting multi_modal_data unsupported")
            except Exception:
                pass

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, name="VLLMBackendLoop", daemon=True)
        self._loop_thread.start()
        cfg = dict(config)
        cfg.setdefault("execution_mode", "native")
        super().__init__(cfg)
        install_signal_cleanup(self.shutdown)

    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
            from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
        except ImportError as exc:
            raise RuntimeError("vllm is not installed") from exc

        self._tokenizer = self._init_tokenizer(config)
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("VLLMBackend requires 'model_path' in the backend config")

        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            enforce_eager=bool(config.get("enforce_eager", False)),
        )
        return AsyncLLMEngine.from_engine_args(engine_args)

    # ------------------------------ #
    # Request preparation
    # ------------------------------ #
    def prepare_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ctx = normalize_vllm_payload(payload)
        merged = dict(payload)
        merged.setdefault("sample", ctx.sample)
        merged["messages"] = ctx.messages
        merged["inputs"] = ctx.inputs
        merged.setdefault("prompt", ctx.prompt)
        merged.setdefault("prompt_meta", ctx.prompt_meta)
        if ctx.cache_namespace is not None:
            merged.setdefault("cache_namespace", ctx.cache_namespace)

        for key in ("chat_template_mode", "template_source", "rendered_by", "cache_suffix", "_tokenizer_path"):
            if key not in merged and key in ctx.sample:
                merged[key] = ctx.sample.get(key)

        self._check_tokenizer_conflict(merged)
        prompt = self._render_prompt(merged)
        mm_data = self._prepare_multi_modal_data(merged)
        sampling = self._build_sampling_params(ctx.sampling_params or {})
        request_id = self._resolve_request_id(merged)
        sample_n = resolve_sample_n(merged, ctx.sampling_params, default=1)
        caps = BackendCapabilities(supports_mm=self._mm_supported, has_processor_chat_template=False)
        cache_suffix = ChatTemplateMixin.get_cache_suffix("text", self._chat_template_policy, caps)
        return {
            "sample": ctx.sample,
            "messages": ctx.messages,
            "inputs": ctx.inputs,
            "prompt": prompt,
            "sampling_params": sampling,
            "multi_modal_data": mm_data,
            "request_id": request_id,
            "cache_suffix": cache_suffix,
            "prompt_meta": ctx.prompt_meta,
            "cache_namespace": ctx.cache_namespace,
            "sample_n": sample_n,
        }

    def _render_prompt(self, payload: Dict[str, Any]) -> str:
        messages = payload.get("messages") or (payload.get("sample") or {}).get("messages") or []
        raw_prompt = payload.get("prompt") or payload.get("text") or (payload.get("sample") or {}).get("prompt") or ""
        policy = self._chat_template_policy
        caps = BackendCapabilities(supports_mm=self._mm_supported, has_processor_chat_template=False)

        if not ChatTemplateMixin.should_render(payload, policy):
            return str(raw_prompt) if raw_prompt else self._simple_render(messages)

        template_source = ChatTemplateMixin.select_template("text", policy, caps)
        template_fn = getattr(self._tokenizer, "apply_chat_template", None) if self._tokenizer else None
        fallback_tpl = None if template_source == "model" else self._fallback_template
        rendered = ChatTemplateMixin.render(
            messages,
            template_fn=template_fn,
            fallback_fn=lambda msgs: self._fallback_render(msgs, fallback_tpl),
            add_generation_prompt=True,
            chat_template=fallback_tpl,
        )
        payload["chat_template_mode"] = "backend"
        payload["template_source"] = "model" if template_fn else "fallback"
        payload["rendered_by"] = "backend"
        return rendered or self._simple_render(messages) or str(raw_prompt)

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

    def _simple_render(self, messages: List[Dict[str, Any]]) -> str:
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

    def _build_sampling_params(self, runtime_params: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm is not installed") from exc
        params = dict(self._default_sampling)
        params.update(runtime_params or {})
        if "max_tokens" not in params and "max_new_tokens" in params:
            params["max_tokens"] = params.pop("max_new_tokens")
        params.setdefault("max_tokens", self._max_tokens)
        return SamplingParams(**params)

    # ------------------------------ #
    # Sync/Async execution wrappers
    # ------------------------------ #
    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_async(self._generate_async(inputs))

    def generate_batch(self, inputs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._run_async(self._generate_batch_async(inputs_list))

    def _run_async(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _generate_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prepared = self.prepare_inputs(inputs)
        return await self._generate_prepared(prepared, batch_path="native_single")

    async def _generate_batch_async(self, inputs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared_list = [self.prepare_inputs(item) for item in inputs_list]
        # sample_n>1 时逐条执行，避免 request_id 冲突
        if any((p.get("sample_n") or 1) > 1 for p in prepared_list):
            logger.debug("vLLMBackend: sample_n>1 detected in batch; falling back to per-request execution")
            return [await self._generate_prepared(item, batch_path="native_single") for item in prepared_list]

        tasks = []
        for item in prepared_list:
            sampling_params = item.get("sampling_params")
            request_id = item.get("request_id") or self._resolve_request_id(item)
            tasks.append(
                asyncio.wait_for(
                    self._generate_one(item, sampling_params, request_id),
                    timeout=self._request_timeout,
                )
            )
        try:
            raw_results = await asyncio.gather(*tasks)
        except asyncio.TimeoutError as exc:
            logger.warning("vLLM batch timeout: {}", exc)
            for item in prepared_list:
                rid = item.get("request_id")
                if rid:
                    with contextlib.suppress(Exception):
                        await self.model.abort(rid)
            raise

        finalized = []
        for prepared, raw in zip(prepared_list, raw_results):
            finalized.append(self._finalize_result(prepared, [raw], sample_n=1, batch_path="native_batch"))
        return finalized

    async def _generate_prepared(self, prepared: Dict[str, Any], *, batch_path: str) -> Dict[str, Any]:
        sampling_params = prepared.get("sampling_params")
        sample_n = max(int(prepared.get("sample_n") or 1), 1)
        request_id = prepared.get("request_id") or self._resolve_request_id(prepared)
        outputs: List[Any] = []
        for idx in range(sample_n):
            rid = request_id if sample_n == 1 else f"{request_id}_{idx}"
            try:
                result = await asyncio.wait_for(
                    self._generate_one(prepared, sampling_params, rid),
                    timeout=self._request_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("vLLM request_id={} timeout after {:.1f}s, aborting", rid, self._request_timeout)
                with contextlib.suppress(Exception):
                    await self.model.abort(rid)
                raise
            outputs.append(result)
        return self._finalize_result(prepared, outputs, sample_n=sample_n, batch_path=batch_path)

    async def _generate_one(self, prepared: Dict[str, Any], sampling_params, request_id: str):
        mm_data = prepared.get("multi_modal_data")
        generate_kwargs = {
            "prompt": prepared.get("prompt") or "",
            "sampling_params": sampling_params,
            "request_id": request_id,
        }
        try:
            if mm_data and self._mm_supported:
                if self._mm_strategy == "inputs":
                    generate_kwargs_mm = dict(generate_kwargs)
                    generate_kwargs_mm["inputs"] = {"prompt": generate_kwargs["prompt"], "multi_modal_data": mm_data}
                    result = self.model.generate(**generate_kwargs_mm)
                else:
                    if self._strict_mm:
                        raise RuntimeError("multi_modal_data disabled for this vLLM version")
                    logger.warning("vLLM multi_modal_data disabled; falling back to prompt-only")
                    result = self.model.generate(**generate_kwargs)
            else:
                result = self.model.generate(**generate_kwargs)
        except TypeError as exc:
            if mm_data and self._mm_supported:
                self._mm_supported = False
                if self._strict_mm:
                    raise RuntimeError(
                        f"multi_modal_data unsupported in this vLLM version; strict mode enabled ({exc})"
                    ) from exc
                logger.warning(
                    "multi_modal_data unsupported; fallback to prompt-only (error=%s)",
                    exc,
                )
                result = self.model.generate(**generate_kwargs)
            else:
                raise

        if hasattr(result, "__aiter__"):
            final = None
            async for item in result:
                final = item
            return final
        if asyncio.iscoroutine(result):
            return await result
        return result

    # ------------------------------ #
    # Output shaping
    # ------------------------------ #
    @staticmethod
    def _extract_answer(entry: Any) -> Any:
        if entry is None:
            return ""
        outputs = None
        if isinstance(entry, dict):
            outputs = entry.get("outputs")
            if outputs is None and "text" in entry:
                return entry.get("text")
        elif hasattr(entry, "outputs"):
            outputs = getattr(entry, "outputs") or []
        if outputs:
            texts = []
            for out in outputs:
                if isinstance(out, dict):
                    text_val = out.get("text") or out.get("output_text") or out
                else:
                    text_val = getattr(out, "text", None) or getattr(out, "output_text", None) or getattr(out, "generated_text", None)
                if text_val is None:
                    continue
                texts.append(text_val.strip() if isinstance(text_val, str) else str(text_val))
            if not texts:
                return ""
            return texts if len(texts) > 1 else texts[0]
        return str(entry)

    def _finalize_result(self, prepared: Dict[str, Any], outputs: List[Any], sample_n: int, *, batch_path: str) -> Dict[str, Any]:
        answers = [self._extract_answer(out) for out in outputs] if isinstance(outputs, list) else [self._extract_answer(outputs)]
        if sample_n <= 1:
            result = {"answer": answers[0] if answers else "", "_batch_path": batch_path}
        else:
            result = {"answer": answers, "_batch_path": batch_path, "_sample_n": sample_n}
        self._attach_template_metadata(prepared, result)
        if prepared.get("prompt_meta"):
            result["prompt_meta"] = prepared["prompt_meta"]
        if prepared.get("cache_namespace"):
            result["cache_namespace"] = prepared["cache_namespace"]
        return result

    def _attach_template_metadata(self, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
        meta_keys = ("chat_template_mode", "template_source", "rendered_by", "cache_suffix")
        sample = payload.get("sample") or {}
        for key in meta_keys:
            value = payload.get(key)
            if value is None:
                value = sample.get(key)
            if value is not None:
                result[key] = value
        if "_tokenizer_path" not in result and self._cfg_tokenizer_path:
            result["_tokenizer_path"] = self._cfg_tokenizer_path

    def shutdown(self) -> None:
        try:
            engine = getattr(self, "model", None)
            if engine and hasattr(engine, "shutdown"):
                engine.shutdown()
        except Exception:
            pass
        try:
            if getattr(self, "_loop", None):
                self._loop.call_soon_threadsafe(self._loop.stop)
            if getattr(self, "_loop_thread", None):
                self._loop_thread.join(timeout=1.0)
        except Exception:
            logger.warning("VLLMBackend shutdown encountered an error", exc_info=True)
        torch_gpu_cleanup()

    # ------------------------------ #
    # Multi-modal helpers
    # ------------------------------ #
    def _prepare_multi_modal_data(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sample = payload.get("sample") or {}
        raw_inputs = payload.get("inputs") or sample.get("inputs") or {}
        mm = raw_inputs.get("multi_modal_data") if isinstance(raw_inputs, dict) else None
        if not mm:
            mm = sample.get("multi_modal_data")

        images: List[Any] = []
        if isinstance(mm, dict):
            images.extend(self._load_images(mm.get("image") or mm.get("images")))
        messages = payload.get("messages") or sample.get("messages") or []
        images.extend(self._load_images(self._extract_images_from_messages(messages)))

        images = [img for img in images if img is not None]
        if not images:
            return None
        return {"image": images}

    @staticmethod
    def _extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
        urls: List[str] = []
        for message in messages or []:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for fragment in content:
                if isinstance(fragment, dict) and fragment.get("type") == "image_url":
                    url = (
                        fragment.get("image_url", {}).get("url")
                        if isinstance(fragment.get("image_url"), dict)
                        else fragment.get("image_url")
                        or fragment.get("url")
                    )
                    if isinstance(url, str):
                        urls.append(url)
        return urls

    def _load_images(self, sources) -> List[Any]:
        if sources is None:
            return []
        if not isinstance(sources, list):
            sources = [sources]
        images: List[Any] = []
        try:  # pragma: no cover - heavy dependency
            from PIL import Image
        except ImportError:
            logger.warning("Pillow not installed; skipping image loading for multi_modal_data")
            return []

        for src in sources:
            if src is None:
                continue
            try:
                if isinstance(src, Image.Image):
                    images.append(src)
                elif isinstance(src, str) and src.startswith("data:"):
                    _, b64 = src.split(",", 1)
                    binary = base64.b64decode(b64)
                    images.append(Image.open(io.BytesIO(binary)).convert("RGB"))
                elif isinstance(src, str):
                    path = Path(src)
                    images.append(Image.open(path).convert("RGB"))
                else:
                    logger.debug("Unsupported image source type {}; skipping", type(src))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load image source {}: {}", src, exc)
        return images

    # ------------------------------ #
    # Misc helpers
    # ------------------------------ #
    def _resolve_request_id(self, payload: Dict[str, Any]) -> str:
        candidate = payload.get("request_id")
        if not candidate:
            sample = payload.get("sample") or {}
            if isinstance(sample, dict):
                for key in ("request_id", "id", "sample_id", "idx"):
                    candidate = sample.get(key)
                    if candidate:
                        break
        if not candidate:
            candidate = payload.get("id") or payload.get("sample_id")

        request_id = str(candidate).strip() if candidate is not None else ""
        if not request_id:
            request_id = f"gage_vllm_{uuid.uuid4().hex}"
        return request_id

    def _check_tokenizer_conflict(self, payload: Dict[str, Any]) -> None:
        dataset_tok = payload.get("_tokenizer_path") or (payload.get("sample") or {}).get("_tokenizer_path")
        backend_tok = self._cfg_tokenizer_path
        if dataset_tok and backend_tok and str(dataset_tok) != str(backend_tok):
            raise ValueError(f"Conflicting tokenizer_path: dataset={dataset_tok} backend={backend_tok}")

    @staticmethod
    def _detect_version() -> Optional[str]:
        try:  # pragma: no cover - optional dependency
            import vllm  # type: ignore
        except Exception:
            logger.warning("vLLM not installed; version detection skipped")
            return None
        ver = getattr(vllm, "__version__", None)
        if isinstance(ver, str):
            return ver
        try:
            from importlib.metadata import version as meta_version

            return meta_version("vllm")
        except Exception:
            return None
