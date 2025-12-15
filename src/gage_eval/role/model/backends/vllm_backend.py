"""VLLM backend adaptor（本地 AsyncLLMEngine，文本+多模态，迁移 llm-eval 思路，非 HTTP 客户端）。"""

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
    desc="vLLM 本地推理后端（tensor parallel 支持，多模态可选）",
    tags=("llm", "local", "serving"),
    modalities=("text", "vision"),
)
class VLLMBackend(EngineBackend):
    """Backend that proxies requests to vLLM AsyncLLMEngine (in-process)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._default_sampling = config.get("sampling_params") or {}
        self._max_tokens = int(config.get("max_tokens", 512))
        self._request_timeout = float(config.get("request_timeout", 300))
        self._mm_supported = True  # 版本探测后可能关闭，遇到 TypeError 也会降级
        self._mm_strategy = "inputs"  # inputs | disabled
        self._list_sampling_supported = True
        self._cfg_tokenizer_path = config.get("tokenizer_path") or config.get("tokenizer_name")
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
            # vLLM 0.8.3 之前多模态支持不稳定，先默认关闭再由运行时探针拉起
            try:
                from packaging import version

                if version.parse(self._vllm_version) < version.parse("0.8.3"):
                    self._mm_supported = False
                    self._mm_strategy = "disabled"
                    logger.warning("vLLM < 0.8.3 detected; defaulting multi_modal_data unsupported")
            except Exception:
                pass
        else:
            self._mm_strategy = "inputs"
        # 持久事件循环线程，避免每次调用 run_until_complete 破坏 vLLM 合批
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, name="VLLMBackendLoop", daemon=True)
        self._loop_thread.start()
        self._chat_template_mode = str(config.get("use_chat_template", "auto"))
        self._chat_template_policy = ChatTemplatePolicy(mode=self._chat_template_mode)
        self._fallback_template = get_fallback_template("text")
        self._tokenizer = None  # optional AutoTokenizer for fallback rendering
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

        # optional tokenizer for chat template fallback
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

    def prepare_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Render prompt +采样参数，并准备 multi_modal_data."""

        self._check_tokenizer_conflict(payload)
        prompt = self._render_prompt(payload)
        sampling = dict(self._default_sampling)
        sampling.update(payload.get("sampling_params") or {})
        mm_data = self._prepare_multi_modal_data(payload)
        request_id = self._resolve_request_id(payload)
        caps = BackendCapabilities(supports_mm=self._mm_supported, has_processor_chat_template=False)
        cache_suffix = ChatTemplateMixin.get_cache_suffix("text", self._chat_template_policy, caps)
        return {
            "prompt": prompt,
            "sampling_params": sampling,
            "multi_modal_data": mm_data,
            "request_id": request_id,
            "cache_suffix": cache_suffix,
        }

    def _resolve_request_id(self, payload: Dict[str, Any]) -> str:
        """Ensure every request has a non-empty, unique id for AsyncLLMEngine."""

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

    def _render_prompt(self, payload: Dict[str, Any]) -> str:
        messages = payload.get("messages") or (payload.get("sample") or {}).get("messages") or []
        raw_prompt = payload.get("prompt") or payload.get("text") or (payload.get("sample") or {}).get("prompt") or ""
        policy = ChatTemplatePolicy(mode=self._chat_template_mode)
        caps = BackendCapabilities(supports_mm=self._mm_supported, has_processor_chat_template=False)

        # 如果已渲染或禁用，回退到原始 prompt/简易拼接
        if not ChatTemplateMixin.should_render(payload, policy):
            if raw_prompt:
                return str(raw_prompt)
            return self._simple_render(messages)

        # 选择模板来源
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

    def shutdown(self) -> None:  # pragma: no cover - best-effort GPU cleanup
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
            pass
        torch_gpu_cleanup()

    def _fallback_render(self, messages: List[Dict[str, Any]], tpl: Optional[str]) -> str:
        # 简单文本拼接；若提供兜底模板字符串，可选择扩展为真正模板渲染
        return self._simple_render(messages)

    def _build_sampling_params(self, runtime_params: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm is not installed") from exc
        params = dict(self._default_sampling)
        params.update(runtime_params or {})
        params.setdefault("max_tokens", self._max_tokens)
        return SamplingParams(**params)

    # ------------------------------ #
    # Sync wrappers for async engine #
    # ------------------------------ #
    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_async(self._generate_async(inputs))

    def generate_batch(self, inputs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._run_async(self._generate_batch_async(inputs_list))

    def _run_async(self, coro):
        # 始终将任务投递到持久事件循环线程，避免在调用线程创建/关闭事件循环。
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _generate_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """单请求 async 路径，支持文本/多模态。"""

        prepared = self.prepare_inputs(inputs)
        sampling_params = self._build_sampling_params(prepared.get("sampling_params") or {})
        request_id = prepared.get("request_id") or self._resolve_request_id(prepared)
        try:
            result = await asyncio.wait_for(
                self._generate_one(prepared, sampling_params, request_id),
                timeout=self._request_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("vLLM request_id={} timeout after {:.1f}s, aborting", request_id, self._request_timeout)
            with contextlib.suppress(Exception):
                await self.model.abort(request_id)
            raise
        final = self._extract_first([result], batch_path="native_batch")
        self._attach_template_metadata(prepared, final)
        return final

    async def _generate_batch_async(self, inputs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量并发提交 AsyncLLMEngine（vLLM 自身负责合批）。"""

        prepared = [self.prepare_inputs(item) for item in inputs_list]
        tasks = []
        for idx, item in enumerate(prepared):
            sampling_params = self._build_sampling_params(item.get("sampling_params") or {})
            request_id = item.get("request_id") or self._resolve_request_id(item)
            tasks.append(
                asyncio.wait_for(
                    self._generate_one(item, sampling_params, request_id),
                    timeout=self._request_timeout,
                )
            )
        try:
            results = await asyncio.gather(*tasks)
        except asyncio.TimeoutError as exc:
            logger.warning("vLLM batch timeout: {}", exc)
            for item in prepared:
                rid = item.get("request_id") or (item.get("sample") or {}).get("idx")
                if rid:
                    with contextlib.suppress(Exception):
                        await self.model.abort(rid)
            raise
        enriched = []
        for prepared_item, out in zip(prepared, results):
            result = self._extract_first([out], batch_path="native_batch")
            self._attach_template_metadata(prepared_item, result)
            enriched.append(result)
        return enriched

    async def _generate_one(self, prepared: Dict[str, Any], sampling_params, request_id: str):
        mm_data = prepared.get("multi_modal_data")
        generate_kwargs = dict(
            prompt=prepared.get("prompt") or "",
            sampling_params=sampling_params,
            request_id=request_id,
        )

        # 多模态尝试：部分 vLLM 不支持 multi_modal_data/inputs，会 TypeError，需降级。
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
                        f"vLLM backend does not support multi_modal_data in this version; strict mode enabled ({exc})"
                    ) from exc
                logger.warning(
                    "vLLM backend detected multi_modal_data unsupported in this vLLM version; falling back to prompt-only (images ignored). Error: %s",
                    exc,
                )
                result = self.model.generate(**generate_kwargs)
            else:
                raise

        # vLLM 0.8.x AsyncLLMEngine.generate 可能返回 async generator；需要异步迭代取最终结果
        if hasattr(result, "__aiter__"):
            final = None
            async for item in result:
                final = item
            return final
        # 或返回 awaitable/RequestOutput
        if asyncio.iscoroutine(result):
            return await result
        return result

    def shutdown(self) -> None:
        """Stop内部事件循环，释放资源。"""

        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=1.0)
        except Exception:
            logger.warning("VLLMBackend shutdown encountered an error", exc_info=True)

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
            if value is not None:
                result[key] = value
        if "_tokenizer_path" not in result and self._cfg_tokenizer_path:
            result["_tokenizer_path"] = self._cfg_tokenizer_path

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

        # 去除空条目
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
                    url = fragment.get("image_url", {}).get("url") if isinstance(fragment.get("image_url"), dict) else fragment.get("image_url") or fragment.get("url")
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

    def _check_tokenizer_conflict(self, payload: Dict[str, Any]) -> None:
        dataset_tok = payload.get("_tokenizer_path") or (payload.get("sample") or {}).get("_tokenizer_path")
        backend_tok = self._cfg_tokenizer_path
        if dataset_tok and backend_tok and str(dataset_tok) != str(backend_tok):
            raise ValueError(f"Conflicting tokenizer_path: dataset={dataset_tok} backend={backend_tok}")
