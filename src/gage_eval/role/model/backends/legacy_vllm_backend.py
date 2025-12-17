"""Legacy vLLM backend skeleton: gage 架构 + llm-eval 兼容功能的承载体."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import types
import inspect
import uuid
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.runtime import BackendCapabilities, ChatTemplateMixin, ChatTemplatePolicy
from gage_eval.utils.chat_templates import get_fallback_template
from gage_eval.utils.cleanup import install_signal_cleanup, torch_gpu_cleanup
from gage_eval.utils.multimodal import load_multimodal_data

try:  # pragma: no cover - optional dependency
    import vllm  # type: ignore
    from packaging import version as _pkg_version
    _VLLM_VERSION = getattr(vllm, "__version__", None)
    _VLLM_PROMPT_V1 = bool(_VLLM_VERSION) and _pkg_version.parse(_VLLM_VERSION) >= _pkg_version.parse("0.8.0")
except Exception:  # pragma: no cover
    _VLLM_VERSION = None
    _VLLM_PROMPT_V1 = False


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
            logger.warning("legacy_vllm_backend failed to set multiprocessing start method to spawn")
    os = __import__("os")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


import threading

@registry.asset(
    "backends",
    "legacy_vllm",
    desc="vLLM 兼容版后端（将逐步覆盖 llm-eval 能力）",
    tags=("llm", "local", "serving"),
    modalities=("text", "vision", "audio"),
)
class LegacyVLLMBackend(EngineBackend, ChatTemplateMixin):
    """Stub backend to host legacy vLLM features (output types/multimodal/patches)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._chat_template_mode = str(config.get("use_chat_template", "auto"))
        self._chat_template_policy = ChatTemplatePolicy(mode=self._chat_template_mode)
        self._fallback_template = get_fallback_template("text")
        self._tokenizer = None
        self._processor = None
        self._cfg_tokenizer_path = config.get("tokenizer_path") or config.get("tokenizer_name")
        self._force_tokenize_prompt = bool(config.get("force_tokenize_prompt"))
        self._default_sampling = config.get("sampling_params") or {}
        self._max_tokens = int(config.get("max_tokens", 512))
        cfg = dict(config)
        cfg.setdefault("execution_mode", "native")
        _ensure_spawn_start_method()
        
        # 初始化后台事件循环线程，避免在主线程反复创建 loop 导致死锁
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, name="LegacyVLLMLoop", daemon=True)
        self._loop_thread.start()
        
        super().__init__(cfg)
        install_signal_cleanup(self.shutdown)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def shutdown(self) -> None:
        try:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread.is_alive():
                self._loop_thread.join(timeout=1.0)
        except Exception:
            logger.warning("LegacyVLLMBackend shutdown error", exc_info=True)
        try:
            model = getattr(self, "model", None)
            if model and hasattr(model, "shutdown"):
                model.shutdown()
        except Exception:
            pass
        torch_gpu_cleanup()

    def load_model(self, config: Dict[str, Any]):
        """加载模型 + processor，并应用兼容性补丁（奖励/MoE/rope scaling/低显存等）。"""

        trust_remote_code = bool(config.get("trust_remote_code", True))
        model_id = config.get("model_path") or config.get("model")
        if not model_id:
            raise ValueError("legacy_vllm_backend requires 'model_path' or 'model' in config")

        output_type = config.get("output_type", "text")
        if output_type == "cosyvoice2" or config.get("cosyvoice2"):
            engine, processor = self._load_cosyvoice2(config)
            self._processor = processor
            return engine
        args_ns = self._build_args_namespace(config, output_type=output_type, trust_remote_code=trust_remote_code)
        cfg_obj = self._load_auto_config(model_id, trust_remote_code=trust_remote_code)
        self._processor = self._load_auto_processor(model_id, trust_remote_code=trust_remote_code)
        self._tokenizer = self._init_tokenizer(config)
        cfg_obj = self._apply_model_patches(cfg_obj, args_ns)
        engine, processor = self._build_engine(cfg_obj, self._processor, model_id, args_ns)
        self._processor = processor
        return engine

    def prepare_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # 将样本侧的模板标记透传进 payload，避免已在预处理阶段渲染的提示被后台重复渲染/改写元数据。
        sample = payload.get("sample") or {}
        for key in ("chat_template_mode", "template_source", "rendered_by", "cache_suffix", "_tokenizer_path"):
            if key not in payload and key in sample:
                payload[key] = sample.get(key)

        self._check_tokenizer_conflict(payload)
        chat_template_kwargs = payload.get("chat_template_kwargs") or sample.get("chat_template_kwargs")
        prompt = self._render_prompt(payload)
        prompt, inputs_val = self._maybe_tokenize_messages(payload, prompt)
        output_type = payload.get("output_type", "text")
        sampling = self._build_sampling_params(output_type, payload.get("sampling_params") or {})
        sample_n = int(payload.get("sample_n") or payload.get("generation_params", {}).get("n") or 1)
        request_id = self._resolve_request_id(payload)
        caps = BackendCapabilities(supports_mm=True, has_processor_chat_template=bool(self._processor))
        cache_suffix = ChatTemplateMixin.get_cache_suffix("text", self._chat_template_policy, caps)
        template_source = payload.get("template_source")
        rendered_by = payload.get("rendered_by")
        chat_mode = payload.get("chat_template_mode")
        return {
            # 保留原始样本/消息/输入，便于下游多模态与元数据处理
            "sample": payload.get("sample"),
            "messages": payload.get("messages") or (payload.get("sample") or {}).get("messages"),
            "inputs": payload.get("inputs") or (payload.get("sample") or {}).get("inputs"),
            "prompt": prompt,
            "sampling_params": sampling,
            "request_id": request_id,
            "cache_suffix": cache_suffix,
            "output_type": output_type,
            "sample_n": sample_n,
            "template_source": template_source,
            "rendered_by": rendered_by,
            "chat_template_mode": chat_mode,
            # "chat_template_kwargs": chat_template_kwargs,
            "messages": payload.get("messages") or (payload.get("sample") or {}).get("messages") or [],
        }

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """同步包装，支持 sample_n 多路采样与 output_type 路由."""

        output_type = inputs.get("output_type", "text")
        sample_n = int(inputs.get("sample_n") or 1)
        sampling_params = inputs.get("sampling_params") or self._build_sampling_params(output_type, {})
        base_request_id = inputs.get("request_id") or self._resolve_request_id(inputs)

        if sample_n > 1:
            results = []
            for i in range(sample_n):
                rid = f"{base_request_id}_{i}"
                result = self._generate_one(inputs, sampling_params, rid)
                results.append(self._convert_output(result, output_type))
            final = {"answer": results, "_backend": "legacy_vllm_stub", "_sample_n": sample_n}
            self._attach_template_metadata(inputs, final)
            return final

        request_id = base_request_id
        result = self._generate_one(inputs, sampling_params, request_id)
        converted = self._convert_output(result, output_type)
        final = {"answer": converted, "_backend": "legacy_vllm_stub"}
        self._attach_template_metadata(inputs, final)
        return final

    def _generate_one(self, prepared: Dict[str, Any], sampling_params: Any, request_id: str) -> Any:
        prompt = prepared.get("prompt") or ""
        # 注意：这里 _load_multimodal_payload 内部已经使用了线程池，是线程安全的同步调用
        mm_raw = self._prepare_multi_modal_data(prepared)
        mm_loaded = self._load_multimodal_payload(mm_raw)
        messages = prepared.get("messages") or []

        # 定义异步任务，将在后台 loop 中运行
        async def _async_generate():
            # 构造 prompt，优先 processor 渲染
            rendered_mm = self._render_with_processor(messages, prompt, prepared.get("chat_template_kwargs"))
            final_prompt_text = rendered_mm if rendered_mm else prompt

            # vLLM v1（>=0.8.0）接口支持单个 PromptInputs dict 作为位置参数；旧接口只接受 keyword prompt/inputs
            use_v1_prompt = bool(_VLLM_PROMPT_V1)
            prompt_input: Any
            mm_payload = mm_loaded
            if mm_payload:
                # 确保 prompt 中包含足够的 <image> 占位，防止 vLLM 提示更新不匹配
                image_field = mm_payload.get("image") if isinstance(mm_payload, dict) else None
                if isinstance(image_field, (list, tuple)):
                    image_count = len(image_field)
                elif image_field is None:
                    image_count = 0
                else:
                    image_count = 1
                final_prompt_text = self._normalize_image_placeholders(final_prompt_text, image_count)
                prompt_input = {"prompt": final_prompt_text, "multi_modal_data": mm_payload}
            else:
                prompt_input = final_prompt_text

            extra_inputs = prepared.get("inputs") or prepared.get("prompt_token_ids")
            if isinstance(extra_inputs, dict):
                # 如有 input_ids/prompt_token_ids，则构造 dict 以走 v1 PromptInputs，兼容 llm-eval 行为
                needs_dict = mm_loaded or use_v1_prompt or "input_ids" in extra_inputs or "prompt_token_ids" in extra_inputs
                if needs_dict and not isinstance(prompt_input, dict):
                    prompt_input = {"prompt": final_prompt_text}
                if isinstance(prompt_input, dict):
                    mapped_extra = dict(extra_inputs)
                    if "input_ids" in mapped_extra and "prompt_token_ids" not in mapped_extra:
                        mapped_extra["prompt_token_ids"] = mapped_extra.get("input_ids")
                    for k, v in mapped_extra.items():
                        prompt_input.setdefault(k, v)

            generate_kwargs = {
                "sampling_params": sampling_params,
                "request_id": request_id,
            }
            if use_v1_prompt:
                generate_args = (prompt_input,)
            else:
                if isinstance(prompt_input, dict):
                    generate_kwargs["inputs"] = prompt_input
                    generate_kwargs["prompt"] = prompt_input.get("prompt", prompt)
                else:
                    generate_kwargs["prompt"] = prompt_input
                if isinstance(extra_inputs, dict):
                    mapped_extra = dict(extra_inputs)
                    if "input_ids" in mapped_extra and "prompt_token_ids" not in mapped_extra:
                        mapped_extra["prompt_token_ids"] = mapped_extra.get("input_ids")
                    for k, v in mapped_extra.items():
                        generate_kwargs.setdefault(k, v)
                
                # 防御性修正：确保 prompt_token_ids 是列表，防止 int 导致 len() 报错
                if "prompt_token_ids" in generate_kwargs:
                    p_ids = generate_kwargs["prompt_token_ids"]
                    if isinstance(p_ids, int):
                        generate_kwargs["prompt_token_ids"] = [p_ids]
                
                generate_args = ()

            try:
                result = self.model.generate(*generate_args, **generate_kwargs)
            except TypeError as exc:
                # 旧版 AsyncLLMEngine.generate 不支持 multi_modal_data/inputs 关键字或 v1 位置参数
                if mm_loaded or generate_args:
                    logger.warning(
                        "legacy_vllm_backend generate encountered TypeError ({}); "
                        "falling back to prompt-only without multi_modal_data",
                        exc,
                    )
                    prompt_for_retry = prompt_input.get("prompt") if isinstance(prompt_input, dict) else prompt_input
                    result = self.model.generate(prompt=prompt_for_retry or prompt, sampling_params=sampling_params, request_id=request_id)
                else:
                    raise
            except Exception as exc:
                logger.warning("legacy_vllm_backend generate failed: {} - {}; returning echo output", type(exc).__name__, exc, exc_info=True)
                return {"outputs": [{"text": str(prompt)}]}

            # 异步收集结果
            if inspect.isasyncgen(result) or isinstance(result, types.AsyncGeneratorType):
                final_item = None
                async for item in result:
                    final_item = item
                return final_item
            elif inspect.iscoroutine(result):
                return await result
            return result

        # 提交到后台线程执行并等待结果
        future = asyncio.run_coroutine_threadsafe(_async_generate(), self._loop)
        try:
            return future.result()
        except Exception as exc:
            logger.error("legacy_vllm_backend: background thread execution failed: {}", exc)
            return {"outputs": [{"text": str(prompt)}]}

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

    def _convert_output(self, result: Any, output_type: str) -> Any:
        if output_type == "cosyvoice2":
            return self._convert_output_cosyvoice(result)

        if isinstance(result, list):
            return [self._convert_output(item, output_type) for item in result]

        if output_type in {"text", "beam"}:
            if isinstance(result, dict):
                if "outputs" in result and result["outputs"]:
                    out = result["outputs"][0]
                    return out.get("text") or out
                if "text" in result:
                    return result["text"]
            # vLLM RequestOutput 对象
            if hasattr(result, "outputs"):
                outputs = getattr(result, "outputs") or []
                if outputs:
                    first = outputs[0]
                    if isinstance(first, dict):
                        return first.get("text") or first
                    if hasattr(first, "text"):
                        return getattr(first, "text")
            return str(result)

        if output_type in {"reward", "loss", "prompt_tokens", "next_token_prob"}:
            return result

        return result

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
            request_id = f"legacy_vllm_{uuid.uuid4().hex}"
        return request_id

    def _render_prompt(self, payload: Dict[str, Any]) -> str:
        messages = payload.get("messages") or (payload.get("sample") or {}).get("messages") or []
        raw_prompt = payload.get("prompt") or payload.get("text") or (payload.get("sample") or {}).get("prompt")
        if not raw_prompt:
            inputs = payload.get("inputs") or (payload.get("sample") or {}).get("inputs")
            if isinstance(inputs, dict):
                raw_prompt = inputs.get("prompt")
        raw_prompt = raw_prompt or ""
        policy = ChatTemplatePolicy(mode=self._chat_template_mode)
        caps = BackendCapabilities(supports_mm=True, has_processor_chat_template=bool(self._processor))
        chat_kwargs = payload.get("chat_template_kwargs") or {}

        if not ChatTemplateMixin.should_render(payload, policy):
            if raw_prompt:
                return str(raw_prompt)
            return self._simple_render(messages)

        template_source = ChatTemplateMixin.select_template("text", policy, caps)
        template_fn = getattr(self._tokenizer, "apply_chat_template", None) if self._tokenizer else None
        fallback_tpl = None if template_source == "model" else self._fallback_template
        rendered = ChatTemplateMixin.render(
            messages,
            template_fn=template_fn,
            fallback_fn=lambda msgs: self._fallback_render(msgs, fallback_tpl),
            add_generation_prompt=True,
            chat_template=fallback_tpl,
            **chat_kwargs,
        )
        payload["chat_template_mode"] = "backend"
        payload["template_source"] = "model" if template_fn else "fallback"
        payload["rendered_by"] = "backend"
        return rendered or self._simple_render(messages) or str(raw_prompt)

    def _maybe_tokenize_messages(self, payload: Dict[str, Any], prompt: str) -> Tuple[str, Any]:
        """使用 backend tokenizer 生成 prompt_token_ids（若预处理器未提供）。"""

        inputs = payload.get("inputs") or (payload.get("sample") or {}).get("inputs") or {}
        if isinstance(inputs, dict) and inputs.get("prompt_token_ids"):
            return prompt, inputs
        if not self._tokenizer and not (self._processor and hasattr(self._processor, "apply_chat_template")):
            return prompt, inputs

        policy = self._chat_template_policy
        force_tokenize = bool(payload.get("force_tokenize_prompt") or self._force_tokenize_prompt)
        if not ChatTemplateMixin.should_render(payload, policy) and not force_tokenize:
            return prompt, inputs

        # 默认仅在多模态场景分词，避免纯文本重复调用 tokenizer
        if not ChatTemplateMixin.detect_multimodal(payload) and not force_tokenize:
            return prompt, inputs

        messages = payload.get("messages") or (payload.get("sample") or {}).get("messages") or []
        if not isinstance(messages, list) or not messages:
            return prompt, inputs

        try:
            chat_template_fn = None
            # 优先使用 processor（多模态安全），否则回退 tokenizer
            if self._processor and hasattr(self._processor, "apply_chat_template"):
                chat_template_fn = self._processor.apply_chat_template
            elif self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
                chat_template_fn = self._tokenizer.apply_chat_template

            def _strip_non_text(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """移除非文本片段，避免 tokenizer 不支持 image/audio 时失败。"""
                sanitized: List[Dict[str, Any]] = []
                for m in msgs:
                    content = m.get("content")
                    if isinstance(content, list):
                        texts = []
                        for frag in content:
                            if isinstance(frag, dict) and frag.get("type") == "text":
                                texts.append(str(frag.get("text", "")))
                        content = [{"type": "text", "text": " ".join(texts)}]
                    sanitized.append({"role": m.get("role", "user"), "content": content})
                return sanitized

            sanitized_messages = messages
            rendered = None
            tokenized = None
            if chat_template_fn:
                try:
                    rendered = chat_template_fn(messages, tokenize=False, add_generation_prompt=True)
                    tokenized = chat_template_fn(messages, tokenize=True, add_generation_prompt=True)
                except Exception:
                    # 对多模态不支持的 tokenizer，降级为文本-only 拼接再分词
                    sanitized_messages = _strip_non_text(messages)
                    rendered = chat_template_fn(sanitized_messages, tokenize=False, add_generation_prompt=True)
                    tokenized = chat_template_fn(sanitized_messages, tokenize=True, add_generation_prompt=True)
            if isinstance(tokenized, list):
                first = tokenized[0] if tokenized else []
                token_ids = first if isinstance(first, (list, tuple)) else tokenized
            else:
                token_ids = tokenized
            new_prompt = str(rendered) if rendered else prompt
            if not isinstance(inputs, dict):
                inputs = {}
            inputs = dict(inputs)
            inputs["prompt"] = new_prompt
            if token_ids is not None:
                inputs["prompt_token_ids"] = token_ids
            payload.setdefault("template_source", "model")
            payload.setdefault("rendered_by", "backend")
            payload.setdefault("chat_template_mode", "backend")
            payload.setdefault("cache_suffix", "-chat_template")
            return new_prompt, inputs
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("legacy_vllm_backend: apply_chat_template tokenize failed, skip prompt_token_ids: {}", exc)
            return prompt, inputs

    def _fallback_render(self, messages: List[Dict[str, Any]], tpl: Optional[str]) -> str:
        return self._simple_render(messages)

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
                    if isinstance(fragment, dict):
                        if fragment.get("type") == "text":
                            text_parts.append(str(fragment.get("text", "")))
                        elif fragment.get("type") in {"image", "image_url"}:
                            text_parts.append("<image>")
                text = " ".join(text_parts)
            else:
                text = str(content) if content is not None else ""
            segments.append(f"{role}: {text}".strip())
        segments.append("assistant:")
        return "\n".join(segments)

    @staticmethod
    def _normalize_messages_for_template(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten list-based content so chat templates see strings instead of lists."""

        normalized: List[Dict[str, Any]] = []
        for message in messages or []:
            item = dict(message)
            content = item.get("content")
            if isinstance(content, list):
                parts: List[str] = []
                for fragment in content:
                    if isinstance(fragment, dict):
                        if fragment.get("type") == "text":
                            parts.append(str(fragment.get("text", "")))
                        elif fragment.get("type") in {"image", "image_url"}:
                            parts.append("<image>")
                    elif fragment is not None:
                        parts.append(str(fragment))
                item["content"] = " ".join(part for part in parts if part).strip()
            elif content is None:
                item["content"] = ""
            normalized.append(item)
        return normalized

    def _render_with_processor(
        self, messages: List[Dict[str, Any]], prompt: str, chat_template_kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        apply_template = getattr(self._processor, "apply_chat_template", None) if self._processor else None
        if not apply_template:
            return None
        try:
            norm_messages = self._normalize_messages_for_template(messages)
            rendered = apply_template(norm_messages, add_generation_prompt=True, tokenize=False, **(chat_template_kwargs or {}))
            if isinstance(rendered, list):
                rendered = rendered[0]
            return str(rendered) if rendered else prompt
        except Exception as exc:
            logger.debug("legacy_vllm_backend processor chat_template failed: {}", exc)
            tok = getattr(self, "_tokenizer", None)
            tok_apply = getattr(tok, "apply_chat_template", None)
            if tok_apply:
                try:
                    rendered = tok_apply(
                        self._normalize_messages_for_template(messages),
                        add_generation_prompt=True,
                        tokenize=False,
                        **(chat_template_kwargs or {}),
                    )
                    if isinstance(rendered, list):
                        rendered = rendered[0]
                    if rendered:
                        return str(rendered)
                except Exception as tok_exc:
                    logger.debug("legacy_vllm_backend tokenizer chat_template failed: {}", tok_exc)
            return None

    def _check_tokenizer_conflict(self, payload: Dict[str, Any]) -> None:
        dataset_tok = payload.get("_tokenizer_path") or (payload.get("sample") or {}).get("_tokenizer_path")
        backend_tok = self._cfg_tokenizer_path
        if dataset_tok and backend_tok and str(dataset_tok) != str(backend_tok):
            raise ValueError(f"Conflicting tokenizer_path: dataset={dataset_tok} backend={backend_tok}")

    @staticmethod
    def _normalize_image_placeholders(prompt: str, image_count: int) -> str:
        """确保 prompt 中有足够的 <image> 占位符以匹配多模态数据。"""

        marker = "<image>"
        if not prompt:
            prompt = ""
        normalized = re.sub(r"<image\s*\d*>", marker, prompt, flags=re.IGNORECASE)
        current = normalized.lower().count(marker)
        missing = max(0, image_count - current)
        if missing > 0:
            prefix = " ".join([marker] * missing)
            normalized = (prefix + " " + normalized).strip()
        return normalized

    def _build_sampling_params(self, output_type: str, runtime_params: Dict[str, Any]):
        base = dict(self._default_sampling)
        base.update(runtime_params or {})

        if output_type == "loss":
            base.setdefault("temperature", 0)
            base.setdefault("prompt_logprobs", 1)
            base["max_tokens"] = 1
        elif output_type == "prompt_tokens":
            base.setdefault("temperature", 0)
            base.setdefault("prompt_logprobs", base.get("prompt_logprobs", 20))
            base["max_tokens"] = 1
        elif output_type == "next_token_prob":
            base.setdefault("temperature", 0)
            base["max_tokens"] = 1
            top_lp = base.pop("top_logprobs_num", None) or base.get("logprobs")
            if top_lp is not None:
                base["logprobs"] = top_lp
        elif output_type == "cosyvoice2":
            base.setdefault("temperature", 1)
            base.setdefault("top_p", 1)
            base.setdefault("top_k", 25)
            base.setdefault("max_tokens", base.get("max_tokens", 2048))
        else:
            base.setdefault("max_tokens", self._max_tokens)

        sampling_cls = self._resolve_sampling_class()
        return sampling_cls(**base)

    @staticmethod
    def _resolve_sampling_class():
        try:  # pragma: no cover - optional dependency
            from vllm import SamplingParams  # type: ignore

            return SamplingParams
        except Exception:
            # 轻量级占位，便于单测检查属性
            class SamplingParams:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

                def __repr__(self) -> str:
                    return f"SamplingParams({self.__dict__})"

            return SamplingParams

    # ------------------------------------------------------------------ #
    # 模型加载与补丁                                                         #
    # ------------------------------------------------------------------ #
    def _build_args_namespace(self, config: Dict[str, Any], *, output_type: str, trust_remote_code: bool):
        args = SimpleNamespace()
        args.model = config.get("model_path") or config.get("model")
        args.output_type = output_type
        args.trust_remote_code = trust_remote_code
        args.max_length = config.get("max_model_len") or config.get("max_length")
        args.low_vram = bool(config.get("low_vram", False))
        args.block_size = config.get("block_size")
        args.tensor_parallel_size = int(config.get("tensor_parallel_size", config.get("tensor_parallel", 1)))
        args.pipeline_parallel = int(config.get("pipeline_parallel", 1))
        args.pipeline_parallel_size = int(config.get("pipeline_parallel_size", args.pipeline_parallel))
        args.enable_expert_parallel = config.get("enable_expert_parallel")
        args.enforce_eager = config.get("enforce_eager")
        # placeholders for cache block hints
        args.num_gpu_blocks = config.get("num_gpu_blocks")
        args.num_cpu_blocks = config.get("num_cpu_blocks")
        args.forced_num_gpu_blocks = config.get("forced_num_gpu_blocks")
        args.num_gpu_blocks_override = config.get("num_gpu_blocks_override")
        return args

    def _load_auto_config(self, model_id: str, trust_remote_code: bool):
        try:  # pragma: no cover - heavy dependency
            from transformers import AutoConfig

            return AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception as exc:
            logger.warning("legacy_vllm_backend: using dummy AutoConfig for {} ({})", model_id, exc)
            text_cfg = SimpleNamespace(
                max_position_embeddings=4096,
                model_type="llama",
                rope_scaling={},
            )
            return SimpleNamespace(
                architectures=["RewardModel"],
                model_type="llama",
                rope_scaling={},
                max_position_embeddings=4096,
                seq_length=0,
                text_config=text_cfg,
                moe_intermediate_size=None,
            )

    def _load_auto_processor(self, model_id: str, trust_remote_code: bool):
        try:  # pragma: no cover - heavy dependency
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception as exc:
            logger.warning("legacy_vllm_backend: using dummy processor for {} ({})", model_id, exc)
            return SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))

    def _apply_model_patches(self, cfg: Any, args: SimpleNamespace):
        """仿照 llm-eval：Reward/MoE/rope scaling/低显存缓存等补丁。"""

        # RewardModel 架构修正
        archs = getattr(cfg, "architectures", []) or []
        model_type = getattr(cfg, "model_type", "") or getattr(getattr(cfg, "text_config", cfg), "model_type", "")
        if "RewardModel" in archs:
            if model_type == "llama" and "LlamaForRewardModel" not in archs:
                archs.append("LlamaForRewardModel")
            elif model_type == "qwen2" and "Qwen2ForRewardModel" not in archs:
                archs.append("Qwen2ForRewardModel")
            cfg.architectures = archs

        # rope scaling 补丁
        rope_scaling = getattr(cfg, "rope_scaling", {}) or {}
        factor = 1.0 if rope_scaling.get("rope_type") == "llama3" else rope_scaling.get("factor", 1.0)
        max_model_length = int(
            (getattr(cfg, "max_position_embeddings", 0) or 0) * factor
            or getattr(cfg, "seq_length", 0)
            or getattr(getattr(cfg, "text_config", cfg), "max_position_embeddings", 0) * factor
        )
        text_config = getattr(cfg, "text_config", cfg)
        if args.max_length and args.max_length != max_model_length:
            target = int(args.max_length / factor) if factor else int(args.max_length)
            try:
                setattr(text_config, "max_position_embeddings", target)
            except Exception:
                pass
        cfg.text_config = text_config

        # MoE 专家并行
        if getattr(cfg, "moe_intermediate_size", None) is not None:
            if getattr(args, "tensor_parallel_size", 1) > 1:
                setattr(args, "enable_expert_parallel", True)

        # 低显存 cache block 估算
        if getattr(args, "low_vram", False):
            max_length = args.max_length or max_model_length or 8192
            block_size = args.block_size or 32
            num_cache_blocks = int((max_length or 8192) * 1.1 / block_size)
            for key in ("num_gpu_blocks", "num_cpu_blocks", "forced_num_gpu_blocks", "num_gpu_blocks_override"):
                if getattr(args, key, None) is None:
                    setattr(args, key, num_cache_blocks)

        return cfg

    def _build_engine(self, cfg: Any, processor: Any, model_id: str, args: SimpleNamespace):
        try:  # pragma: no cover - heavy dependency
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            limit_mm = self._resolve_mm_limits(cfg, args) if self._is_multimodal_config(cfg, processor) else None
            engine_kwargs = dict(
                model=model_id,
                tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
                trust_remote_code=getattr(args, "trust_remote_code", True),
                enforce_eager=getattr(args, "enforce_eager", False),
            )
            if limit_mm is not None:
                engine_kwargs["limit_mm_per_prompt"] = limit_mm

            engine_args = AsyncEngineArgs(**engine_kwargs)
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            engine.log_requests = False
            try:
                (getattr(engine, "engine_core", None) or engine.engine).log_stats = False
            except Exception:
                pass
            return engine, processor
        except Exception as exc:
            logger.warning("legacy_vllm_backend: vLLM not available, using DummyEngine ({})", exc)

            class DummyEngine:
                def __init__(self, cfg_obj, args_obj):
                    self.cfg = cfg_obj
                    self.args = args_obj
                    self.calls: List[Dict[str, Any]] = []

                def generate(self, **kwargs):
                    self.calls.append(kwargs)
                    rid = kwargs.get("request_id", "")
                    return {"outputs": [{"text": f"dummy-{rid}"}]}

                def abort(self, *_args, **_kwargs):
                    return None

            dummy = DummyEngine(cfg, args)
            dummy.processor = processor
            return dummy, processor

    def _is_multimodal_config(self, cfg: Any, processor: Any) -> bool:
        """Best-effort detection to decide whether to pass multi-modal limits to vLLM."""

        model_type = str(getattr(cfg, "model_type", "")).lower()
        if any(tag in model_type for tag in ("vision", "vl", "multimodal")):
            return True

        # HuggingFace configs for VL models usually carry one of these fields
        cfg_dict = cfg.__dict__ if hasattr(cfg, "__dict__") else {}
        vision_keys = (
            "vision_config",
            "multi_modal_config",
            "mm_vision_tower",
            "vision_tower",
            "mm_projector",
        )
        if any(getattr(cfg, key, None) is not None or key in cfg_dict for key in vision_keys):
            return True

        # Processor hints (AutoProcessor for VL models exposes image/vision attributes)
        if processor is not None:
            processor_keys = ("image_processor", "vision_model", "image_token_id")
            if any(hasattr(processor, key) for key in processor_keys):
                return True

        return False

    def _resolve_mm_limits(self, cfg: Any, args: SimpleNamespace) -> Dict[str, int]:
        """Resolve multi-modal limits; default to a very high image cap to avoid ValueError."""

        # Prefer explicit limits from args/config
        limit_cfg = getattr(args, "limit_mm_per_prompt", None) or getattr(cfg, "limit_mm_per_prompt", None)
        if isinstance(limit_cfg, dict) and limit_cfg:
            return limit_cfg

        env_image = os.environ.get("GAGE_EVAL_VLLM_IMAGE_LIMIT")
        env_audio = os.environ.get("GAGE_EVAL_VLLM_AUDIO_LIMIT")

        limits: Dict[str, int] = {}
        if env_image:
            try:
                limits["image"] = int(env_image)
            except ValueError:
                pass
        if env_audio:
            try:
                limits["audio"] = int(env_audio)
            except ValueError:
                pass

        # Default: effectively unlimited images; avoids vLLM default of 1 image raising ValueError
        if not limits:
            limits["image"] = 1_000_000
        return limits

    def _init_tokenizer(self, config: Dict[str, Any]):
        tok_name = config.get("tokenizer_name") or config.get("tokenizer_path") or config.get("model_path")
        if not tok_name:
            return None
        try:  # pragma: no cover - optional dependency
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                tok_name, trust_remote_code=bool(config.get("trust_remote_code", True))
            )
        except Exception as exc:
            logger.warning("legacy_vllm_backend failed to load tokenizer '{}': {}", tok_name, exc)
            return None

    # ---------------------- #
    # CosyVoice2 专用加载分支 #
    # ---------------------- #
    def _load_cosyvoice2(self, config: Dict[str, Any]):
        model_dir = config.get("model_path") or config.get("model")
        if not model_dir:
            raise ValueError("cosyvoice2 backend requires model_path")

        try:  # pragma: no cover - heavy dependency
            # 依赖 legacy 路径：utils.cosyvoice.model.CosyVoice2Model，项目结构变更需同步此处
            from utils.cosyvoice.model import CosyVoice2Model
            from hyperpyyaml import load_hyperpyyaml
        except Exception as exc:
            raise RuntimeError("cosyvoice2 dependencies are missing") from exc

        yaml_path = Path(model_dir) / "cosyvoice2.yaml"
        with open(yaml_path, "r") as f:
            configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": model_dir})

        model = CosyVoice2Model(model_dir, configs["flow"], configs["hift"], bool(config.get("fp16", False)))
        model.load(str(Path(model_dir) / "flow.pt"), str(Path(model_dir) / "hift.pt"))

        if config.get("load_jit", True):
            try:
                model.load_jit(
                    str(Path(model_dir) / f"flow.encoder.{ 'fp16' if config.get('fp16') else 'fp32'}.zip")
                )
            except Exception:
                logger.warning("cosyvoice2 load_jit failed; continue", exc_info=True)

        if config.get("load_trt"):
            try:
                model.load_trt(
                    str(Path(model_dir) / f"flow.decoder.estimator.{ 'fp16' if config.get('fp16') else 'fp32'}.mygpu.plan"),
                    str(Path(model_dir) / "flow.decoder.estimator.fp32.onnx"),
                    bool(config.get("fp16", False)),
                )
            except Exception:
                logger.warning("cosyvoice2 load_trt failed; continue", exc_info=True)

        tokenizer = configs["get_tokenizer"]()
        return model, tokenizer

    def _convert_output_cosyvoice(self, result: Any) -> Dict[str, Any]:
        """标准化 CosyVoice2 输出，便于下游处理."""

        if isinstance(result, dict):
            outputs = result.get("outputs") or result.get("data") or []
        elif isinstance(result, list):
            outputs = result
        else:
            outputs = [result]

        normalized = []
        for item in outputs:
            if isinstance(item, dict):
                normalized.append(item)
            elif hasattr(item, "__dict__") and item.__dict__:
                normalized.append(dict(item.__dict__))
            else:
                normalized.append({"raw": item if isinstance(item, (str, int, float, bool)) else repr(item)})

        return {"raw_outputs": normalized, "type": "cosyvoice2"}

    def _prepare_multi_modal_data(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sample = payload.get("sample") or {}
        raw_inputs = payload.get("inputs") or sample.get("inputs") or {}
        mm = raw_inputs.get("multi_modal_data") if isinstance(raw_inputs, dict) else None
        if not mm:
            mm = sample.get("multi_modal_data")

        image_sources: List[Any] = []
        audio_sources: List[Any] = []
        if isinstance(mm, dict):
            mm_images = mm.get("image") or mm.get("images")
            if mm_images is not None:
                image_sources.extend(mm_images if isinstance(mm_images, list) else [mm_images])
            audio_raw = mm.get("audio") or mm.get("audios")
            if audio_raw:
                audio_sources.extend(audio_raw if isinstance(audio_raw, list) else [audio_raw])

        messages = payload.get("messages") or sample.get("messages") or []
        image_sources.extend(self._extract_images_from_messages(messages))

        image_sources = self._dedup_media_sources(image_sources)
        images = self._load_images(image_sources)

        images = [img for img in images if img is not None]
        audios = [au for au in self._dedup_media_sources(audio_sources) if au is not None]
        if not images and not audios:
            return None
        result: Dict[str, Any] = {}
        if images:
            result["image"] = images
        if audios:
            result["audio"] = audios
        return result

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

    @staticmethod
    def _dedup_media_sources(sources: List[Any]) -> List[Any]:
        """简单去重媒体来源，避免 messages 与 inputs 重复计数."""

        seen = set()
        deduped: List[Any] = []
        for src in sources:
            key = None
            if isinstance(src, str):
                key = src
            else:
                try:
                    key = hash(src)
                except Exception:
                    key = None
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            deduped.append(src)
        return deduped

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

    def _load_multimodal_payload(self, mm: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not mm:
            return None
        processor = getattr(self, "_processor", None)
        if processor is None:
            logger.warning("legacy_vllm_backend: processor missing, skipping multi_modal_data load")
            return mm

        # 使用线程池执行同步的 load_multimodal_data，避免 asyncio 事件循环嵌套死锁
        from concurrent.futures import ThreadPoolExecutor
        
        def _safe_load():
            try:
                return load_multimodal_data(processor, mm.get("image"), mm.get("audio"), True)
            except Exception as exc:
                logger.error(
                    "legacy_vllm_backend: load_multimodal_data failed in thread; fallback to raw mm (error={})",
                    exc,
                )
                return mm

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_safe_load)
            try:
                return future.result()
            except Exception as exc:
                logger.error("legacy_vllm_backend: thread pool execution failed: {}", exc)
                return mm
