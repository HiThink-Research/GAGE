"""vLLM backend with legacy compatibility (AsyncLLMEngine, text/vision/audio)."""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.backends.shared_utils import (
    build_engine_args,
    build_sampling_params,
    build_sampling_params_base,
    check_tokenizer_conflict,
    collect_multimodal_sources,
    convert_text_like_output,
    detect_vllm_version,
    ensure_spawn_start_method,
    finalize_backend_result,
    graceful_loop_shutdown,
    has_multimodal_inputs,
    load_images,
    load_multimodal_payload,
    maybe_tokenize_messages,
    normalize_image_placeholders,
    normalize_messages_safe,
    render_prompt_with_template,
    render_with_processor,
    resolve_request_id,
    resolve_sampling_class,
    resolve_vllm_mm_support,
    run_coroutine_threadsafe_with_timeout,
    simple_render_messages,
)
from gage_eval.role.model.backends.vllm_request import normalize_request_payload
from gage_eval.role.model.runtime import BackendCapabilities, ChatTemplateMixin, ChatTemplatePolicy
from gage_eval.utils.chat_templates import get_fallback_template
from gage_eval.utils.cleanup import install_signal_cleanup

try:  # pragma: no cover - optional dependency
    import vllm  # type: ignore
    from packaging import version as _pkg_version

    _VLLM_VERSION = getattr(vllm, "__version__", None)
    _VLLM_PROMPT_V1 = bool(_VLLM_VERSION) and _pkg_version.parse(_VLLM_VERSION) >= _pkg_version.parse("0.8.0")
except Exception:  # pragma: no cover
    _VLLM_VERSION = None
    _VLLM_PROMPT_V1 = False


@registry.asset(
    "backends",
    "vllm",
    desc="vLLM 本地推理后端（AsyncLLMEngine，文本/多模态）",
    tags=("llm", "local", "serving"),
    modalities=("text", "vision", "audio"),
)
class VLLMBackend(EngineBackend, ChatTemplateMixin):
    """Backend that proxies requests to vLLM AsyncLLMEngine with unified request shaping."""

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
        self._request_timeout = float(config.get("request_timeout", 300))
        cfg = dict(config)
        cfg.setdefault("execution_mode", "native")
        ensure_spawn_start_method()
        self._vllm_version = detect_vllm_version()
        if self._vllm_version:
            logger.info("Detected vLLM version {}", self._vllm_version)
        self._mm_supported, self._mm_strategy, self._strict_mm = resolve_vllm_mm_support(config, self._vllm_version)

        # Initialize a dedicated event loop thread to avoid deadlocks when vLLM spins loops internally.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, name="VLLMBackendLoop", daemon=True)
        self._loop_thread.start()

        super().__init__(cfg)
        install_signal_cleanup(self.shutdown)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def shutdown(self) -> None:
        graceful_loop_shutdown(self._loop, self._loop_thread, getattr(self, "model", None))

    def load_model(self, config: Dict[str, Any]):
        """加载模型 + processor，并应用兼容性补丁（奖励/MoE/rope scaling/低显存等）。"""

        trust_remote_code = bool(config.get("trust_remote_code", True))
        model_id = config.get("model_path") or config.get("model")
        if not model_id:
            raise ValueError("vllm_backend requires 'model_path' or 'model' in config")

        output_type = config.get("output_type", "text")
        args_ns = build_engine_args(config, output_type=output_type, trust_remote_code=trust_remote_code)
        cfg_obj = self._load_auto_config(model_id, trust_remote_code=trust_remote_code)
        self._processor = self._load_auto_processor(model_id, trust_remote_code=trust_remote_code)
        self._tokenizer = self._init_tokenizer(config)
        cfg_obj = self._apply_model_patches(cfg_obj, args_ns)
        engine, processor = self._build_engine(cfg_obj, self._processor, model_id, args_ns)
        self._processor = processor
        return engine

    def prepare_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize inbound payloads and attach chat/multimodal metadata."""

        ctx = normalize_request_payload(payload, request_prefix="vllm")
        prepared: Dict[str, Any] = {
            "sample": ctx.sample,
            "messages": ctx.messages or [],
            "inputs": ctx.inputs or {},
            "prompt": ctx.prompt or (ctx.inputs.get("prompt") if isinstance(ctx.inputs, dict) else "") or "",
            "prompt_meta": ctx.prompt_meta or {},
            "cache_namespace": ctx.cache_namespace,
            "chat_template_kwargs": ctx.chat_template_kwargs or {},
            "output_type": ctx.output_type or "text",
            "sampling_params": ctx.sampling_params or {},
            "sample_n": max(int(ctx.sample_n or 1), 1),
            "request_id": ctx.request_id,
        }
        prepared.update(ctx.chat_meta)

        check_tokenizer_conflict(prepared, self._cfg_tokenizer_path)
        prompt = self._render_prompt(prepared)
        prompt, inputs_val = self._maybe_tokenize_messages(prepared, prompt)
        prepared["prompt"] = prompt
        prepared["inputs"] = inputs_val

        sampling_base = build_sampling_params_base(
            self._default_sampling, prepared.get("sampling_params") or {}, max_tokens=self._max_tokens
        )
        sampling = build_sampling_params(
            ctx.output_type,
            sampling_base,
            default_sampling=self._default_sampling,
            max_tokens=self._max_tokens,
            sampling_class_resolver=resolve_sampling_class,
        )
        prepared["sampling_params"] = sampling
        if not prepared.get("request_id"):
            prepared["request_id"] = resolve_request_id(prepared, prefix="vllm")

        caps = BackendCapabilities(supports_mm=self._mm_supported, has_processor_chat_template=bool(self._processor))
        prepared["cache_suffix"] = ChatTemplateMixin.get_cache_suffix("text", self._chat_template_policy, caps)

        return prepared

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """同步包装，支持 sample_n 多路采样与 output_type 路由."""

        logger.info(
            "vllm_backend handling request_id={} output_type={} sample_n={}",
            inputs.get("request_id"),
            inputs.get("output_type"),
            inputs.get("sample_n"),
        )
        return self._generate_prepared(inputs, batch_path="native_single")

    def generate_batch(self, inputs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared_list = [self.prepare_inputs(item) for item in inputs_list]
        if any((item.get("sample_n") or 1) > 1 for item in prepared_list):
            logger.debug("vllm_backend: sample_n>1 in batch; falling back to per-request execution")
            return [self._generate_prepared(item, batch_path="native_single") for item in prepared_list]

        results: List[Dict[str, Any]] = []
        for item in prepared_list:
            results.append(self._generate_prepared(item, batch_path="native_batch"))
        return results

    def _generate_prepared(self, prepared: Dict[str, Any], *, batch_path: str) -> Dict[str, Any]:
        output_type = prepared.get("output_type") or "text"
        sample_n = max(int(prepared.get("sample_n") or 1), 1)
        sampling_params = prepared["sampling_params"]
        base_request_id = prepared.get("request_id") or resolve_request_id(prepared, prefix="vllm")
        outputs = []
        for idx in range(sample_n):
            rid = base_request_id if sample_n == 1 else f"{base_request_id}_{idx}"
            result = self._generate_one(prepared, sampling_params, rid)
            outputs.append(self._convert_output(result, output_type))
        return finalize_backend_result(
            prepared,
            outputs,
            sample_n=sample_n,
            batch_path=batch_path,
            backend_tag="vllm_backend",
            cfg_tokenizer_path=self._cfg_tokenizer_path,
        )

    def _generate_one(self, prepared: Dict[str, Any], sampling_params: Any, request_id: str) -> Any:
        prompt = prepared["prompt"]
        messages = prepared["messages"]
        mm_requested = has_multimodal_inputs(prepared)
        if mm_requested and not self._mm_supported:
            if self._strict_mm:
                raise RuntimeError("multi_modal_data disabled for this vLLM version")
            logger.warning("vllm_backend multi_modal_data disabled; falling back to prompt-only")
        # 注意：这里 _load_multimodal_payload 内部已经使用了线程池，是线程安全的同步调用
        mm_raw = self._prepare_multi_modal_data(prepared) if mm_requested and self._mm_supported else None
        mm_loaded = self._load_multimodal_payload(mm_raw) if mm_raw else None

        # 定义异步任务，将在后台 loop 中运行
        async def _async_generate():
            # 构造 prompt，优先 processor 渲染
            rendered_mm = self._render_with_processor(messages, prompt, prepared.get("chat_template_kwargs"))
            final_prompt_text = rendered_mm if rendered_mm else prompt

            # vLLM v1（>=0.8.0）接口支持单个 PromptInputs dict 作为位置参数；旧接口只接受 keyword prompt/inputs
            use_v1_prompt = bool(_VLLM_PROMPT_V1)
            prompt_input: Any
            mm_payload = None
            if mm_loaded and self._mm_supported:
                if self._mm_strategy == "inputs":
                    mm_payload = mm_loaded
                else:
                    if self._strict_mm:
                        raise RuntimeError("multi_modal_data disabled for this vLLM version")
                    logger.warning("vllm_backend multi_modal_data disabled; falling back to prompt-only")
            if mm_payload:
                # 确保 prompt 中包含足够的 <image> 占位，防止 vLLM 提示更新不匹配
                image_field = mm_payload.get("image") if isinstance(mm_payload, dict) else None
                if isinstance(image_field, (list, tuple)):
                    image_count = len(image_field)
                elif image_field is None:
                    image_count = 0
                else:
                    image_count = 1
                final_prompt_text = normalize_image_placeholders(final_prompt_text, image_count)
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
                    if mm_loaded:
                        self._mm_supported = False
                        self._mm_strategy = "disabled"
                        if self._strict_mm:
                            raise RuntimeError(
                                f"multi_modal_data unsupported in this vLLM version; strict mode enabled ({exc})"
                            ) from exc
                    logger.warning(
                        "vllm_backend generate encountered TypeError ({}); "
                        "falling back to prompt-only without multi_modal_data",
                        exc,
                    )
                    prompt_for_retry = prompt_input.get("prompt") if isinstance(prompt_input, dict) else prompt_input
                    result = self.model.generate(prompt=prompt_for_retry or prompt, sampling_params=sampling_params, request_id=request_id)
                else:
                    raise
            except Exception as exc:
                logger.warning(
                    "vllm_backend generate failed: {} - {}; returning echo output", type(exc).__name__, exc, exc_info=True
                )
                return {"outputs": [{"text": str(prompt)}]}

            # 异步收集结果
            if inspect.isasyncgen(result) or isinstance(result, types.AsyncGeneratorType):
                final_item = None
                async for item in result:
                    final_item = item
                return final_item
            if inspect.iscoroutine(result):
                return await result
            return result

        # 提交到后台线程执行并等待结果
        try:
            abort_fn = getattr(self.model, "abort", None)
            return run_coroutine_threadsafe_with_timeout(
                self._loop,
                _async_generate(),
                timeout=self._request_timeout,
                request_id=request_id,
                abort_fn=abort_fn,
                logger_prefix="vllm_backend",
                timeout_result_fn=lambda: {"outputs": [{"text": str(prompt)}]},
            )
        except Exception as exc:
            logger.error("vllm_backend: background thread execution failed: {}", exc)
            return {"outputs": [{"text": str(prompt)}]}

    # ------------------------------------------------------------------ #
    # 模型加载与补丁                                                         #
    # ------------------------------------------------------------------ #
    def _load_auto_config(self, model_id: str, trust_remote_code: bool):
        try:  # pragma: no cover - heavy dependency
            from transformers import AutoConfig

            return AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception as exc:
            logger.warning("vllm_backend: using dummy AutoConfig for {} ({})", model_id, exc)
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
            logger.warning("vllm_backend: using dummy processor for {} ({})", model_id, exc)
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
            from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
            from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore

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
            logger.warning("vllm_backend: vLLM not available, using DummyEngine ({})", exc)

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
            logger.warning("vllm_backend failed to load tokenizer '{}': {}", tok_name, exc)
            return None

    # ------------------------------------------------------------------ #
    # 适配/渲染/采样辅助                                                      #
    # ------------------------------------------------------------------ #
    def _convert_output(self, result: Any, output_type: str) -> Any:
        if output_type in {"text", "beam"}:
            return convert_text_like_output(result)
        if output_type in {"reward", "loss", "prompt_tokens", "next_token_prob"}:
            return result
        return result

    def _render_prompt(self, prepared: Dict[str, Any]) -> str:
        messages = prepared["messages"]
        raw_prompt = prepared["prompt"]
        policy = ChatTemplatePolicy(mode=self._chat_template_mode)
        caps = BackendCapabilities(supports_mm=self._mm_supported, has_processor_chat_template=bool(self._processor))
        chat_kwargs = prepared["chat_template_kwargs"]
        return render_prompt_with_template(
            prepared,
            tokenizer=self._tokenizer,
            fallback_template=self._fallback_template,
            policy=policy,
            caps=caps,
            chat_kwargs=chat_kwargs,
            simple_renderer=simple_render_messages,
        )

    def _maybe_tokenize_messages(self, prepared: Dict[str, Any], prompt: str) -> Tuple[str, Any]:
        """使用 backend tokenizer 生成 prompt_token_ids（若预处理器未提供）。"""

        force_tokenize = bool(prepared.get("force_tokenize_prompt") or self._force_tokenize_prompt)
        new_prompt, new_inputs, meta = maybe_tokenize_messages(
            prepared,
            prompt,
            tokenizer=self._tokenizer,
            processor=self._processor,
            policy=self._chat_template_policy,
            force_tokenize=force_tokenize,
        )
        for k, v in (meta or {}).items():
            prepared.setdefault(k, v)
        return new_prompt, new_inputs

    def _fallback_render(self, messages: List[Dict[str, Any]], tpl: Optional[str]) -> str:
        return simple_render_messages(messages)

    def _render_with_processor(
        self, messages: List[Dict[str, Any]], prompt: str, chat_template_kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        rendered = render_with_processor(self._processor, normalize_messages_safe(messages), prompt, chat_template_kwargs)
        if rendered is not None:
            return rendered
        tok = getattr(self, "_tokenizer", None)
        tok_apply = getattr(tok, "apply_chat_template", None)
        if tok_apply:
            try:
                rendered_tok = tok_apply(
                    normalize_messages_safe(messages),
                    add_generation_prompt=True,
                    tokenize=False,
                    **(chat_template_kwargs or {}),
                )
                if isinstance(rendered_tok, list):
                    rendered_tok = rendered_tok[0]
                if rendered_tok:
                    return str(rendered_tok)
            except Exception as tok_exc:
                logger.debug("vllm_backend tokenizer chat_template failed: {}", tok_exc)
        return None

    def _prepare_multi_modal_data(self, prepared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sources = collect_multimodal_sources(prepared)
        images = self._load_images(sources.get("image"))
        audios = sources.get("audio") or []
        images = [img for img in images if img is not None]
        audios = [au for au in audios if au is not None]
        if not images and not audios:
            return None
        result: Dict[str, Any] = {}
        if images:
            result["image"] = images
        if audios:
            result["audio"] = audios
        return result

    def _load_images(self, sources) -> List[Any]:
        return load_images(sources)

    def _load_multimodal_payload(self, mm: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return load_multimodal_payload(mm, getattr(self, "_processor", None), logger_prefix="vllm_backend")
