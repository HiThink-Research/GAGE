"""vLLM backend with legacy compatibility (AsyncLLMEngine, text/vision/audio)."""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import threading
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from gage_eval.compat.vllm_renderer_patch import (
    detect_vllm_engine_multimodal_support as _detect_vllm_engine_multimodal_support,
    install_vllm_renderer_compat_patches as _install_vllm_renderer_compat_patches,
    prime_vllm_engine_renderer_state as _prime_vllm_engine_renderer_state,
)
from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.common.backend_utils import (
    build_sampling_params_base,
    check_tokenizer_conflict,
    collect_multimodal_sources,
    convert_text_like_output,
    finalize_backend_result,
    graceful_loop_shutdown,
    has_multimodal_inputs,
    load_hf_processor,
    load_hf_tokenizer,
    load_images,
    load_multimodal_payload,
    log_debug_mm_prompt,
    maybe_tokenize_messages,
    normalize_image_placeholders,
    normalize_messages_safe,
    normalize_prompt_token_ids,
    propagate_metadata_flags,
    render_prompt_with_template,
    render_with_processor,
    resolve_request_id,
    run_coroutine_threadsafe_with_timeout,
    simple_render_messages,
)
from gage_eval.role.model.backends.vllm.vllm_request import (
    build_engine_args,
    build_sampling_params,
    detect_vllm_version,
    ensure_spawn_start_method,
    normalize_request_payload,
    resolve_sampling_class,
    resolve_vllm_mm_support,
)
from gage_eval.role.model.backends.vllm.runtime_compat import (
    load_vllm_engine_runtime,
    prepare_async_engine_kwargs,
)
from gage_eval.role.model.runtime import BackendCapabilities, ChatTemplateMixin, ChatTemplatePolicy
from gage_eval.utils.chat_templates import get_fallback_template
from gage_eval.utils.cleanup import install_signal_cleanup


@registry.asset(
    "backends",
    "vllm",
    desc="vLLM local inference backend (AsyncLLMEngine; text/multimodal)",
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
        self._model_supports_mm = True
        self._engine_mm_support: Optional[bool] = None
        self._engine_runtime = None
        self._shutdown_lock = threading.Lock()
        self._loop_start_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_completed = False
        self._cleanup_unregister = lambda: None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        if config.get("gpu_groups") is not None or config.get("auto_gpu_groups") is not None:
            raise ValueError(
                "vllm_backend no longer supports gpu_groups/auto_gpu_groups router mode. "
                "Use vLLM native data_parallel_* options instead."
            )
        cfg = dict(config)
        cfg.setdefault("execution_mode", "native")
        ensure_spawn_start_method()
        self._vllm_version = detect_vllm_version()
        if self._vllm_version:
            logger.info("Detected vLLM version {}", self._vllm_version)
        logger.info(
            "vllm_backend mode=single_engine tensor_parallel_size={} pipeline_parallel_size={} data_parallel_size={}",
            config.get("tensor_parallel_size"),
            config.get("pipeline_parallel_size"),
            config.get("data_parallel_size"),
        )
        self._mm_supported, self._mm_strategy, self._strict_mm = resolve_vllm_mm_support(config, self._vllm_version)

        # Initialize a dedicated event loop thread to avoid deadlocks when vLLM spins loops internally.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, name="VLLMBackendLoop", daemon=True)
        self._loop_thread.start()

        super().__init__(cfg)
        install_signal_cleanup(self.shutdown)
        import atexit
        atexit.register(self.shutdown)

    def _run_loop(self) -> None:
        if self._loop is None:  # pragma: no cover - defensive guard
            return
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_background_loop(self) -> asyncio.AbstractEventLoop:
        """Start the background loop lazily so idle backends avoid extra threads."""

        loop = self._loop
        loop_thread = self._loop_thread
        if loop is not None and loop_thread is not None and loop_thread.is_alive():
            return loop
        with self._loop_start_lock:
            loop = self._loop
            loop_thread = self._loop_thread
            if loop is not None and loop_thread is not None and loop_thread.is_alive():
                return loop
            with self._shutdown_lock:
                if self._shutdown_started or self._shutdown_completed:
                    raise RuntimeError("vllm_backend background loop is unavailable after shutdown")
            loop = asyncio.new_event_loop()
            loop_thread = threading.Thread(
                target=self._run_loop,
                name="VLLMBackendLoop",
                daemon=True,
            )
            self._loop = loop
            self._loop_thread = loop_thread
            loop_thread.start()
            self._cleanup_unregister = install_signal_cleanup(self.shutdown)
            return loop

    def shutdown(self) -> None:
<<<<<<< HEAD
        with self._shutdown_lock:
            if self._shutdown_started or self._shutdown_completed:
                return
            self._shutdown_started = True
            cleanup_unregister = self._cleanup_unregister
            self._cleanup_unregister = lambda: None
        try:
            graceful_loop_shutdown(self._loop, self._loop_thread, getattr(self, "model", None))
        finally:
            cleanup_unregister()
            with self._shutdown_lock:
                self._shutdown_completed = True
=======
        # vLLM v1 engine shutdown: suppress the spurious EngineDeadError by
        # (1) cancelling the output_handler, (2) marking engine_dead before
        # engine_core.shutdown() so the monitor thread doesn't log an error.
        try:
            model = getattr(self, "model", None)
            if model is not None:
                handler = getattr(model, "output_handler", None)
                if handler is not None:
                    try:
                        from vllm.utils.async_utils import cancel_task_threadsafe
                        cancel_task_threadsafe(handler)
                        # Poll briefly until the task actually finishes to avoid
                        # a race where the handler logs an exception after core
                        # shutdown.
                        import time
                        for _ in range(30):
                            if handler.done():
                                break
                            time.sleep(0.05)
                    except Exception:
                        pass

                if engine_core := getattr(model, "engine_core", None):
                    resources = getattr(engine_core, "resources", None)
                    if resources is not None:
                        try:
                            resources.engine_dead = True
                        except Exception:
                            pass
                    try:
                        engine_core.shutdown()
                    except Exception:
                        pass

                if renderer := getattr(model, "renderer", None):
                    try:
                        renderer.shutdown()
                    except Exception:
                        pass

                # Drop the reference so AsyncLLM.__del__ doesn't race during
                # Python process exit and log a TypeError.
                self.model = None
        except Exception:
            pass
        graceful_loop_shutdown(self._loop, self._loop_thread, None)
>>>>>>> benchmark video-mme

    def load_model(self, config: Dict[str, Any]):
        """Load the model/processor and apply compatibility patches (reward/MoE/rope scaling/low-memory)."""

        trust_remote_code = bool(config.get("trust_remote_code", True))
        model_id = config.get("model_path") or config.get("model")
        if not model_id:
            raise ValueError("vllm_backend requires 'model_path' or 'model' in config")

        output_type = config.get("output_type", "text")
        args_ns = build_engine_args(config, output_type=output_type, trust_remote_code=trust_remote_code)
        _install_vllm_renderer_compat_patches()
        cfg_obj = self._load_auto_config(model_id, trust_remote_code=trust_remote_code)
        self._processor = self._load_auto_processor(model_id, trust_remote_code=trust_remote_code)
        self._tokenizer = self._init_tokenizer(config)
        self._model_supports_mm = self._is_multimodal_config(cfg_obj, self._processor)
        cfg_obj = self._apply_model_patches(cfg_obj, args_ns)
        engine, processor = self._build_engine(cfg_obj, self._processor, model_id, args_ns, config)
        self._processor = processor
        self._refresh_engine_mm_support(require_mm_processor=False, engine=engine)
        return engine

    def prepare_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize inbound payloads and attach chat/multimodal metadata."""

        ctx = normalize_request_payload(payload, request_prefix="vllm")
        chat_template_kwargs = dict(getattr(self, "_resolved_thinking_kwargs", {}) or {})
        chat_template_kwargs.update(ctx.chat_template_kwargs or {})
        prepared: Dict[str, Any] = {
            "sample": ctx.sample,
            "messages": ctx.messages or [],
            "inputs": ctx.inputs or {},
            "prompt": ctx.prompt or (ctx.inputs.get("prompt") if isinstance(ctx.inputs, dict) else "") or "",
            "prompt_meta": ctx.prompt_meta or {},
            "cache_namespace": ctx.cache_namespace,
            "chat_template_kwargs": chat_template_kwargs,
            "output_type": ctx.output_type or "text",
            "sampling_params": ctx.sampling_params or {},
            "sample_n": max(int(ctx.sample_n or 1), 1),
            "request_id": ctx.request_id,
        }
        prepared.update(ctx.chat_meta)

        # Ensure metadata-based flags are propagated
        propagate_metadata_flags(prepared, payload)

        check_tokenizer_conflict(prepared, self._cfg_tokenizer_path)
        # NOTE: For multimodal requests, skip tokenizer-based rendering in prepare_inputs.
        # The processor in _async_generate will handle proper vision token insertion.
        mm_detected = has_multimodal_inputs(prepared)
        mm_render_enabled = mm_detected and self._supports_multimodal_requests()
        if mm_render_enabled:
            # Keep raw prompt from messages; processor will format it later
            prompt = prepared.get("prompt") or ""
            inputs_val = prepared.get("inputs") or {}
        else:
            prompt = self._render_prompt(prepared)
            prompt, inputs_val = self._maybe_tokenize_messages(prepared, prompt)
        prepared["prompt"] = prompt
        prepared["inputs"] = inputs_val

        sampling_base = build_sampling_params_base(
            self._default_sampling, prepared.get("sampling_params") or {}, max_tokens=self._max_tokens
        )
        sample_n_override = _resolve_sample_n_override(sampling_base)
        if sample_n_override is not None and int(prepared.get("sample_n") or 1) <= 1:
            prepared["sample_n"] = sample_n_override
        sampling_base = _strip_sample_n_params(sampling_base)
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

        caps = BackendCapabilities(
            supports_mm=self._supports_multimodal_requests(),
            has_processor_chat_template=bool(self._processor),
        )
        prepared["cache_suffix"] = ChatTemplateMixin.get_cache_suffix("text", self._chat_template_policy, caps)

        return prepared

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single request synchronously with `sample_n` and `output_type` routing."""

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

        # STEP 1: Resolve whether the live engine can stay on the multimodal path.
        mm_requested = has_multimodal_inputs(prepared)
        if mm_requested:
            self._refresh_engine_mm_support(require_mm_processor=False)
        mm_enabled = bool(mm_requested and self._supports_multimodal_requests())
        if mm_requested and not mm_enabled:
            if self._strict_mm:
                raise RuntimeError("configured vLLM model does not support multi_modal_data")
            logger.warning(
                "vllm_backend received multimodal inputs for a text-only model; "
                "falling back to prompt-only request_id={}",
                request_id,
            )
            prompt = self._render_text_fallback(messages, self._strip_mm_tokens(prompt))
        elif mm_enabled:
            self._refresh_engine_mm_support(require_mm_processor=True)
            mm_enabled = bool(mm_requested and self._supports_multimodal_requests())
            if not mm_enabled:
                if self._strict_mm:
                    raise RuntimeError("configured vLLM model does not support multi_modal_data")
                logger.warning(
                    "vllm_backend multimodal renderer probe failed; "
                    "falling back to prompt-only request_id={}",
                    request_id,
                )
                prompt = self._render_text_fallback(messages, self._strip_mm_tokens(prompt))
        # NOTE: `_load_multimodal_payload` already uses a thread pool internally; this is a thread-safe sync call.
        mm_raw = self._prepare_multi_modal_data(prepared) if mm_enabled and self._mm_supported else None
        mm_loaded = self._load_multimodal_payload(mm_raw) if mm_raw else None
        mm_payload = mm_loaded if mm_loaded and self._mm_supported else None
        mm_available = self._has_mm_payload(mm_payload)
        prompt_has_mm_tokens = self._prompt_has_mm_tokens(prompt)

        # Define an async task to be executed on the background event loop thread.
        async def _async_generate():
            # STEP 2: Build the final prompt payload that will be handed to vLLM.
            # Render the prompt with processor for multimodal requests.
            # For multimodal, prepare_inputs skipped tokenizer rendering to avoid double vision tokens.
            # So we MUST render here with processor to get proper Qwen vision tokens.
            if mm_enabled and not mm_available:
                logger.warning(
                    "vllm_backend: multimodal inputs missing payload; falling back to text-only request_id={}",
                    request_id,
                )
                final_prompt_text = self._render_text_fallback(messages, self._strip_mm_tokens(prompt))
                mm_requested_local = False
                mm_payload_local = None
            elif mm_enabled:
                if prompt_has_mm_tokens and prompt:
                    final_prompt_text = prompt
                else:
                    rendered_mm = self._render_with_processor(
                        messages,
                        prompt,
                        prepared.get("chat_template_kwargs"),
                        preserve_multimodal=True,
                    )
                    final_prompt_text = rendered_mm if rendered_mm else prompt
                mm_requested_local = True
                mm_payload_local = mm_payload
            else:
                # Non-multimodal: check render flags
                pre_rendered = prepared.get("chat_template_mode") == "preprocess" or prepared.get("rendered_by") == "preprocess"
                if pre_rendered:
                    final_prompt_text = self._strip_mm_tokens(prompt) if prompt_has_mm_tokens else prompt
                else:
                    rendered_mm = self._render_with_processor(messages, prompt, prepared.get("chat_template_kwargs"))
                    final_prompt_text = rendered_mm if rendered_mm else prompt
                mm_requested_local = False
                mm_payload_local = None
                if self._prompt_has_mm_tokens(final_prompt_text):
                    logger.warning(
                        "vllm_backend: multimodal tokens detected without payload; stripping tokens request_id={}",
                        request_id,
                    )
                    final_prompt_text = self._strip_mm_tokens(final_prompt_text)

            prompt_input: Any
            final_mm_payload = None
            if mm_payload_local and self._mm_supported:
                if self._mm_strategy == "inputs":
                    final_mm_payload = mm_payload_local
                else:
                    if self._strict_mm:
                        raise RuntimeError("multi_modal_data disabled for this vLLM version")
                    logger.warning("vllm_backend multi_modal_data disabled; falling back to prompt-only")
            if final_mm_payload:
                # Ensure the prompt contains enough `<image>` placeholders to match the multimodal payload.
                image_field = final_mm_payload.get("image") if isinstance(final_mm_payload, dict) else None
                if isinstance(image_field, (list, tuple)):
                    image_count = len(image_field)
                elif image_field is None:
                    image_count = 0
                else:
                    image_count = 1
                final_prompt_text = normalize_image_placeholders(final_prompt_text, image_count)
                prompt_input = {"prompt": final_prompt_text, "multi_modal_data": final_mm_payload}
            else:
                prompt_input = final_prompt_text

            log_debug_mm_prompt(final_prompt_text, request_id)

            extra_inputs = prepared.get("inputs") or prepared.get("prompt_token_ids")
            if isinstance(extra_inputs, dict):
                extra_inputs = dict(extra_inputs)
                prompt_token_ids = extra_inputs.get("prompt_token_ids")
                if prompt_token_ids is None and "input_ids" in extra_inputs:
                    prompt_token_ids = extra_inputs.get("input_ids")
                normalized_prompt_token_ids = normalize_prompt_token_ids(prompt_token_ids)
                if normalized_prompt_token_ids is not None:
                    extra_inputs["prompt_token_ids"] = normalized_prompt_token_ids
                extra_inputs.pop("input_ids", None)
            if mm_requested_local and isinstance(extra_inputs, dict):
                extra_inputs = {k: v for k, v in extra_inputs.items() if k != "prompt_token_ids"}
            if isinstance(extra_inputs, dict):
                # Keep prompt-token ids inside the PromptInputs dict so both 0.9.x and 0.17 share one call shape.
                needs_dict = mm_loaded or "prompt_token_ids" in extra_inputs
                if needs_dict and not isinstance(prompt_input, dict):
                    prompt_input = {"prompt": final_prompt_text}
                if isinstance(prompt_input, dict):
                    mapped_extra = dict(extra_inputs)
                    for k, v in mapped_extra.items():
                        prompt_input.setdefault(k, v)

            # STEP 3: Dispatch the request using the prompt shape shared by vLLM 0.9.x and 0.17.
            generate_args = (prompt_input,)
            generate_kwargs = {"sampling_params": sampling_params, "request_id": request_id}
            try:
                result = self.model.generate(*generate_args, **generate_kwargs)
            except TypeError as exc:
                result = self._retry_generate_legacy(
                    prompt_input=prompt_input,
                    prompt=prompt,
                    extra_inputs=extra_inputs,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    mm_loaded=bool(mm_loaded),
                    error=exc,
                )
            except Exception as exc:
                logger.error(
                    "vllm_backend generate failed request_id={} error_type={} error={}",
                    request_id,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                raise RuntimeError(f"vllm_backend generate failed: {type(exc).__name__}: {exc}") from exc

            # Collect results from async generators/coroutines.
            if inspect.isasyncgen(result) or isinstance(result, types.AsyncGeneratorType):
                final_item = None
                async for item in result:
                    final_item = item
                return final_item
            if inspect.iscoroutine(result):
                return await result
            return result

        # Submit to the background event loop thread and wait for completion.
        try:
            abort_fn = getattr(self.model, "abort", None)
            loop = self._ensure_background_loop()
            return run_coroutine_threadsafe_with_timeout(
                loop,
                _async_generate(),
                timeout=self._request_timeout,
                request_id=request_id,
                abort_fn=abort_fn,
                logger_prefix="vllm_backend",
            )
        except Exception as exc:
            logger.error(
                "vllm_backend background execution failed request_id={} error_type={} error={}",
                request_id,
                type(exc).__name__,
                exc,
            )
            raise RuntimeError(f"vllm_backend background execution failed: {type(exc).__name__}: {exc}") from exc

    def _supports_multimodal_requests(self) -> bool:
        """Return whether the loaded engine can handle multimodal payloads."""

        if not self._mm_supported:
            return False
        if self._engine_mm_support is True:
            return True
        if self._engine_mm_support is False:
            return False
        return bool(self._model_supports_mm)

    def _refresh_engine_mm_support(
        self,
        *,
        require_mm_processor: bool,
        engine: Optional[Any] = None,
    ) -> Optional[bool]:
        """Probe a live vLLM engine and cache whether it supports multimodal data."""

        target_engine = engine if engine is not None else getattr(self, "model", None)
        if target_engine is None:
            return self._engine_mm_support

        _prime_vllm_engine_renderer_state(target_engine, require_mm_processor=require_mm_processor)
        detected = _detect_vllm_engine_multimodal_support(target_engine)
        if detected is not None:
            self._engine_mm_support = detected
            self._model_supports_mm = detected
        return self._engine_mm_support

    # ------------------------------------------------------------------ #
    # Model loading and compatibility patches                              #
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
        proc = load_hf_processor(model_id, trust_remote_code)
        if proc:
            return proc
        logger.warning("vllm_backend: using dummy processor for {}", model_id)
        return SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))

    def _apply_model_patches(self, cfg: Any, args: SimpleNamespace):
        """Apply lightweight compatibility patches inspired by llm-eval."""

        # RewardModel architecture normalization.
        archs = getattr(cfg, "architectures", []) or []
        model_type = getattr(cfg, "model_type", "") or getattr(getattr(cfg, "text_config", cfg), "model_type", "")
        if "RewardModel" in archs:
            if model_type == "llama" and "LlamaForRewardModel" not in archs:
                archs.append("LlamaForRewardModel")
            elif model_type == "qwen2" and "Qwen2ForRewardModel" not in archs:
                archs.append("Qwen2ForRewardModel")
            cfg.architectures = archs

        # Rope scaling patch.
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

        # MoE expert parallelism.
        if getattr(cfg, "moe_intermediate_size", None) is not None:
            if getattr(args, "tensor_parallel_size", 1) > 1:
                setattr(args, "enable_expert_parallel", True)

        # Low-memory cache block estimation.
        if getattr(args, "low_vram", False):
            max_length = args.max_length or max_model_length or 8192
            block_size = args.block_size or 32
            num_cache_blocks = int((max_length or 8192) * 1.1 / block_size)
            for key in ("num_gpu_blocks", "num_cpu_blocks", "forced_num_gpu_blocks", "num_gpu_blocks_override"):
                if getattr(args, key, None) is None:
                    setattr(args, key, num_cache_blocks)

        return cfg

    def _build_engine(
        self, cfg: Any, processor: Any, model_id: str, args: SimpleNamespace, config: Dict[str, Any]
    ):
        try:  # pragma: no cover - heavy dependency
            # STEP 1: Inspect the installed vLLM runtime instead of branching on version strings.
            runtime = load_vllm_engine_runtime()
            self._engine_runtime = runtime
            logger.info(
                "vllm_backend resolved AsyncLLMEngine variant={} module={}",
                runtime.engine_variant,
                runtime.async_llm_engine_cls.__module__,
            )
            limit_mm = self._resolve_mm_limits(config, args) if self._is_multimodal_config(cfg, processor) else None

            # STEP 2: Assemble the superset of engine kwargs, then filter to the installed signature.
            engine_kwargs = self._collect_engine_kwargs(model_id, args, config, limit_mm=limit_mm)
            filtered_engine_kwargs, dropped_keys = prepare_async_engine_kwargs(engine_kwargs, runtime)
            if dropped_keys:
                logger.info(
                    "vllm_backend ignoring unsupported AsyncEngineArgs fields for version {}: {}",
                    self._vllm_version or "unknown",
                    ", ".join(dropped_keys),
                )

            # STEP 3: Build the engine with the filtered kwargs and apply runtime patches.
            engine_args = runtime.async_engine_args_cls(**filtered_engine_kwargs)
            engine = runtime.async_llm_engine_cls.from_engine_args(engine_args)
            _prime_vllm_engine_renderer_state(engine)
            engine.log_requests = False
            try:
                (getattr(engine, "engine_core", None) or engine.engine).log_stats = False
            except Exception:
                pass
            return engine, processor
        except Exception as exc:
            logger.error("vllm_backend failed to initialize vLLM engine: {}: {}", type(exc).__name__, exc)
            raise RuntimeError(f"vllm_backend failed to initialize vLLM engine: {type(exc).__name__}: {exc}") from exc

    def _collect_engine_kwargs(
        self,
        model_id: str,
        args: SimpleNamespace,
        config: Dict[str, Any],
        *,
        limit_mm: Optional[Dict[str, int]],
    ) -> Dict[str, Any]:
        """Collect candidate AsyncEngineArgs kwargs before version filtering."""

        engine_kwargs: Dict[str, Any] = {
            "model": model_id,
            "tensor_parallel_size": getattr(args, "tensor_parallel_size", 1),
            "trust_remote_code": getattr(args, "trust_remote_code", True),
        }
        tokenizer_id = getattr(args, "tokenizer", None)
        if tokenizer_id:
            engine_kwargs["tokenizer"] = str(tokenizer_id)

        enforce_eager = getattr(args, "enforce_eager", None)
        if enforce_eager is not None:
            engine_kwargs["enforce_eager"] = bool(enforce_eager)
        if getattr(args, "pipeline_parallel_size", None) is not None:
            engine_kwargs["pipeline_parallel_size"] = int(args.pipeline_parallel_size)
        if getattr(args, "data_parallel_size", None) is not None:
            engine_kwargs["data_parallel_size"] = int(args.data_parallel_size)
        if getattr(args, "data_parallel_rank", None) is not None:
            engine_kwargs["data_parallel_rank"] = int(args.data_parallel_rank)
        if getattr(args, "data_parallel_size_local", None) is not None:
            engine_kwargs["data_parallel_size_local"] = int(args.data_parallel_size_local)
        if getattr(args, "data_parallel_address", None):
            engine_kwargs["data_parallel_address"] = str(args.data_parallel_address)
        if getattr(args, "data_parallel_rpc_port", None) is not None:
            engine_kwargs["data_parallel_rpc_port"] = int(args.data_parallel_rpc_port)
        if getattr(args, "data_parallel_backend", None):
            engine_kwargs["data_parallel_backend"] = str(args.data_parallel_backend)
        if getattr(args, "distributed_executor_backend", None):
            engine_kwargs["distributed_executor_backend"] = str(args.distributed_executor_backend)
        if getattr(args, "enable_expert_parallel", None) is not None:
            engine_kwargs["enable_expert_parallel"] = bool(args.enable_expert_parallel)
        if getattr(args, "max_length", None) is not None:
            engine_kwargs["max_model_len"] = int(args.max_length)

        for key in (
            "block_size",
            "num_gpu_blocks",
            "num_cpu_blocks",
            "forced_num_gpu_blocks",
            "num_gpu_blocks_override",
        ):
            value = getattr(args, key, None)
            if value is not None:
                engine_kwargs[key] = int(value)

        for key, caster in (
            ("gpu_memory_utilization", float),
            ("max_num_batched_tokens", int),
            ("max_num_seqs", int),
            ("swap_space", int),
        ):
            value = config.get(key)
            if value is not None:
                engine_kwargs[key] = caster(value)

        if "max_num_seqs" not in engine_kwargs:
            fallback_batch = config.get("max_batch_size")
            if fallback_batch is not None:
                engine_kwargs["max_num_seqs"] = int(fallback_batch)

        for key in ("dtype", "kv_cache_dtype"):
            value = config.get(key)
            if value is not None:
                engine_kwargs[key] = value

        if limit_mm is not None:
            engine_kwargs["limit_mm_per_prompt"] = limit_mm

        return engine_kwargs

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

    def _resolve_mm_limits(self, config: Dict[str, Any], args: SimpleNamespace) -> Optional[Dict[str, int]]:
        """Resolve explicit multi-modal limits with a conservative backend default."""

        # STEP 1: Prefer explicit limits from backend config or normalized args.
        limit_cfg = config.get("limit_mm_per_prompt") or getattr(args, "limit_mm_per_prompt", None)
        if isinstance(limit_cfg, dict) and limit_cfg:
            limits = {str(key): int(value) for key, value in limit_cfg.items()}
            logger.info("vllm_backend using multimodal limits source=config value={}", limits)
            return limits

        # STEP 2: Fall back to environment overrides for legacy setups.
        env_image = os.environ.get("GAGE_EVAL_VLLM_IMAGE_LIMIT")
        env_audio = os.environ.get("GAGE_EVAL_VLLM_AUDIO_LIMIT")

        limits: Dict[str, int] = {}
        if env_image:
            try:
                limits["image"] = int(env_image)
            except ValueError:
                logger.warning("vllm_backend ignoring invalid GAGE_EVAL_VLLM_IMAGE_LIMIT={!r}", env_image)
        if env_audio:
            try:
                limits["audio"] = int(env_audio)
            except ValueError:
                logger.warning("vllm_backend ignoring invalid GAGE_EVAL_VLLM_AUDIO_LIMIT={!r}", env_audio)

        if limits:
            logger.info("vllm_backend using multimodal limits source=env value={}", limits)
            return limits

        # STEP 3: Fall back to a conservative backend default instead of vLLM runtime defaults.
        default_limits = {"image": 1}
        logger.info(
            "vllm_backend using multimodal limits source=backend_default value={}; "
            "set config.limit_mm_per_prompt to make the boundary explicit",
            default_limits,
        )
        return default_limits

    def _init_tokenizer(self, config: Dict[str, Any]):
        return load_hf_tokenizer(config)

    # ------------------------------------------------------------------ #
    # Adapters, rendering, and sampling helpers                            #
    # ------------------------------------------------------------------ #
    def _convert_output(self, result: Any, output_type: str) -> Any:
        if output_type in {"text", "beam"}:
            return convert_text_like_output(result)
        if output_type in {"reward", "loss", "prompt_tokens", "next_token_prob"}:
            return result
        return result

    def _render_prompt(self, prepared: Dict[str, Any]) -> str:
        policy = ChatTemplatePolicy(mode=self._chat_template_mode)
        caps = BackendCapabilities(
            supports_mm=self._supports_multimodal_requests(),
            has_processor_chat_template=bool(self._processor),
        )
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
        """Generate `prompt_token_ids` via the backend tokenizer when preprocessors did not provide them."""

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
        self, messages: List[Dict[str, Any]], prompt: str, chat_template_kwargs: Optional[Dict[str, Any]] = None,
        *, preserve_multimodal: bool = False
    ) -> Optional[str]:
        # NOTE: For multimodal, don't normalize messages - processor needs raw image_url structure
        # to insert proper vision tokens like <|vision_start|><|image_pad|><|vision_end|>
        proc_messages = messages if preserve_multimodal else normalize_messages_safe(messages)
        rendered = render_with_processor(self._processor, proc_messages, prompt, chat_template_kwargs, skip_normalize=preserve_multimodal)
        if rendered is not None:
            return rendered
        tok = getattr(self, "_tokenizer", None)
        tok_apply = getattr(tok, "apply_chat_template", None)
        if tok_apply:
            try:
                rendered_tok = tok_apply(
                    proc_messages,
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

    def _render_text_fallback(self, messages: List[Dict[str, Any]], prompt: str) -> str:
        if prompt:
            return prompt
        return simple_render_messages(messages)

    def _prompt_has_mm_tokens(self, prompt: str) -> bool:
        if not prompt:
            return False
        lowered = prompt.lower()
        if "<|vision_start|>" in lowered or "<|image_pad|>" in lowered:
            return True
        return re.search(r"<image\\s*\\d*>", lowered) is not None

    def _strip_mm_tokens(self, prompt: str) -> str:
        if not prompt:
            return ""
        cleaned = prompt.replace("<|vision_start|>", "").replace("<|vision_end|>", "").replace("<|image_pad|>", "")
        cleaned = re.sub(r"<image\\s*\\d*>", "", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split())

    def _has_mm_payload(self, mm_payload: Optional[Dict[str, Any]]) -> bool:
        if not mm_payload or not isinstance(mm_payload, dict):
            return False
        for key in ("image", "audio", "video"):
            value = mm_payload.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                if any(item is not None for item in value):
                    return True
            else:
                return True
        return False

    def _prepare_multi_modal_data(self, prepared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sources = collect_multimodal_sources(prepared)
        images = self._load_images(sources.get("image"))
        audios = sources.get("audio") or []
        videos = sources.get("video") or []
        images = [img for img in images if img is not None]
        audios = [au for au in audios if au is not None]
        videos = [v for v in videos if v is not None]
        if not images and not audios and not videos:
            return None
        result: Dict[str, Any] = {}
        if images:
            result["image"] = images
        if audios:
            result["audio"] = audios
        if videos:
            result["video"] = videos
        return result

    def _load_images(self, sources) -> List[Any]:
        return load_images(sources)

    def _load_multimodal_payload(self, mm: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return load_multimodal_payload(mm, getattr(self, "_processor", None), logger_prefix="vllm_backend")

    def _retry_generate_legacy(
        self,
        *,
        prompt_input: Any,
        prompt: str,
        extra_inputs: Any,
        sampling_params: Any,
        request_id: str,
        mm_loaded: bool,
        error: TypeError,
    ) -> Any:
        """Retry generation with keyword arguments for older or wrapped runtimes."""

        if mm_loaded:
            self._mm_supported = False
            self._mm_strategy = "disabled"
            if self._strict_mm:
                raise RuntimeError(
                    f"multi_modal_data unsupported in this vLLM version; strict mode enabled ({error})"
                ) from error

        logger.warning(
            "vllm_backend generate encountered TypeError ({}); retrying with legacy prompt kwargs",
            error,
        )

        generate_kwargs: Dict[str, Any] = {
            "sampling_params": sampling_params,
            "request_id": request_id,
        }
        prompt_for_retry = prompt_input.get("prompt") if isinstance(prompt_input, dict) else prompt_input
        generate_kwargs["prompt"] = prompt_for_retry or prompt

        merged_inputs: Dict[str, Any] = {}
        if isinstance(prompt_input, dict):
            merged_inputs.update(prompt_input)
        if isinstance(extra_inputs, dict):
            merged_inputs.update(extra_inputs)

        prompt_token_ids = normalize_prompt_token_ids(merged_inputs.get("prompt_token_ids"))
        if prompt_token_ids is None:
            prompt_token_ids = normalize_prompt_token_ids(merged_inputs.get("input_ids"))
        if isinstance(prompt_token_ids, int):
            prompt_token_ids = [prompt_token_ids]
        if prompt_token_ids is not None:
            generate_kwargs["prompt_token_ids"] = prompt_token_ids

        return self.model.generate(**generate_kwargs)


def _resolve_sample_n_override(sampling_base: Dict[str, Any]) -> Optional[int]:
    candidate = sampling_base.get("n") or sampling_base.get("num_samples")
    if candidate is None:
        return None
    try:
        value = int(candidate)
    except (TypeError, ValueError):
        return None
    return value if value > 1 else None


def _strip_sample_n_params(sampling_base: Dict[str, Any]) -> Dict[str, Any]:
    stripped = dict(sampling_base)
    stripped.pop("n", None)
    stripped.pop("num_samples", None)
    return stripped
