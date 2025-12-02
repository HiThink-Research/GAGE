"""HuggingFace AutoModelForImageTextToText backend adapted from lighteval."""

from __future__ import annotations

import inspect
import os
import re
from datetime import timedelta
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.config.vlm_transformers import VLMTransformersBackendConfig
from gage_eval.role.model.runtime import BackendCapabilities, ChatTemplateMixin, ChatTemplatePolicy
from gage_eval.utils.multimodal import load_multimodal_data
from gage_eval.utils.chat_templates import get_fallback_template

try:  # pragma: no cover - optional dependency
    from accelerate import Accelerator, InitProcessGroupKwargs
    from accelerate.utils import get_max_memory
except ImportError:  # pragma: no cover - accelerate optional
    Accelerator = None
    InitProcessGroupKwargs = None
    get_max_memory = None


@registry.asset(
    "backends",
    "vlm_transformers",
    desc="HuggingFace Vision-Language Transformers 本地推理后端",
    tags=("vlm", "local", "multimodal"),
    modalities=("text", "vision", "audio"),
    config_schema_ref="gage_eval.role.model.config.vlm_transformers:VLMTransformersBackendConfig",
)
class VLMTransformersBackend(EngineBackend):
    """Backend that executes HuggingFace vision-language models locally."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._parsed_config = VLMTransformersBackendConfig(**config)
        self._chat_template_mode = self._parsed_config.use_chat_template_vlm or "auto"
        self._chat_template_policy = ChatTemplatePolicy(mode=self._chat_template_mode)
        self._fallback_template = get_fallback_template("vlm")
        self._image_resize_opts = self._build_image_resize_opts()
        self._lock = Lock()
        self._cfg_tokenizer_path = (
            config.get("processor_name_or_path")
            or config.get("tokenizer_path")
            or config.get("model_path")
            or config.get("model_name_or_path")
        )
        # 本地 Native 执行模式
        config = dict(config)
        config.setdefault("execution_mode", "native")
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def load_model(self, _: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            from transformers.utils.quantization_config import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers + torch are required for VLMTransformersBackend") from exc

        self._torch = torch
        self.accelerator = self._init_accelerator()
        self.device = self._resolve_device()

        cfg = self._parsed_config
        model_parallel, max_memory, device_map = self._init_model_parallel(cfg.model_parallel)
        quantization_config = self._build_quantization_config(BitsAndBytesConfig)
        torch_dtype = self._resolve_torch_dtype()
        self._torch_dtype = torch_dtype

        model_name = cfg.model_path or cfg.model_name_or_path
        processor_name = cfg.processor_name_or_path or model_name
        revision = self._compose_revision(cfg.revision, cfg.subfolder)
        self._chat_template_mode = cfg.use_chat_template_vlm or "auto"
        self._chat_template_policy = ChatTemplatePolicy(mode=self._chat_template_mode)
        self._fallback_template = get_fallback_template("vlm")

        attn_impl = cfg.attn_implementation
        if attn_impl is None and self._torch.cuda.is_available():
            # H100 推荐默认开启 FlashAttention 2
            attn_impl = "flash_attention_2"

        model_kwargs = {
            "revision": revision,
            "torch_dtype": torch_dtype,
            "trust_remote_code": cfg.trust_remote_code,
            "quantization_config": quantization_config,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        if cfg.cache_dir:
            model_kwargs["cache_dir"] = cfg.cache_dir
        if cfg.device_map:
            model_kwargs["device_map"] = cfg.device_map
        elif device_map:
            model_kwargs["device_map"] = device_map
        if max_memory:
            model_kwargs["max_memory"] = max_memory

        model = AutoModelForImageTextToText.from_pretrained(model_name, **_drop_none(model_kwargs))
        model.eval()
        torch.set_grad_enabled(False)

        if cfg.compile:
            raise NotImplementedError("Compiling VLM models is not supported yet")
        if not model_parallel and quantization_config is None:
            logger.info("Loading VLM model on device {}", self.device)
            model = model.to(self.device)

        processor_kwargs = dict(cfg.extra_processor_kwargs or {})
        processor_kwargs.update(
            {
                "revision": revision,
                "trust_remote_code": cfg.trust_remote_code,
                "padding_side": processor_kwargs.get("padding_side", "left"),
                "truncation_side": processor_kwargs.get("truncation_side", "left"),
            }
        )
        if cfg.use_fast_image_processor is not None:
            processor_kwargs["use_fast"] = cfg.use_fast_image_processor
        if cfg.cache_dir:
            processor_kwargs["cache_dir"] = cfg.cache_dir

        processor = AutoProcessor.from_pretrained(processor_name, **processor_kwargs)
        tokenizer = getattr(processor, "tokenizer", None)
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model

        self.pad_token_id = self._resolve_pad_token_id()
        self.eos_token_id = self._resolve_eos_token_id()

        self._max_length = self._resolve_max_length(model.config)
        self._processor_call_kwargs = {
            "padding": "longest",
            "truncation": "longest_first",
            "add_special_tokens": cfg.add_special_tokens,
            "return_tensors": "pt",
        }
        if self._max_length:
            self._processor_call_kwargs["max_length"] = max(1, self._max_length - 1)

        self._base_generation_kwargs = self._build_base_generation_kwargs()
        self._default_stop_sequences = cfg.generation_parameters.stop_sequences()
        self.batch_size = cfg.batch_size

        return model

    # ------------------------------------------------------------------ #
    # Execution                                                          #
    # ------------------------------------------------------------------ #
    def generate_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """尝试批处理多模态请求，失败则回退串行。"""

        if not requests:
            return []

        for req in requests:
            self._check_tokenizer_conflict(req)

        # 若采样参数不一致，直接串行避免错配。
        sampling_params_list = [
            req.get("sampling_params") or (req.get("sample") or {}).get("sampling_params") or {} for req in requests
        ]
        first_sampling = sampling_params_list[0]
        if any(params != first_sampling for params in sampling_params_list[1:]):
            return [self.generate(req) for req in requests]

        prompts: List[str] = []
        images_batch: List[Any] = []
        audios_batch: List[Any] = []

        for req in requests:
            sample = req.get("sample", {})
            raw_inputs = req.get("inputs") or sample.get("inputs") or {}
            messages = req.get("messages") or sample.get("messages") or []
            message_images = self._extract_images_from_messages(messages)
            prompt = self._resolve_prompt(req, sample, raw_inputs, messages, message_images)
            prompts.append(prompt)

            mm = raw_inputs.get("multi_modal_data") if isinstance(raw_inputs, dict) else None
            if not mm:
                mm = sample.get("multi_modal_data")
            if (not mm) and message_images:
                mm = {"image": message_images}
            img_paths = None
            audio_paths = None
            if isinstance(mm, dict):
                img_paths = mm.get("image") or mm.get("images")
                audio_paths = mm.get("audio") or mm.get("audios")

            images, audios = load_multimodal_data(
                self.processor,
                img_paths,
                audio_paths,
                resize_opts=self._image_resize_opts,
            )
            images_batch.append(images or [])
            audios_batch.append(audios or [])

        payload = dict(self._processor_call_kwargs)
        payload["text"] = prompts
        if any(images_batch):
            payload["images"] = images_batch
        if any(audios_batch):
            payload["audios"] = audios_batch
            if not self._processor_accepts_arg("audios") and self._processor_accepts_arg("audio"):
                payload["audio"] = payload.pop("audios")

        try:
            encoding = self.processor(**payload)
            model_inputs, input_len, _ = self._batch_encoding_to_tensors(encoding)
            generation_kwargs, stop_sequences = self._build_generation_kwargs(first_sampling)
            num_return_sequences = generation_kwargs.pop("num_return_sequences", 1)
            do_sample = generation_kwargs.pop("do_sample")

            with self._lock:
                with self._torch.inference_mode():
                    outputs = self.model.generate(
                        **model_inputs,
                        num_return_sequences=num_return_sequences,
                        do_sample=do_sample,
                        **generation_kwargs,
                    )

            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            if sequences.dim() == 1:
                sequences = sequences.unsqueeze(0)
            generated_tokens = sequences[:, input_len:]
            decoded = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)
            stop_sequences = stop_sequences or self._default_stop_sequences
            decoded = [_apply_stop_sequences(text, stop_sequences) for text in decoded]

            results: List[Dict[str, Any]] = []
            for idx, text in enumerate(decoded):
                results.append(
                    {
                        "answer": text,
                        "token_ids": generated_tokens[idx].tolist(),
                        "_batch_path": "native_batch",
                    }
                )
            for req, res in zip(requests, results):
                self._attach_template_metadata(req, res)
            return results
        except Exception as exc:
            logger.warning("VLMTransformersBackend generate_batch fallback to serial due to: {}", exc)
            return [self.generate(req) for req in requests]

    def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        sample = request.get("sample", {})
        raw_inputs = request.get("inputs") or sample.get("inputs")
        messages = request.get("messages") or sample.get("messages") or []
        message_images = self._extract_images_from_messages(messages)
        self._check_tokenizer_conflict(request)
        prompt = self._resolve_prompt(request, sample, raw_inputs, messages, message_images)

        model_inputs, input_len, padded_tokens = self._prepare_model_inputs(prompt, raw_inputs, messages, message_images)
        sampling_params = request.get("sampling_params") or {}
        generation_kwargs, stop_sequences = self._build_generation_kwargs(sampling_params)
        num_return_sequences = generation_kwargs.pop("num_return_sequences", 1)
        do_sample = generation_kwargs.pop("do_sample")

        with self._lock:  # Avoid overlapping accelerator contexts
            with self._torch.inference_mode():
                outputs = self.model.generate(
                    **model_inputs,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    **generation_kwargs,
                )

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        generated_tokens = sequences[:, input_len:]
        decoded = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)
        stop_sequences = stop_sequences or self._default_stop_sequences
        decoded = [_apply_stop_sequences(text, stop_sequences) for text in decoded]

        responses = []
        for idx, text in enumerate(decoded):
            responses.append(
                {
                    "answer": text,
                    "token_ids": generated_tokens[idx].tolist(),
                }
            )

        result = responses[0]
        if len(responses) > 1:
            result = dict(result)
            result["alternatives"] = responses[1:]
        result.update(
            {
                "input_length": input_len,
                "padded_tokens": padded_tokens,
            }
        )
        if stop_sequences:
            result["stop_sequences"] = stop_sequences
        self._attach_template_metadata(request, result)
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _init_accelerator(self):
        if Accelerator is None:
            return None
        timeout_seconds = max(60, int(self._parsed_config.distributed_timeout))
        try:
            return Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_seconds))])
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Failed to initialize accelerate (fallback to single-device): {}", exc)
            return None

    def _resolve_device(self):
        cfg_device = self._parsed_config.device
        if cfg_device:
            return self._torch.device(cfg_device)
        if self.accelerator:
            return self.accelerator.device
        if self._torch.cuda.is_available():
            return self._torch.device("cuda")
        return self._torch.device("cpu")

    def _init_model_parallel(self, requested: Optional[bool]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        torch = self._torch
        if not torch.cuda.is_available():
            return False, None, None
        if self.accelerator is None:
            if requested:
                raise RuntimeError("accelerate must be installed to enable model_parallel execution")
            return False, None, None

        num_local = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        total_gpus = torch.cuda.device_count()
        if total_gpus == 0:
            return False, None, None
        num_machines = max(1, total_gpus // max(1, num_local))
        max_memory_devices = None
        if requested is None:
            if get_max_memory is None:
                requested = False
            else:
                max_memory_devices = {k: v for k, v in get_max_memory().items() if k != "cpu"}
                requested = bool(num_local < len(max_memory_devices))
        elif requested and get_max_memory is not None:
            max_memory_devices = {k: v for k, v in get_max_memory().items() if k != "cpu"}

        if num_machines == 1 and requested:
            logger.info("Single-machine environment detected, disabling model-parallel")
            requested = False

        if not requested:
            return False, None, None

        if max_memory_devices is None and get_max_memory is not None:
            max_memory_devices = {k: v for k, v in get_max_memory().items() if k != "cpu"}
        max_mem_this_process = None
        if max_memory_devices:
            process_index = self.accelerator.process_index % max(1, num_local)
            max_mem_this_process = {
                k: v for k, v in max_memory_devices.items() if k % max(1, num_local) == process_index
            }
        logger.info(
            "Enabling model-parallel loading with max_memory={} and device_map='auto'",
            max_mem_this_process,
        )
        return True, max_mem_this_process, "auto"

    def _build_quantization_config(self, bitsandbytes_cls):
        dtype = (self._parsed_config.dtype or "").lower()
        if dtype == "4bit":
            return bitsandbytes_cls(load_in_4bit=True, bnb_4bit_compute_dtype=self._torch.float16)
        if dtype == "8bit":
            return bitsandbytes_cls(load_in_8bit=True)
        return None

    def _resolve_torch_dtype(self):
        dtype = self._parsed_config.dtype
        if dtype in (None, "", "auto", "4bit", "8bit"):
            return None
        if isinstance(dtype, str) and hasattr(self._torch, dtype):
            return getattr(self._torch, dtype)
        torch_dtype_type = type(self._torch.float16)
        if isinstance(dtype, torch_dtype_type):  # pragma: no cover - defensive
            return dtype
        return None

    def _compose_revision(self, revision: str, subfolder: Optional[str]) -> str:
        if subfolder:
            return f"{revision}/{subfolder}"
        return revision

    def _resolve_pad_token_id(self) -> Optional[int]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer and tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer and tokenizer.pad_token_id is not None:
            return tokenizer.pad_token_id
        return getattr(self.model.config, "pad_token_id", None) or getattr(self.model.config, "bos_token_id", None)

    def _resolve_eos_token_id(self) -> Optional[int]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer and tokenizer.eos_token_id is not None:
            return tokenizer.eos_token_id
        return getattr(self.model.config, "eos_token_id", None)

    def _resolve_max_length(self, config) -> Optional[int]:
        cfg = self._parsed_config
        if cfg.max_length:
            return cfg.max_length
        text_config = getattr(config, "get_text_config", None)
        if callable(text_config):
            text_cfg = text_config()
            if hasattr(text_cfg, "max_position_embeddings"):
                return text_cfg.max_position_embeddings
        if hasattr(config, "max_position_embeddings"):
            return config.max_position_embeddings
        logger.warning("Unable to infer max_length from config; defaulting to 2048 tokens")
        return 2048

    def _build_base_generation_kwargs(self) -> Dict[str, Any]:
        cfg = self._parsed_config
        params = cfg.generation_parameters.to_transformers_dict()
        if cfg.generation_size and not params.get("max_new_tokens"):
            params["max_new_tokens"] = cfg.generation_size
        params.setdefault("max_new_tokens", 512)
        params.update(
            {
                "pad_token_id": self.pad_token_id,
                "eos_token_id": self.eos_token_id,
                "return_dict_in_generate": True,
                "output_scores": False,
                "renormalize_logits": True,
            }
        )
        return params

    def _resolve_prompt(
        self,
        request: Dict[str, Any],
        sample: Dict[str, Any],
        raw_inputs,
        messages: Optional[List[Dict[str, Any]]] = None,
        message_images: Optional[List[Any]] = None,
    ) -> str:
        caps = BackendCapabilities(
            supports_mm=True, has_processor_chat_template=bool(getattr(self.processor, "apply_chat_template", None))
        )
        request["cache_suffix"] = ChatTemplateMixin.get_cache_suffix("vlm", self._chat_template_policy, caps)
        explicit_prompt = None
        if isinstance(raw_inputs, dict) and raw_inputs.get("prompt"):
            explicit_prompt = str(raw_inputs["prompt"])

        fallback_prompt = ""
        for key in ("prompt", "question", "text"):
            candidate = request.get(key) or sample.get(key)
            if candidate:
                fallback_prompt = str(candidate)
                break

        messages = messages if messages is not None else (request.get("messages") or sample.get("messages") or [])
        if message_images is None:
            message_images = self._extract_images_from_messages(messages)

        prompt = explicit_prompt or ""
        rendered = self._render_messages(messages, payload=request)
        if not prompt and rendered:
            prompt = rendered
        if not prompt:
            prompt = fallback_prompt
        # 避免空提示导致 image tokens > text tokens 的报错，至少提供占位提示
        if (messages or (isinstance(raw_inputs, dict) and (raw_inputs.get("multi_modal_data")))) and not prompt:
            prompt = "Please describe the image and answer the question."
        if message_images:
            prompt = self._normalize_image_placeholders(prompt, len(message_images))
        return prompt

    def _render_messages(self, messages: List[Dict[str, Any]], payload: Optional[Dict[str, Any]] = None) -> str:
        if not messages:
            return ""
        payload = payload or {}
        policy = self._chat_template_policy
        caps = BackendCapabilities(supports_mm=True, has_processor_chat_template=bool(getattr(self.processor, "apply_chat_template", None)))

        if not ChatTemplateMixin.should_render(payload, policy):
            payload["cache_suffix"] = ChatTemplateMixin.get_cache_suffix("vlm", policy, caps)
            return self._simple_render(messages)

        template_source = ChatTemplateMixin.select_template("vlm", policy, caps)
        apply_template = getattr(self.processor, "apply_chat_template", None) if template_source == "model" else None
        fallback_tpl = None if template_source == "model" else self._fallback_template
        rendered = ChatTemplateMixin.render(
            messages,
            template_fn=apply_template,
            fallback_fn=lambda msgs: self._fallback_render(msgs, fallback_tpl),
            add_generation_prompt=True,
            chat_template=fallback_tpl,
        )
        payload["cache_suffix"] = ChatTemplateMixin.get_cache_suffix("vlm", policy, caps)
        payload["chat_template_mode"] = "backend"
        payload["template_source"] = "model" if apply_template else "fallback"
        payload["rendered_by"] = "backend"
        return rendered

    def _simple_render(self, messages: List[Dict[str, Any]]) -> str:
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

    def _prepare_model_inputs(self, prompt: str, raw_inputs, messages: List[Dict[str, Any]], message_images: List[Any]):
        if isinstance(raw_inputs, dict):
            return self._prepare_multimodal_inputs(prompt, raw_inputs, messages, message_images)
        if messages:
            # 消息中含图片但 inputs 为空时，仍需构造多模态输入
            return self._prepare_multimodal_inputs(prompt, {}, messages, message_images)
        if isinstance(raw_inputs, list):
            return self._prepare_token_inputs(raw_inputs)
        return self._prepare_text_inputs(prompt)

    def _prepare_text_inputs(self, prompt: str):
        payload = dict(self._processor_call_kwargs)
        payload["text"] = [prompt]
        encoding = self.processor(**payload)
        return self._batch_encoding_to_tensors(encoding)

    def _prepare_multimodal_inputs(
        self,
        prompt: str,
        raw_inputs: Dict[str, Any],
        messages: List[Dict[str, Any]],
        message_images: List[Any],
    ):
        payload = dict(self._processor_call_kwargs)
        payload["text"] = [raw_inputs.get("prompt") or prompt]
        mm_data = raw_inputs.get("multi_modal_data") or {}
        if not mm_data and message_images:
            mm_data = {"image": message_images}
        image_paths = mm_data.get("image") or mm_data.get("images")
        audio_paths = mm_data.get("audio") or mm_data.get("audios")
        images, audios = load_multimodal_data(
            self.processor,
            image_paths,
            audio_paths,
            resize_opts=self._image_resize_opts,
        )
        if images:
            payload["images"] = [images]
        if audios:
            payload["audios"] = [audios]
            if not self._processor_accepts_arg("audios") and self._processor_accepts_arg("audio"):
                payload["audio"] = payload.pop("audios")
        encoding = self.processor(**payload)
        return self._batch_encoding_to_tensors(encoding)

    def _processor_accepts_arg(self, arg_name: str) -> bool:
        try:
            signature = inspect.signature(self.processor.__call__)
        except (TypeError, ValueError):  # pragma: no cover
            return False
        return arg_name in signature.parameters

    @staticmethod
    def _extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Any]:
        """Gather image URLs/data from chat messages for multi-modal inputs."""

        urls: List[Any] = []
        for message in messages or []:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for fragment in content:
                if isinstance(fragment, dict) and fragment.get("type") == "image_url":
                    image_url = fragment.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                    else:
                        url = image_url
                    if url is None:
                        url = fragment.get("url")
                    if url is not None:
                        urls.append(url)
        return urls

    def _prepare_token_inputs(self, token_ids: List[int]):
        tensor = self._torch.tensor([token_ids], device=self.device)
        attention_mask = self._torch.ones_like(tensor, device=self.device)
        return {"input_ids": tensor, "attention_mask": attention_mask}, tensor.shape[-1], 0

    def _batch_encoding_to_tensors(self, encoding):
        encoding = encoding.to(self.device)
        tensors = {k: v for k, v in encoding.items()}
        if self._torch_dtype is not None:
            tensors = {
                k: self._cast_tensor(v) if hasattr(v, "is_floating_point") and v.is_floating_point() else v
                for k, v in tensors.items()
            }
        input_ids = tensors.get("input_ids")
        attention_mask = tensors.get("attention_mask")
        input_len = int(input_ids.shape[-1]) if input_ids is not None else 0
        padded_tokens = 0
        if attention_mask is not None:
            mask = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
            padded_tokens = int((mask == 0).sum().item())
        return tensors, input_len, padded_tokens

    def _cast_tensor(self, tensor):
        if self._torch_dtype is None:
            return tensor
        return tensor.to(self._torch_dtype)

    @staticmethod
    def _normalize_image_placeholders(prompt: str, image_count: int) -> str:
        """Ensure prompt carries <image> placeholders aligned with loaded images."""

        marker = "<image>"
        if prompt:
            prompt = re.sub(r"<image\s*\d*>", marker, prompt, flags=re.IGNORECASE)
        current = prompt.lower().count(marker) if prompt else 0
        missing = max(0, image_count - current)
        if missing > 0:
            prefix = " ".join([marker] * missing)
            prompt = (prefix + " " + prompt).strip() if prompt else prefix
        return prompt

    def _build_generation_kwargs(self, sampling_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[List[str]]]:
        kwargs = dict(self._base_generation_kwargs)
        overrides = {k: sampling_params.get(k) for k in ("max_new_tokens", "temperature", "top_p", "top_k")}
        overrides.update(
            {
                "repetition_penalty": sampling_params.get("repetition_penalty"),
                "presence_penalty": sampling_params.get("presence_penalty"),
                "frequency_penalty": sampling_params.get("frequency_penalty"),
            }
        )
        kwargs.update({k: v for k, v in overrides.items() if v is not None})

        num_return = sampling_params.get("n") or sampling_params.get("num_return_sequences") or 1
        kwargs["num_return_sequences"] = max(1, int(num_return))

        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p")
        do_sample = sampling_params.get("do_sample")
        if do_sample is None:
            do_sample = (temperature is not None and temperature > 0) or (top_p is not None and top_p < 1.0)
        kwargs["do_sample"] = bool(do_sample)

        stop_sequences = sampling_params.get("stop") or sampling_params.get("stop_sequences")
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        return kwargs, stop_sequences

    def _build_image_resize_opts(self) -> Dict[str, Any]:
        cfg = self._parsed_config
        opts: Dict[str, Any] = {}
        if cfg.max_pixels is not None:
            opts["max_pixels"] = cfg.max_pixels
        if cfg.min_pixels is not None:
            opts["min_pixels"] = cfg.min_pixels
        if cfg.image_factor is not None:
            opts["factor"] = cfg.image_factor
        return opts

    def _check_tokenizer_conflict(self, payload: Dict[str, Any]) -> None:
        dataset_tok = payload.get("_tokenizer_path") or (payload.get("sample") or {}).get("_tokenizer_path")
        backend_tok = self._cfg_tokenizer_path
        if dataset_tok and backend_tok and str(dataset_tok) != str(backend_tok):
            raise ValueError(f"Conflicting tokenizer_path: dataset={dataset_tok} backend={backend_tok}")

    def _attach_template_metadata(self, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
        meta_keys = ("chat_template_mode", "template_source", "rendered_by", "cache_suffix")
        sample = payload.get("sample") or {}
        for key in meta_keys:
            value = payload.get(key)
            if value is None:
                value = sample.get(key)
            if value is not None and key not in result:
                result[key] = value
        if "_tokenizer_path" not in result and self._cfg_tokenizer_path:
            result["_tokenizer_path"] = self._cfg_tokenizer_path


def _apply_stop_sequences(text: str, stop) -> str:
    if not stop:
        return text
    if isinstance(stop, str):
        stop = [stop]
    for token in stop:
        idx = text.find(token)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def _drop_none(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}
