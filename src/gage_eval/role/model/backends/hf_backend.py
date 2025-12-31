"""Hugging Face transformers backend (text + multi-modal)."""

from __future__ import annotations

import inspect
from threading import Lock
from typing import Any, Dict, Tuple, Optional

from gage_eval.utils.multimodal import load_multimodal_data
from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.utils.cleanup import install_signal_cleanup, torch_gpu_cleanup


@registry.asset(
    "backends",
    "hf",
    desc="HuggingFace Transformers 本地推理后端",
    tags=("llm", "local", "multimodal"),
    modalities=("text", "audio", "vision"),
)
class HFBackend(EngineBackend):
    """Backend that runs HuggingFace transformers locally.

    Compared with the initial stub this version mirrors the behaviour of
    ``predict_async_hf.py``: it supports multi-modal inputs, optional
    `turn_taking` outputs and honours sampling parameters passed via
    ``sampling_params``.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._lock = Lock()
        self.default_params = {
            "max_new_tokens": config.get("max_new_tokens", 512),
            "temperature": config.get("temperature"),
            "top_p": config.get("top_p"),
            "repetition_penalty": config.get("repetition_penalty"),
            "presence_penalty": config.get("presence_penalty"),
            "stop": config.get("stop"),
        }
        self._default_output_type = config.get("output_type", "text")
        cfg = dict(config)
        cfg.setdefault("execution_mode", "native")
        super().__init__(cfg)
        install_signal_cleanup(self.shutdown)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover - heavy dependency
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers + torch are required for HFBackend") from exc

        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("HFBackend requires 'model_name'")

        tokenizer_name = config.get("tokenizer_name") or model_name
        trust_remote_code = config.get("trust_remote_code", True)
        tokenizer_kwargs = dict(config.get("tokenizer_kwargs", {}))
        tokenizer_kwargs.setdefault("trust_remote_code", trust_remote_code)
        processor = transformers.AutoProcessor.from_pretrained(tokenizer_name, **tokenizer_kwargs)

        model_kwargs = dict(config.get("model_kwargs", {}))
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)
        dtype = config.get("dtype", "auto")
        if dtype == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif dtype == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        device = config.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        tokenizer = getattr(processor, "tokenizer", processor)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model
        self._torch = torch

    # ------------------------------------------------------------------ #
    # Execution                                                          #
    # ------------------------------------------------------------------ #
    def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        sample = request.get("sample", {})
        prompt = request.get("prompt") or sample.get("prompt") or sample.get("text") or ""
        raw_inputs = request.get("inputs") or sample.get("inputs")
        sampling_params = self._prepare_sampling_params(
            request.get("sampling_params"),
            sample.get("sampling_params"),
        )
        output_type = (
            request.get("output_type")
            or sample.get("output_type")
            or self._default_output_type
        )

        with self._lock:
            model_inputs, input_len = self._prepare_model_inputs(prompt, raw_inputs)
            if output_type == "text":
                return self._generate_text(model_inputs, input_len, sampling_params)
            if output_type == "turn_taking":
                return self._generate_turn_taking(model_inputs)
            raise NotImplementedError(f"HFBackend output_type '{output_type}' is not supported")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _prepare_sampling_params(
        self,
        runtime_params: Optional[Dict[str, Any]],
        sample_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        params = dict(self.default_params)
        if sample_params:
            params.update(sample_params)
        if runtime_params:
            params.update(runtime_params)
        return params

    def _prepare_model_inputs(self, prompt: str, raw_inputs) -> Tuple[Dict[str, Any], int]:
        if isinstance(raw_inputs, dict):  # multi-modal dict produced by preprocess
            return self._prepare_multimodal_inputs(raw_inputs)
        if isinstance(raw_inputs, list):  # token ids
            return self._prepare_token_inputs(raw_inputs)
        return self._prepare_text_inputs(prompt)

    def _prepare_text_inputs(self, prompt: str) -> Tuple[Dict[str, Any], int]:
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = encoded["input_ids"].shape[1]
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}, input_len

    def _prepare_token_inputs(self, token_ids) -> Tuple[Dict[str, Any], int]:
        input_ids = self._torch.tensor([token_ids], device=self.model.device)
        attention_mask = self._torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, len(token_ids)

    def _prepare_multimodal_inputs(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        prompt = inputs.get("prompt") or ""
        mm_data = inputs.get("multi_modal_data") or {}
        images, audios = load_multimodal_data(
            self.processor,
            mm_data.get("image"),
            mm_data.get("audio"),
        )
        payload = dict(
            text=[prompt],
            **{k: v for k, v in (("images", images), ("audios", audios)) if v},
            return_tensors="pt",
        )
        if audios:
            try:
                payload["sampling_rate"] = self.processor.feature_extractor.sampling_rate
            except AttributeError:
                pass
            if (
                "audios" not in inspect.signature(self.processor).parameters
                and "audio" in inspect.signature(self.processor).parameters
            ):
                payload["audio"] = payload.pop("audios")

        tensors = self.processor(**payload).to(self.model.device)
        input_len = tensors["input_ids"].shape[1]
        return tensors, input_len

    def _generate_text(self, model_inputs: Dict[str, Any], input_len: int, params: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = self._build_generation_kwargs(params)
        output = self.model.generate(**model_inputs, **kwargs)
        generated_tokens = output[0, input_len:].tolist()
        stop_sequences = params.get("stop")
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        text = _apply_stop_sequences(text, stop_sequences)
        return {"answer": text, "token_ids": generated_tokens}

    def _generate_turn_taking(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        model = getattr(self.model, "thinker", self.model)
        outputs = model(
            **model_inputs,
            output_logits=False,
            is_turn_taking=True,
        )
        audio_token_id = getattr(model.config, "audio_token_index", None)
        if audio_token_id is None:
            raise RuntimeError("Model config missing audio_token_index for turn-taking output")
        input_ids = model_inputs["input_ids"][0]
        mask = (input_ids == audio_token_id).cpu().numpy()[::-1]
        end = mask.argmax()
        remaining = mask[end:]
        if remaining.size == 0:
            raise RuntimeError("Unable to locate audio token span for turn-taking output")
        start = remaining.argmin() + end if remaining.any() else len(mask)
        logits = outputs.logits[0, -start: -end or None]
        probs = logits.squeeze(-1).tolist()
        return {"turn_probabilities": probs}

    def _build_generation_kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        temperature = params.get("temperature")
        top_p = params.get("top_p")
        do_sample = temperature not in (None, 0) or (top_p is not None and top_p < 1.0)
        kwargs = {
            "max_new_tokens": params.get("max_new_tokens", 512),
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample and top_p is not None else None,
            "repetition_penalty": params.get("repetition_penalty"),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        return {k: v for k, v in kwargs.items() if v is not None}

    def shutdown(self) -> None:  # pragma: no cover - best-effort cleanup
        with self._lock:
            try:
                model = getattr(self, "model", None)
                if model and hasattr(model, "to"):
                    model.to("cpu")
            except Exception:
                pass
        torch_gpu_cleanup()


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
