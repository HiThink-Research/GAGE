"""Nanotron distributed backend (adapts lighteval Nanotron model)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.config.nanotron import NanotronBackendConfig
from gage_eval.utils.cleanup import install_signal_cleanup, torch_gpu_cleanup


@registry.asset(
    "backends",
    "nanotron",
    desc="Nanotron tensor/pipeline-parallel inference backend",
    tags=("vlm", "local", "distributed"),
    modalities=("text",),
    config_schema_ref="gage_eval.role.model.config.nanotron:NanotronBackendConfig",
)
class NanotronBackend(EngineBackend):
    """Backend wrapping Nanotron checkpoints for local inference."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._parsed_config = NanotronBackendConfig(**config)
        self._runner: Optional[_NanotronRunner] = None
        super().__init__(config)
        install_signal_cleanup(self.shutdown)

    def load_model(self, _: Dict[str, Any]):
        self._runner = _NanotronRunner(self._parsed_config)
        return self._runner

    def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not self._runner:
            raise RuntimeError("Nanotron backend is not initialised correctly")
        sample = request.get("sample", {})
        prompt = (
            request.get("prompt")
            or sample.get("prompt")
            or sample.get("question")
            or sample.get("text")
            or ""
        )
        if not isinstance(prompt, str):
            prompt = str(prompt)

        sampling_params = request.get("sampling_params") or {}
        dataset_params = sample.get("sampling_params") or {}
        merged_params = dict(dataset_params)
        merged_params.update(sampling_params)
        stop_candidates = merged_params.get("stop") or merged_params.get("stop_sequences") or []
        if isinstance(stop_candidates, str):
            stop_sequences: List[str] = [stop_candidates]
        elif isinstance(stop_candidates, list):
            stop_sequences = [str(item) for item in stop_candidates if item is not None]
        else:
            stop_sequences = []

        answer, token_ids = self._runner.generate(
            prompt=prompt,
            stop_sequences=stop_sequences,
            sampling_params=merged_params,
        )
        return {"answer": answer, "token_ids": token_ids}

    def shutdown(self) -> None:  # pragma: no cover - best-effort GPU cleanup
        torch_gpu_cleanup()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _GenerationArgs:
    """Subset of nanotron GenerationArgs schema used by decode_tokenized."""

    sampler: Optional[Any] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    n_samples: Optional[int] = None
    eos: Optional[str] = None
    seed: Optional[int] = None
    use_cache: Optional[bool] = False


class _NanotronRunner:
    """Lightweight port of lighteval Nanotron model for single-sample inference."""

    def __init__(self, config: NanotronBackendConfig) -> None:
        self.config = config
        self._torch = self._import_and_check()
        self._init_nanotron_modules()
        self._load_config_objects()
        self._build_parallel_context()
        self._build_model()
        self._build_tokenizer()
        self._max_length = self._resolve_max_length()
        self._generation_defaults = self._build_generation_args(config.generation_parameters)
        self.batch_size = config.batch_size

    # ------------------------------------------------------------------ #
    # Initialisation                                                     #
    # ------------------------------------------------------------------ #
    def _import_and_check(self):
        try:  # pragma: no cover - heavy dependency
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Nanotron backend requires torch to be installed") from exc
        return torch

    def _init_nanotron_modules(self) -> None:
        try:  # pragma: no cover - heavy dependency
            from nanotron import distributed as dist
            from nanotron import logging as nano_logging
            from nanotron.config import GeneralArgs, ModelArgs, TokenizerArgs, get_config_from_dict
            from nanotron.config.parallelism_config import ParallelismArgs
            from nanotron.generation.decode import decode_tokenized
            from nanotron.generation.sampler import SamplerType
            from nanotron.logging import human_format, log_rank
            from nanotron.models import build_model
            from nanotron.parallel.context import ParallelContext
            from nanotron.parallel.parameters import sanity_check
            from nanotron.parallel.pipeline_parallel.block import get_min_max_rank
            from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
            from nanotron.random import RandomStates, get_current_random_state, get_synced_random_state, set_random_seed
            from nanotron.serialize import load_weights
            from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("nanotron package is required for this backend") from exc

        self._dist = dist
        self._nanotron_logging = nano_logging
        self._ModelArgs = ModelArgs
        self._TokenizerArgs = TokenizerArgs
        self._GeneralArgs = GeneralArgs
        self._ParallelismArgs = ParallelismArgs
        self._get_config_from_dict = get_config_from_dict
        self._decode_tokenized = decode_tokenized
        self._SamplerType = SamplerType
        self._log_rank = log_rank
        self._human_format = human_format
        self._build_model = build_model
        self._ParallelContext = ParallelContext
        self._sanity_check = sanity_check
        self._get_min_max_rank = get_min_max_rank
        self._TensorParallelLinearMode = TensorParallelLinearMode
        self._RandomStates = RandomStates
        self._get_current_random_state = get_current_random_state
        self._get_synced_random_state = get_synced_random_state
        self._set_random_seed = set_random_seed
        self._load_weights = load_weights
        self._CONFIG_TO_MODEL_CLASS = CONFIG_TO_MODEL_CLASS
        self._mark_tied_parameters = mark_tied_parameters

    def _load_config_objects(self) -> None:
        with open(self.config.checkpoint_config_path, "r", encoding="utf-8") as handle:
            yaml_cfg = yaml.safe_load(handle)

        model_cfg = yaml_cfg.get("model") or {}
        tokenizer_cfg = yaml_cfg.get("tokenizer") or {}
        general_cfg = yaml_cfg.get("general") or {}
        self.model_args = self._get_config_from_dict(
            model_cfg,
            self._ModelArgs,
            skip_unused_config_keys=True,
            skip_null_keys=True,
        )
        self.tokenizer_args = self._get_config_from_dict(
            tokenizer_cfg,
            self._TokenizerArgs,
            skip_unused_config_keys=True,
            skip_null_keys=True,
        )
        self.general_args = self._get_config_from_dict(
            general_cfg,
            self._GeneralArgs,
            skip_unused_config_keys=True,
            skip_null_keys=True,
        )
        self.parallel_config = self._ParallelismArgs(**self.config.parallelism.dict())

    def _build_parallel_context(self) -> None:
        self._dist.initialize_torch_distributed()
        self.parallel_context = self._ParallelContext(
            tensor_parallel_size=self.parallel_config.tp,
            pipeline_parallel_size=self.parallel_config.pp,
            data_parallel_size=self.parallel_config.dp,
        )
        self._nanotron_logging.log_rank(
            "Nanotron parallel context initialised",
            logger=logger,
            level="INFO",
            group=self.parallel_context.dp_pg,
            rank=0,
        )

    def _build_model(self) -> None:
        torch = self._torch
        cfg = self.config
        model_config = self.model_args.model_config
        if cfg.debug_one_layer_model:
            try:
                model_config.num_hidden_layers = 1
            except AttributeError:
                pass

        dtype = self._resolve_dtype(cfg.dtype)
        self._set_random_seed(42)

        if (
            self.parallel_config.tp_mode == self._TensorParallelLinearMode.ALL_REDUCE
            and self.parallel_context.tp_pg is not None
        ):
            random_states = self._RandomStates(
                {
                    "tp_synced": self._get_synced_random_state(
                        random_state=self._get_current_random_state(),
                        pg=self.parallel_context.tp_pg,
                    )
                }
            )
        else:
            random_states = self._RandomStates({})

        model_config_cls = model_config.__class__.__name__
        if model_config_cls not in self._CONFIG_TO_MODEL_CLASS:
            raise ValueError(
                f"Unsupported Nanotron model config {model_config_cls}. "
                f"Available: {list(self._CONFIG_TO_MODEL_CLASS.keys())}"
            )

        logger.info("Building Nanotron model {}", model_config_cls)
        model = self._build_model(
            model_builder=lambda: self._CONFIG_TO_MODEL_CLASS[model_config_cls](
                config=model_config,
                parallel_context=self.parallel_context,
                parallel_config=self.parallel_config,
                random_states=random_states,
            ),
            dtype=dtype,
            parallel_context=self.parallel_context,
        )
        self._mark_tied_parameters(model=model, parallel_context=self.parallel_context, parallel_config=self.parallel_config)
        self._sanity_check(root_module=model)

        checkpoint_dir = cfg.checkpoint_path or os.path.dirname(cfg.checkpoint_config_path)
        if not checkpoint_dir:
            raise ValueError("Unable to resolve Nanotron checkpoint directory")

        logger.info("Loading Nanotron checkpoint from {}", checkpoint_dir)
        self._load_weights(model=model, parallel_context=self.parallel_context, root_folder=_safe_xpath(checkpoint_dir))
        model.eval()

        if model_config_cls == "FalconConfig":
            self.model = model.transformer
        else:
            self.model = getattr(model, "model", model)
        self.input_pp_rank, self.output_pp_rank = self._get_min_max_rank(module=self.model)
        self._dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        tokenizer_path = self.tokenizer_args.tokenizer_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

    def _resolve_dtype(self, dtype: Optional[str]):
        torch = self._torch
        if dtype is None:
            return torch.bfloat16
        if isinstance(dtype, str) and hasattr(torch, dtype):
            return getattr(torch, dtype)
        raise ValueError(f"Unsupported dtype for Nanotron backend: {dtype}")

    def _resolve_max_length(self) -> int:
        cfg = self.config
        if cfg.max_length:
            return cfg.max_length
        model_config = self.model_args.model_config
        for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
            if hasattr(model_config, attr):
                return getattr(model_config, attr)
        tokenizer_max = getattr(self.tokenizer, "model_max_length", None)
        if tokenizer_max and tokenizer_max != int(1e30):
            return tokenizer_max
        return 2048

    def _build_generation_args(self, params) -> _GenerationArgs:
        return _GenerationArgs(
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            n_samples=None,
            eos=None,
            seed=None,
            use_cache=True,
        )

    # ------------------------------------------------------------------ #
    # Inference                                                         #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        *,
        prompt: str,
        stop_sequences: Optional[List[str]],
        sampling_params: Dict[str, Any],
    ) -> tuple[str, List[int]]:
        torch = self._torch
        tokenizer_inputs = self.tokenizer(
            [prompt],
            truncation="longest_first",
            padding="longest",
            return_tensors="pt",
            max_length=max(1, self._max_length - 1),
            add_special_tokens=self.config.add_special_tokens,
        )
        tokenizer_inputs = {k: v.to(self.device) for k, v in tokenizer_inputs.items()}
        context_size = tokenizer_inputs["input_ids"].shape[1]
        requested_max_new = (
            sampling_params.get("max_new_tokens")
            or sampling_params.get("max_tokens")
            or self.config.max_new_tokens
            or self.config.generation_parameters.max_new_tokens
            or 256
        )
        max_new_tokens = min(max(1, self._max_length - context_size), requested_max_new)

        generation_config = self._override_generation_args(self._generation_defaults, sampling_params)

        outputs = self._decode_tokenized(
            input_ids=tokenizer_inputs["input_ids"],
            input_mask=tokenizer_inputs["attention_mask"],
            model=self.model,
            parallel_context=self.parallel_context,
            max_new_tokens=max_new_tokens,
            max_micro_batch_size=min(self.batch_size, 1),
            returns_logits=False,
            tokenizer=self.tokenizer,
            generation_config=generation_config,
        )
        self._dist.barrier()
        outputs = list(outputs)
        if not outputs:
            raise RuntimeError("Nanotron decode returned empty output")
        output = outputs[0]
        generation = output.generation_ids[output.input_ids.shape[0] :].detach().cpu()
        token_ids = generation.tolist()
        decoded = self.tokenizer.decode(generation, skip_special_tokens=False)
        for token in stop_sequences or []:
            if token and token in decoded:
                decoded = decoded.split(token)[0]
        decoded = decoded.strip()
        return decoded, token_ids

    def _override_generation_args(
        self,
        base: _GenerationArgs,
        overrides: Dict[str, Any],
    ) -> _GenerationArgs:
        updated = _GenerationArgs(
            sampler=base.sampler,
            temperature=overrides.get("temperature", base.temperature),
            top_k=overrides.get("top_k", base.top_k),
            top_p=overrides.get("top_p", base.top_p),
            n_samples=overrides.get("n") or overrides.get("num_samples") or base.n_samples,
            eos=base.eos,
            seed=overrides.get("seed", base.seed),
            use_cache=True,
        )
        if updated.n_samples not in (None, 1):
            logger.warning("Nanotron backend currently only supports num_samples=1 (got {})", updated.n_samples)
            updated.n_samples = 1
        return updated


def _safe_xpath(path: str):
    """Resolve datasets' streaming manager helper if available."""

    try:
        from datasets.download.streaming_download_manager import xPath
    except Exception:  # pragma: no cover - optional dependency
        return path
    return xPath(path)
