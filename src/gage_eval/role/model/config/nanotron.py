"""Schema for Nanotron distributed backend configuration."""

from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class NanotronParallelismConfig(BackendConfigBase):
    """Mirror of Nanotron ParallelismArgs fields we rely on."""

    dp: int = Field(default=1, ge=1, description="Data parallel world size")
    pp: int = Field(default=1, ge=1, description="Pipeline parallel stages")
    tp: int = Field(default=1, ge=1, description="Tensor parallel world size")
    tp_mode: str = Field(default="ALL_REDUCE", description="Tensor parallel linear mode")
    pp_engine: str = Field(default="1f1b", description="Pipeline scheduler")
    tp_linear_async_communication: bool = Field(default=False, description="Enable async TP communication")


class NanotronBackendConfig(BackendConfigBase):
    """Pydantic schema consumed by the Nanotron backend."""

    checkpoint_config_path: str = Field(
        description="Path to Nanotron YAML config (containing model/tokenizer/general entries)"
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Directory storing Nanotron checkpoint shards; defaults to config file parent",
    )
    parallelism: NanotronParallelismConfig = Field(
        default_factory=NanotronParallelismConfig,
        description="Distributed topology injected into ParallelContext",
    )
    batch_size: int = Field(default=1, ge=1, description="Maximum micro batch size per decode call")
    max_new_tokens: Optional[int] = Field(default=None, ge=1, description="Upper bound for generation tokens")
    max_length: Optional[int] = Field(default=None, ge=1, description="Override for model max input length")
    add_special_tokens: bool = Field(default=False, description="Whether tokenizer should add special tokens")
    dtype: Optional[str] = Field(default=None, description="torch dtype string (float16/bfloat16/etc.)")
    trust_remote_code: bool = Field(default=False, description="Forwarded to AutoTokenizer")
    debug_one_layer_model: bool = Field(default=False, description="Force single layer (Nanotron debug)")
    generation_parameters: GenerationParameters = Field(
        default_factory=GenerationParameters,
        description="Default sampling parameters applied to decode calls",
    )
    multichoice_continuations_start_space: Optional[bool] = Field(
        default=None,
        description="Match lighteval switch controlling whitespace before MC answers",
    )
    pairwise_tokenization: bool = Field(
        default=False,
        description="Reuse tokenizer pairwise splitting logic from lighteval (not used in backend yet)",
    )

    @field_validator("checkpoint_config_path", mode="after")
    def _validate_paths(cls, value: str) -> str:
        if not value:
            raise ValueError("Nanotron backend requires checkpoint_config_path")
        return value
