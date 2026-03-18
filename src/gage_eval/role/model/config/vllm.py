"""Schema for vLLM backend configuration."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, PositiveInt

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class VLLMBackendConfig(BackendConfigBase):
    """Configuration payload consumed by the vLLM backend."""

    model_path: str = Field(description="Filesystem path or HuggingFace repo id")
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: Optional[int] = Field(default=None, ge=1)
    data_parallel_rank: Optional[int] = Field(default=None, ge=0)
    data_parallel_size_local: Optional[int] = Field(default=None, ge=1)
    data_parallel_address: Optional[str] = None
    data_parallel_rpc_port: Optional[int] = Field(default=None, ge=1)
    data_parallel_backend: Optional[str] = None
    distributed_executor_backend: Optional[str] = None
    request_timeout: float = Field(default=300.0, gt=0.0)
    max_model_len: Optional[int] = Field(default=None, ge=128)
    dtype: Optional[str] = None
    trust_remote_code: bool = True
    output_type: str = Field(default="text", description="Controls output adapter behavior")
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
    limit_mm_per_prompt: Optional[dict[str, PositiveInt]] = Field(
        default=None,
        description="Explicit per-prompt multimodal limits forwarded to vLLM, for example {'image': 1, 'audio': 1}",
    )
    use_chat_template: Literal["auto", "never"] = Field(
        default="auto",
        description="文本任务是否使用 chat_template：auto 默认预处理渲染，never 直传 plain",
    )
