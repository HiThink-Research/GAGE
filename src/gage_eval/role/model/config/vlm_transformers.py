"""Schema describing the HuggingFace VLM transformers backend."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field, validator

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class VLMTransformersBackendConfig(BackendConfigBase):
    """Configuration payload consumed by VLMTransformersBackend."""

    model_name_or_path: Optional[str] = Field(
        default=None,
        description="HuggingFace 仓库 ID 或本地权重目录（允许被 model_id 覆盖）",
    )
    model_path: Optional[str] = Field(default=None, description="由模型资产注入的本地权重路径")
    processor_name_or_path: Optional[str] = Field(default=None, description="可自定义的 processor 资源")
    revision: str = Field(default="main", description="模型 revision")
    subfolder: Optional[str] = Field(default=None, description="模型子目录")
    cache_dir: Optional[str] = Field(default=None, description="权重/processor 缓存目录")
    use_fast_image_processor: Optional[bool] = Field(default=None, description="是否优先使用 fast image processor")
    batch_size: int = Field(default=1, ge=1, description="保留字段，兼容未来批量请求")
    max_length: Optional[int] = Field(default=None, ge=1, description="文本 + 生成 token 的最大长度")
    generation_size: Optional[int] = Field(default=None, ge=1, description="每次生成的最大 token 数，如未提供则使用 generation_parameters")
    add_special_tokens: bool = Field(default=True, description="在 processor.encode 时是否补充 special token")
    model_parallel: Optional[bool] = Field(
        default=None, description="是否启用 accelerate model parallel（None 表示自动探测）"
    )
    dtype: Optional[str] = Field(default=None, description="torch dtype / 4bit / 8bit / auto")
    max_pixels: Optional[int] = Field(
        default=None, ge=1, description="图像缩放后的像素上限（不填则使用框架默认值）"
    )
    min_pixels: Optional[int] = Field(
        default=None, ge=1, description="图像缩放后的最小像素（不填则使用框架默认值）"
    )
    image_factor: Optional[int] = Field(
        default=None, ge=1, description="图像缩放因子（默认 28，对齐 Qwen 系列图像尺寸约束）"
    )
    attn_implementation: Optional[str] = Field(
        default=None, description="注意力实现，例如 flash_attention_2（透传给 from_pretrained）"
    )
    device: Optional[str] = Field(default=None, description="显式指定推理设备（覆盖自动探测）")
    device_map: Optional[str] = Field(default=None, description="显式 device map 配置，优先级最高")
    trust_remote_code: bool = Field(default=True, description="加载模型/processor 时是否 trust_remote_code")
    compile: bool = Field(default=False, description="预留字段，暂不支持编译 VLM")
    distributed_timeout: int = Field(default=3000, ge=60, description="多机初始化超时时间（秒）")
    generation_parameters: GenerationParameters = Field(
        default_factory=GenerationParameters, description="采样参数默认值"
    )
    system_prompt: Optional[str] = Field(default=None, description="保留字段，交由 Prompt 渲染层处理")
    extra_processor_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="覆盖 AutoProcessor.from_pretrained 的自定义 keyword arguments"
    )

    @validator("model_name_or_path", always=True)
    def validate_model_source(cls, value, values):
        if (value is None or value == "") and not values.get("model_path"):
            raise ValueError("VLMTransformersBackend requires 'model_name_or_path' or injected 'model_path'")
        return value
