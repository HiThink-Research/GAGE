"""Configuration schemas for HuggingFace inference backends."""

from __future__ import annotations

import re
from typing import Dict, Optional

from pydantic import Field, root_validator

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class HFServerlessBackendConfig(BackendConfigBase):
    """Config for HuggingFace InferenceClient (serverless)."""

    model_name: str = Field(description="HuggingFace Hub 模型 ID，如 meta-llama/Llama-3.1-8B-Instruct")
    timeout: int = Field(default=60, ge=1, description="HTTP 超时秒数")
    max_retries: int = Field(default=3, ge=0, description="失败后的最大重试次数")
    wait_for_model: bool = Field(default=True, description="调用时是否等待模型冷启动")
    extra_headers: Dict[str, str] = Field(default_factory=dict, description="附加 HTTP Header")
    generation_parameters: GenerationParameters = Field(
        default_factory=GenerationParameters, description="默认采样参数"
    )


class HFInferenceEndpointBackendConfig(BackendConfigBase):
    """Config for dedicated HuggingFace Inference Endpoints (TGI-based)."""

    endpoint_name: Optional[str] = Field(
        default=None, description="Endpoint 名称；为空则根据模型名自动生成"
    )
    namespace: Optional[str] = Field(default=None, description="Endpoint 所属 namespace/organization")
    model_name: Optional[str] = Field(default=None, description="需要部署的模型 ID（endpoint 不存在时创建所需）")
    revision: str = Field(default="main", description="模型 revision")
    reuse_existing: bool = Field(default=True, description="若 endpoint 已存在则直接复用")
    auto_start: bool = Field(default=True, description="若 endpoint 不存在，是否自动创建")
    delete_on_exit: bool = Field(default=False, description="Backend 关闭时删除自动创建的 endpoint")
    accelerator: str = Field(default="gpu", description="推理加速器类型")
    vendor: str = Field(default="aws", description="云厂商，例如 aws/azure/gcp")
    region: str = Field(default="us-east-1", description="部署区域")
    instance_type: Optional[str] = Field(default=None, description="GPU 类型，例如 nvidia-a10g")
    instance_size: Optional[str] = Field(default=None, description="实例规模，例如 x1/x4")
    endpoint_type: str = Field(default="protected", description="Endpoint 类型（protected/public）")
    framework: str = Field(default="pytorch", description="底层框架")
    dtype: Optional[str] = Field(default=None, description="推理精度（float16/bfloat16/awq 等）")
    image_url: Optional[str] = Field(default=None, description="自定义 TGI 镜像地址")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="附加环境变量")
    wait_timeout: int = Field(default=1800, ge=60, description="等待 endpoint 就绪的最长秒数")
    poll_interval: int = Field(default=60, ge=5, description="轮询间隔秒")
    huggingface_token: Optional[str] = Field(default=None, description="显式 HuggingFace token（否则读取环境变量）")
    generation_parameters: GenerationParameters = Field(
        default_factory=GenerationParameters, description="默认采样参数"
    )

    @root_validator(skip_on_failure=True)
    def _ensure_name_or_model(cls, values):
        endpoint_name = values.get("endpoint_name")
        model_name = values.get("model_name")
        auto = values.get("auto_start")
        if not endpoint_name and not model_name:
            raise ValueError("HFInferenceEndpointBackendConfig 需要 endpoint_name 或 model_name 至少一个")
        if not endpoint_name and not auto:
            raise ValueError("未提供 endpoint_name 时必须允许 auto_start 以创建实例")
        return values

    def normalized_endpoint_name(self) -> str:
        if self.endpoint_name:
            return self.endpoint_name
        assert self.model_name, "Auto name requires model_name"
        return re.sub(r"[^a-zA-Z0-9-]", "-", f"{self.model_name}-gage").lower()
