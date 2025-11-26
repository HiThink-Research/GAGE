"""Global registry singleton used across gage-eval."""

from __future__ import annotations

from typing import Dict

from gage_eval.registry.entry import RegistryEntry
from gage_eval.registry.manager import RegistryManager
from gage_eval.registry.utils import ensure_async, run_sync

registry = RegistryManager()

DEFAULT_KINDS: Dict[str, str] = {
    "backends": "模型推理与外部服务后端",
    "roles": "角色适配器（DUT/Judge/Helper 等）",
    "metrics": "指标计算器",
    "dataset_hubs": "数据集源（HF、ModelScope、本地等）",
    "dataset_loaders": "数据集加载器",
    "dataset_preprocessors": "数据集预处理器",
    "doc_converters": "Doc -> 样本转换器",
    "prompts": "Prompt 模版资产",
    "model_hubs": "模型资产下载/缓存源",
    "templates": "配置模版资产",
    "reporting_sinks": "报告下沉（文件/HTTP 等）",
    "observability_plugins": "观测插件",
    "pipeline_steps": "Pipeline 执行步骤",
}

for _kind, _desc in DEFAULT_KINDS.items():
    registry.declare_kind(_kind, desc=_desc)

__all__ = [
    "RegistryEntry",
    "RegistryManager",
    "registry",
    "ensure_async",
    "run_sync",
]
