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
    "context_impls": "上下文提供实现（Repo/检索等）",
    "judge_impls": "裁判扩展实现（Docker/脚本等）",
    "arena_impls": "对局环境实现（游戏规则/状态机）",
    "parser_impls": "对局解析器实现（动作解析与坐标校验）",
    "renderer_impls": "对局渲染器实现（棋盘/UI 渲染）",
    "dataset_hubs": "数据集源（HF、ModelScope、本地等）",
    "dataset_loaders": "数据集加载器",
    "bundles": "Resource Provider",
    "dataset_preprocessors": "数据集预处理器",
    "doc_converters": "Doc -> 样本转换器",
    "prompts": "Prompt 模版资产",
    "model_hubs": "模型资产下载/缓存源",
    "templates": "配置模版资产",
    "reporting_sinks": "报告下沉（文件/HTTP 等）",
    "summary_generators": "报告摘要生成器（业务/平台适配）",
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
