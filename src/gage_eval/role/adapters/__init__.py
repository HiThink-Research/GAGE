from __future__ import annotations

import importlib


_ADAPTER_EXPORTS = {
    "ModelRoleAdapter": "gage_eval.role.adapters.model_role_adapter:ModelRoleAdapter",
    "DUTModelAdapter": "gage_eval.role.adapters.dut_model:DUTModelAdapter",
    "DUTAgentAdapter": "gage_eval.role.adapters.dut_agent:DUTAgentAdapter",
    "JudgeModelAdapter": "gage_eval.role.adapters.judge_model:JudgeModelAdapter",
    "JudgeExtendAdapter": "gage_eval.role.adapters.judge_extend:JudgeExtendAdapter",
    "HelperModelAdapter": "gage_eval.role.adapters.helper_model:HelperModelAdapter",
    "ContextProviderAdapter": "gage_eval.role.adapters.context_provider:ContextProviderAdapter",
    "ToolchainAdapter": "gage_eval.role.toolchain:ToolchainAdapter",
    "ModalProcessorAdapter": "gage_eval.role.adapters.modal_processor:ModalProcessorAdapter",
    "ArenaRoleAdapter": "gage_eval.role.adapters.arena:ArenaRoleAdapter",
    "HumanAdapter": "gage_eval.role.adapters.human:HumanAdapter",
}

__all__ = list(_ADAPTER_EXPORTS)


def __getattr__(name: str):
    target = _ADAPTER_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, symbol = target.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, symbol)
