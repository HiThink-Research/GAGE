"""Preprocessor that removes ground-truth assistant turns from chat-style samples."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Sequence


def convert_sample_to_inputs(
    sample: Dict[str, Any],
    *,
    roles_to_remove: Sequence[str] = ("assistant",),
    field: str = "messages",
) -> List[Dict]:
    """Drop assistant turns (often containing reference answers) from ``sample[field]``.

    MMMU 等数据集常把标准答案作为最后一条 ``assistant`` 消息下发，
    如果不清理，真实模型会把这条消息视为历史对话直接复述/不输出内容。
    该预处理在加载阶段剥离指定角色，保证推理端只看到 system/user 指令。
    """

    messages = sample.get(field)
    if not isinstance(messages, list):
        return sample.get("inputs") or []

    normalized_roles = {role.lower() for role in roles_to_remove if isinstance(role, str)}
    filtered: List[Dict] = []
    removed = False

    for message in messages:
        role = str(message.get("role", "")).lower()
        if normalized_roles and role in normalized_roles:
            removed = True
            continue
        filtered.append(deepcopy(message))

    if removed:
        sample[field] = filtered
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        existing = metadata.get("removed_roles") or []
        if not isinstance(existing, list):
            existing = [existing]
        metadata["removed_roles"] = sorted({str(role) for role in existing if role} | normalized_roles)
        sample["metadata"] = metadata

    return filtered
