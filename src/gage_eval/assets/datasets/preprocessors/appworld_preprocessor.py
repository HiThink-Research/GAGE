"""AppWorld JSONL preprocessor for standardized Sample envelopes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor

_TEST_SUBSETS = {"test_normal", "test_challenge"}
_GROUND_TRUTH_MODES = {"full", "partial", "minimal"}
_DEFAULT_HELPER_APPS = ("api_docs", "supervisor")


class AppWorldPreprocessor(BasePreprocessor):
    """Normalize AppWorld JSONL records into the Sample schema."""

    name = "appworld_preprocessor"

    def __init__(
        self,
        *,
        subset: Optional[str] = None,
        ground_truth_mode: Optional[str] = "minimal",
        force_minimal_on_test: bool = True,
        experiment_name: Optional[str] = None,
        data_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._default_subset = subset
        self._default_ground_truth_mode = ground_truth_mode
        self._force_minimal_on_test = bool(force_minimal_on_test)
        if experiment_name is None or not str(experiment_name).strip():
            experiment_name = _build_default_experiment_name()
        self._default_experiment_name = experiment_name
        self._data_path = data_path

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError("AppWorld preprocessor expects dict records")
        appworld_meta = _resolve_appworld_meta(record)
        subset = _resolve_subset(self._default_subset, appworld_meta, record)
        task_id = _resolve_task_id(record, appworld_meta)
        instruction = _resolve_instruction(record)
        messages = _normalize_messages(record.get("messages"), instruction)
        allowed_apps = _resolve_allowed_apps(record, appworld_meta)
        experiment_name = _resolve_experiment_name(
            appworld_meta,
            record,
            kwargs.get("experiment_name"),
            self._default_experiment_name,
        )
        ground_truth_mode = _resolve_ground_truth_mode(
            self._default_ground_truth_mode,
            appworld_meta,
            subset=subset,
            force_minimal=self._force_minimal_on_test,
        )
        sample: Dict[str, Any] = {
            "id": task_id,
            "task_type": "agent",
            "messages": messages,
            "metadata": {"appworld": {"task_id": task_id}},
        }
        if subset:
            sample["metadata"]["appworld"]["subset"] = subset
        if allowed_apps:
            sample["metadata"]["appworld"]["allowed_apps"] = allowed_apps
        if experiment_name:
            sample["metadata"]["appworld"]["experiment_name"] = experiment_name
        if ground_truth_mode:
            sample["metadata"]["appworld"]["ground_truth_mode"] = ground_truth_mode
        if "tool_choice" in record:
            sample["tool_choice"] = record.get("tool_choice")
        return sample


def _resolve_appworld_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    appworld_meta = meta.get("appworld") if isinstance(meta.get("appworld"), dict) else {}
    return dict(appworld_meta)


def _resolve_subset(default_subset: Optional[str], meta: Dict[str, Any], record: Dict[str, Any]) -> Optional[str]:
    subset = meta.get("subset") or record.get("subset") or default_subset
    if subset:
        return str(subset)
    return None


def _resolve_task_id(record: Dict[str, Any], meta: Dict[str, Any]) -> str:
    task_id = record.get("task_id") or record.get("id") or meta.get("task_id")
    if not task_id:
        raise ValueError("AppWorld record missing task_id")
    return str(task_id)


def _resolve_instruction(record: Dict[str, Any]) -> str:
    instruction = record.get("instruction")
    if instruction:
        return str(instruction)
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0] if isinstance(messages[0], dict) else None
        if first:
            content = first.get("content")
            if isinstance(content, list) and content:
                text = content[0].get("text")
                if text:
                    return str(text)
            if isinstance(content, str):
                return content
    return ""


def _normalize_messages(raw_messages: Any, instruction: str) -> List[Dict[str, Any]]:
    if isinstance(raw_messages, list):
        normalized = [msg for msg in raw_messages if isinstance(msg, dict)]
        if normalized:
            return normalized
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": instruction}],
        }
    ]


def _resolve_allowed_apps(record: Dict[str, Any], meta: Dict[str, Any]) -> Optional[List[str]]:
    allowed = meta.get("allowed_apps") or record.get("allowed_apps")
    if isinstance(allowed, list):
        merged = [str(item) for item in allowed if item]
        for helper in _DEFAULT_HELPER_APPS:
            if helper not in merged:
                merged.append(helper)
        return merged
    return None


def _resolve_experiment_name(
    meta: Dict[str, Any],
    record: Dict[str, Any],
    override: Optional[str],
    default: Optional[str],
) -> Optional[str]:
    value = meta.get("experiment_name") or record.get("experiment_name") or override or default
    if not value:
        return None
    return str(value)


def _resolve_ground_truth_mode(
    default_mode: Optional[str],
    meta: Dict[str, Any],
    *,
    subset: Optional[str],
    force_minimal: bool,
) -> Optional[str]:
    mode = meta.get("ground_truth_mode") or default_mode
    if subset in _TEST_SUBSETS and force_minimal:
        mode = "minimal"
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if normalized not in _GROUND_TRUTH_MODES:
        return None
    return normalized


def _build_default_experiment_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"appworld-experiment-{stamp}"


__all__ = ["AppWorldPreprocessor"]
