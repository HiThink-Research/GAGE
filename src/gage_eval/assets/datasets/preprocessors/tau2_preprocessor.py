"""Preprocessor for Tau2 task records."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.sample import Message, MessageContent, Sample, SCHEMA_VERSION


class Tau2Preprocessor(BasePreprocessor):
    """Normalize Tau2 task records into the Sample schema."""

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
        # STEP 1: Validate and materialize the tau2 Task object.
        task_dict = _extract_task_dict(record)
        if not task_dict:
            return None
        task = _build_task(task_dict)

        # STEP 2: Collect tau2 metadata for downstream runtime/judge.
        domain = record.get("_tau2_domain") or record.get("domain")
        task_set = record.get("_tau2_task_set") or record.get("task_set") or record.get("task_set_name")
        task_split = record.get("_tau2_split") or record.get("task_split") or record.get("split")
        trial = record.get("trial")
        seed = record.get("seed")

        tau2_meta: Dict[str, Any] = {
            "task_id": task.id,
            "domain": domain,
            "task_set": task_set,
            "task_split": task_split,
            "trial": trial,
            "seed": seed,
            "user_scenario": _safe_model_dump(task.user_scenario),
            "reward_basis": _extract_reward_basis(task),
            "has_initial_state": task.initial_state is not None,
        }
        if record.get("annotations") is not None:
            tau2_meta["annotations"] = record.get("annotations")

        # STEP 3: Build raw_assets payload for judge evaluation.
        raw_assets = {
            "tau2": {
                "task": _safe_model_dump(task),
                "evaluation_criteria": _safe_model_dump(task.evaluation_criteria),
                "initial_state": _safe_model_dump(task.initial_state),
            }
        }

        sample_id = _build_sample_id(task.id, domain=domain, trial=trial)
        placeholder = Message(role="system", content=[MessageContent(type="text", text="")])
        return Sample(
            schema_version=SCHEMA_VERSION,
            id=sample_id,
            messages=[placeholder],
            metadata={"tau2": tau2_meta},
            raw_assets=raw_assets,
            tools=[],
        )


def _extract_task_dict(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    if isinstance(record.get("task"), dict):
        return dict(record["task"])
    return dict(record)


def _build_task(task_dict: Dict[str, Any]):
    try:
        from tau2.data_model.tasks import Task  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "tau2 is required for tau2_preprocessor. Install tau2-bench or set PYTHONPATH accordingly."
        ) from exc
    return Task.model_validate(task_dict)


def _safe_model_dump(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    return value if isinstance(value, dict) else {"value": value}


def _extract_reward_basis(task: Any) -> Optional[list[str]]:
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria is None:
        return None
    basis = getattr(criteria, "reward_basis", None)
    if basis is None:
        return None
    return [str(item) for item in basis]


def _build_sample_id(task_id: str, *, domain: Optional[str], trial: Optional[int]) -> str:
    base = f"{domain}_{task_id}" if domain else str(task_id)
    if trial is None:
        return base
    return f"{base}__trial_{trial}"
