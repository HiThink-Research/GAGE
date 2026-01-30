"""Tau2 judge implementation using tau2 evaluators."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from gage_eval.registry import registry
from gage_eval.role.judge.base import JudgeImplementation
from gage_eval.sandbox.provider import SandboxProvider


@registry.asset(
    "judge_impls",
    "tau2_eval",
    desc="Tau2 evaluation using tau2.evaluator.evaluate_simulation",
    tags=("tau2", "judge"),
)
class Tau2Evaluate(JudgeImplementation):
    """Evaluate tau2 trajectories using the official evaluator."""

    def invoke(self, payload: Dict[str, Any], state: Any = None) -> Dict[str, Any]:
        sample = payload.get("sample") or {}
        runtime_state = _resolve_runtime_state(payload)
        task = _build_tau2_task(sample)
        domain = _resolve_domain(sample, runtime_state)
        simulation = _build_simulation(task, runtime_state)
        evaluation_type = _resolve_evaluation_type(task)

        reward_info = _evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=evaluation_type,
            domain=domain,
        )
        tau2_payload = _build_output_payload(simulation, reward_info, domain, evaluation_type)
        return {"tau2": tau2_payload}


def _resolve_runtime_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    sandbox_provider = payload.get("sandbox_provider")
    if isinstance(sandbox_provider, SandboxProvider):
        handle = sandbox_provider.get_handle()
        runtime = handle.sandbox if handle else None
        getter = getattr(runtime, "get_state", None)
        if callable(getter):
            return getter() or {}
    return {}


def _build_tau2_task(sample: Dict[str, Any]) -> Any:
    raw_assets = sample.get("raw_assets") if isinstance(sample.get("raw_assets"), dict) else {}
    tau2_payload = raw_assets.get("tau2") if isinstance(raw_assets.get("tau2"), dict) else {}
    task_payload = tau2_payload.get("task") or sample.get("task") or sample
    try:
        from tau2.data_model.tasks import Task  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 is required to evaluate tau2 tasks") from exc
    return Task.model_validate(task_payload)


def _resolve_domain(sample: Dict[str, Any], runtime_state: Dict[str, Any]) -> str:
    if runtime_state.get("domain"):
        return str(runtime_state["domain"])
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    tau2_meta = meta.get("tau2") if isinstance(meta.get("tau2"), dict) else {}
    return str(tau2_meta.get("domain") or "airline")


def _build_simulation(task: Any, runtime_state: Dict[str, Any]) -> Any:
    try:
        from tau2.data_model.simulation import SimulationRun, TerminationReason  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 SimulationRun unavailable") from exc

    start_time = runtime_state.get("start_time") or _now()
    end_time = _now()
    duration = _duration_seconds(start_time, end_time)
    termination = runtime_state.get("termination_reason") or TerminationReason.AGENT_ERROR
    return SimulationRun(
        id=str(runtime_state.get("simulation_id") or runtime_state.get("task_id") or task.id),
        task_id=str(task.id),
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        termination_reason=termination,
        agent_cost=runtime_state.get("agent_cost"),
        user_cost=runtime_state.get("user_cost"),
        reward_info=None,
        messages=runtime_state.get("messages") or [],
        trial=runtime_state.get("trial"),
        seed=runtime_state.get("seed"),
    )


def _resolve_evaluation_type(task: Any):
    try:
        from tau2.evaluator.evaluator import EvaluationType  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 EvaluationType unavailable") from exc
    criteria = getattr(task, "evaluation_criteria", None)
    basis = getattr(criteria, "reward_basis", None) if criteria else None
    if basis:
        for item in basis:
            if str(item).upper().endswith("NL_ASSERTION"):
                return EvaluationType.ALL_WITH_NL_ASSERTIONS
    return EvaluationType.ALL


def _evaluate_simulation(*, simulation: Any, task: Any, evaluation_type: Any, domain: str) -> Any:
    try:
        from tau2.evaluator.evaluator import evaluate_simulation  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 evaluator unavailable") from exc
    return evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=evaluation_type,
        solo_mode=False,
        domain=domain,
    )


def _build_output_payload(simulation: Any, reward_info: Any, domain: str, evaluation_type: Any) -> Dict[str, Any]:
    reward_payload = _safe_model_dump(reward_info)
    return {
        "task_id": simulation.task_id,
        "domain": domain,
        "trial": simulation.trial,
        "seed": simulation.seed,
        "termination_reason": _stringify(getattr(simulation, "termination_reason", None)),
        "reward": reward_payload.get("reward"),
        "reward_basis": reward_payload.get("reward_basis"),
        "reward_breakdown": reward_payload.get("reward_breakdown"),
        "reward_info": reward_payload,
        "agent_cost": simulation.agent_cost,
        "user_cost": simulation.user_cost,
        "evaluation_type": _stringify(evaluation_type),
    }


def _safe_model_dump(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def _stringify(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        try:
            return str(value.value)
        except Exception:
            pass
    return str(value)


def _duration_seconds(start_time: str, end_time: str) -> float:
    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
        return (end - start).total_seconds()
    except Exception:
        return 0.0


def _now() -> str:
    return datetime.now().isoformat()
