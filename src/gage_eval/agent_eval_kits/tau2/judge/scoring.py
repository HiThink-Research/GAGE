"""Tau2 verifier scoring helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from gage_eval.agent_eval_kits.tau2._helpers import resolve_tau2_termination_reason


def evaluate_tau2_sample(
    *,
    sample: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a Tau2 sample against the captured runtime state."""

    task = build_tau2_task(sample)
    domain = resolve_domain(sample, runtime_state)
    simulation = build_simulation(task, runtime_state)
    evaluation_type = resolve_evaluation_type(task)
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=evaluation_type,
        domain=domain,
    )
    tau2_payload = build_output_payload(simulation, reward_info, domain, evaluation_type)
    return {
        "tau2": tau2_payload,
        "diagnostic_reason": str(
            tau2_payload.get("termination_reason") or "tau2_evaluation_completed"
        ),
        "diagnostic_details": {
            "tau2": {
                "reward": tau2_payload.get("reward"),
                "termination_reason": tau2_payload.get("termination_reason"),
                "reward_basis": tau2_payload.get("reward_basis"),
                "reward_info": tau2_payload.get("reward_info"),
            }
        },
    }


def build_tau2_task(sample: Mapping[str, Any]) -> Any:
    raw_assets = sample.get("raw_assets") if isinstance(sample.get("raw_assets"), Mapping) else {}
    tau2_payload = raw_assets.get("tau2") if isinstance(raw_assets.get("tau2"), Mapping) else {}
    task_payload = tau2_payload.get("task") or sample.get("task") or sample
    try:
        from tau2.data_model.tasks import Task  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 is required to evaluate tau2 tasks") from exc
    return Task.model_validate(task_payload)


def resolve_domain(sample: Mapping[str, Any], runtime_state: Mapping[str, Any]) -> str:
    if runtime_state.get("domain"):
        return str(runtime_state["domain"])
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
    tau2_meta = metadata.get("tau2") if isinstance(metadata.get("tau2"), Mapping) else {}
    return str(tau2_meta.get("domain") or "airline")


def build_simulation(task: Any, runtime_state: Mapping[str, Any]) -> Any:
    try:
        from tau2.data_model.simulation import SimulationRun  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 SimulationRun unavailable") from exc

    metadata = runtime_state.get("metadata") if isinstance(runtime_state.get("metadata"), Mapping) else {}
    start_time = runtime_state.get("start_time") or _now()
    end_time = runtime_state.get("end_time") or _now()
    return SimulationRun(
        id=str(runtime_state.get("simulation_id") or runtime_state.get("task_id") or task.id),
        task_id=str(task.id),
        start_time=start_time,
        end_time=end_time,
        duration=_duration_seconds(str(start_time), str(end_time)),
        termination_reason=resolve_tau2_termination_reason(
            runtime_state.get("termination_reason"),
            fallback="too_many_errors",
        ),
        agent_cost=runtime_state.get("agent_cost"),
        user_cost=runtime_state.get("user_cost"),
        reward_info=None,
        messages=runtime_state.get("messages") or [],
        trial=runtime_state.get("trial", metadata.get("trial")),
        seed=runtime_state.get("seed", metadata.get("seed")),
    )


def resolve_evaluation_type(task: Any) -> Any:
    try:
        from tau2.evaluator.evaluator import EvaluationType  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 EvaluationType unavailable") from exc
    criteria = getattr(task, "evaluation_criteria", None)
    basis = getattr(criteria, "reward_basis", None) if criteria else None
    basis_values = _reward_basis_values(basis)
    if not basis_values:
        return _evaluation_type(EvaluationType, "ALL")
    if "NL_ASSERTION" in basis_values:
        if len(basis_values) == 1:
            return _evaluation_type(
                EvaluationType,
                "NL_ASSERTIONS",
                fallback="ALL_WITH_NL_ASSERTIONS",
            )
        return _evaluation_type(EvaluationType, "ALL_WITH_NL_ASSERTIONS")
    if basis_values <= {"DB", "ENV_ASSERTION"}:
        return _evaluation_type(EvaluationType, "ENV")
    if basis_values == {"ACTION"}:
        return _evaluation_type(EvaluationType, "ACTION")
    if basis_values == {"COMMUNICATE"}:
        return _evaluation_type(EvaluationType, "COMMUNICATE")
    return _evaluation_type(EvaluationType, "ALL")


def evaluate_simulation(*, simulation: Any, task: Any, evaluation_type: Any, domain: str) -> Any:
    try:
        from tau2.evaluator.evaluator import evaluate_simulation as tau2_evaluate_simulation  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 evaluator unavailable") from exc
    return tau2_evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=evaluation_type,
        solo_mode=False,
        domain=domain,
    )


def build_output_payload(
    simulation: Any,
    reward_info: Any,
    domain: str,
    evaluation_type: Any,
) -> dict[str, Any]:
    reward_payload = safe_model_dump(reward_info)
    return {
        "task_id": simulation.task_id,
        "domain": domain,
        "trial": simulation.trial,
        "seed": simulation.seed,
        "termination_reason": stringify(getattr(simulation, "termination_reason", None)),
        "reward": reward_payload.get("reward"),
        "reward_basis": reward_payload.get("reward_basis"),
        "reward_breakdown": reward_payload.get("reward_breakdown"),
        "reward_info": reward_payload,
        "agent_cost": simulation.agent_cost,
        "user_cost": simulation.user_cost,
        "evaluation_type": stringify(evaluation_type),
    }


def tau2_metric(result: Mapping[str, Any]) -> dict[str, Any]:
    tau2_payload = result.get("tau2") if isinstance(result.get("tau2"), Mapping) else {}
    reward = tau2_payload.get("reward")
    try:
        score = float(reward)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "score": score,
        "resolved": score >= 1.0,
        "failure_reason": result.get("failure_reason"),
    }


def safe_model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def stringify(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        try:
            return str(value.value)
        except Exception:
            pass
    return str(value)


def _reward_basis_values(basis: Any) -> set[str]:
    if not basis:
        return set()
    values: set[str] = set()
    for item in basis:
        raw = getattr(item, "value", item)
        values.add(str(raw).upper())
    return values


def _evaluation_type(evaluation_type: Any, name: str, *, fallback: str = "ALL") -> Any:
    value = getattr(evaluation_type, name, None)
    if value is not None:
        return value
    fallback_value = getattr(evaluation_type, fallback, None)
    if fallback_value is not None:
        return fallback_value
    return evaluation_type.ALL


def _duration_seconds(start_time: str, end_time: str) -> float:
    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
        return (end - start).total_seconds()
    except Exception:
        return 0.0


def _now() -> str:
    return datetime.now().isoformat()
