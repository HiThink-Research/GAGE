"""Metrics for Harbor external harness imports."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "harbor_resolve_rate",
    desc="Harbor resolve rate using completed-trial denominator",
    tags=("external_harness", "harbor"),
    default_aggregation="mean",
)
class HarborResolveRateMetric(SimpleMetric):
    value_key = "resolve_rate"

    def compute_value(self, context: MetricContext) -> Tuple[float, dict[str, Any]]:
        direct = _coerce_float(_first_present(
            context.get("sample.eval_result.harbor_resolve_rate"),
            context.get("judge_output.harbor_resolve_rate"),
        ))
        if direct is not None:
            return direct, {"source": "harbor_resolve_rate", "missing": False}
        derived = _completed_trial_pass_values(context)
        if derived:
            return sum(1 for value in derived if value) / len(derived), {
                "source": "trial_results",
                "completed_trial_count": len(derived),
                "missing": False,
            }
        fallback = _bool_or_none(_first_present(
            context.get("sample.evaluation.passed"),
            context.get("judge_output.passed"),
            context.get("judge_output.resolved"),
        ))
        if fallback is not None:
            return 1.0 if fallback else 0.0, {"source": "evaluation.passed", "missing": False}
        return 0.0, {
            "source": None,
            "missing": True,
            "empty_completed_trials": True,
            "empty_completed_trials_semantics": "0.0",
        }


@registry.asset(
    "metrics",
    "harbor_score_mean",
    desc="Harbor mean score using completed-trial denominator",
    tags=("external_harness", "harbor"),
    default_aggregation="mean",
)
class HarborScoreMeanMetric(SimpleMetric):
    value_key = "score_mean"

    def compute_value(self, context: MetricContext) -> Tuple[float, dict[str, Any]]:
        direct = _coerce_float(_first_present(
            context.get("sample.eval_result.harbor_score_mean"),
            context.get("judge_output.harbor_score_mean"),
        ))
        if direct is not None:
            return direct, {"source": "harbor_score_mean", "missing": False}
        scores = _completed_trial_scores(context)
        if scores:
            return sum(scores) / len(scores), {
                "source": "trial_results",
                "completed_trial_count": len(scores),
                "missing": False,
            }
        fallback = _coerce_float(_first_present(
            context.get("sample.evaluation.score"),
            context.get("judge_output.score"),
            context.get("judge_output.reward"),
        ))
        if fallback is not None:
            return fallback, {"source": "evaluation.score", "missing": False}
        return 0.0, {
            "source": None,
            "missing": True,
            "empty_completed_trials": True,
            "empty_completed_trials_semantics": "0.0",
        }


@registry.asset(
    "metrics",
    "external_trial_pass_hat_k",
    desc="External harness trial pass-hat@k from compact trial pass projection",
    tags=("external_harness",),
    default_aggregation="mean",
)
class ExternalTrialPassHatKMetric(SimpleMetric):
    value_key = "pass_hat"

    def compute_value(self, context: MetricContext) -> Tuple[float, dict[str, Any]]:
        k = _positive_int(self.args.get("k"), default=1)
        raw_values = _external_trial_pass_values(context)
        values = [_bool_or_none(value) for value in raw_values]
        total = len(values)
        successes = sum(1 for value in values if value is True)
        none_count = sum(1 for value in values if value is None)
        if total <= 0:
            return 0.0, {
                "k": k,
                "trial_count": 0,
                "success_count": 0,
                "none_as_failed": True,
                "empty_trial_values_semantics": "0.0",
            }
        value = _pass_hat_k(successes=successes, total=total, k=k)
        return value, {
            "k": k,
            "trial_count": total,
            "success_count": successes,
            "failure_count": total - successes,
            "none_count": none_count,
            "none_as_failed": True,
            "projection": _metric_projection(context),
        }


def _completed_trial_pass_values(context: MetricContext) -> list[bool]:
    values: list[bool] = []
    for trial in _trial_results(context):
        if str(trial.get("status")) != "completed":
            continue
        value = _bool_or_none(_first_present(
            _mapping(trial.get("verifier_result")).get("passed"),
            _mapping(trial.get("verifier_result")).get("resolved"),
            _mapping(trial.get("verifier_result")).get("pass"),
        ))
        if value is not None:
            values.append(value)
    return values


def _completed_trial_scores(context: MetricContext) -> list[float]:
    scores: list[float] = []
    for trial in _trial_results(context):
        if str(trial.get("status")) != "completed":
            continue
        score = _coerce_float(_first_present(
            _mapping(trial.get("verifier_result")).get("score"),
            _mapping(trial.get("verifier_result")).get("reward"),
        ))
        if score is not None:
            scores.append(score)
    return scores


def _external_trial_pass_values(context: MetricContext) -> list[Any]:
    raw = _first_present(
        context.get("sample.eval_result.external_trial_pass_values"),
        context.get("judge_output.external_trial_pass_values"),
    )
    if isinstance(raw, list):
        return list(raw)
    return []


def _metric_projection(context: MetricContext) -> dict[str, Any]:
    raw = _first_present(
        context.get("sample.eval_result.external_trial_metric_projection"),
        context.get("judge_output.external_trial_metric_projection"),
    )
    return dict(raw) if isinstance(raw, Mapping) else {}


def _trial_results(context: MetricContext) -> list[Mapping[str, Any]]:
    raw = _first_present(
        context.get("judge_output.trial_results"),
        context.get("sample.trial_results"),
        context.get("sample.metadata.trial_results"),
    )
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _pass_hat_k(*, successes: int, total: int, k: int) -> float:
    if k <= 0 or total < k:
        return 0.0
    if successes < k:
        return 0.0
    return math.comb(successes, k) / math.comb(total, k)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _positive_int(value: Any, *, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved >= 1 else default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "ExternalTrialPassHatKMetric",
    "HarborResolveRateMetric",
    "HarborScoreMeanMetric",
]
