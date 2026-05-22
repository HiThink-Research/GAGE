"""Shared reason-code extraction helpers for summary generators."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


_CANONICAL_REASON_CODE_ALIASES = {
    "missing_appworld_success_signal": "appworld.missing_success_signal",
}


def extract_attention_reason_codes(
    record: Mapping[str, Any],
    *,
    trial: Mapping[str, Any] | None = None,
    fallback: str = "score.low",
) -> list[str]:
    """Extract attention reason codes using the summary-generator contract order."""

    codes: list[str] = []
    trial_payload = (
        trial
        or _mapping(record.get("trial"))
        or _mapping(record.get("trial_result"))
        or first_agentkit_trial_result(record)
    )
    runtime_judge_outcomes = _runtime_judge_outcomes(record)

    for source in (
        _values(
            _path(trial_payload, "failure", "failure_code"),
            _path(trial_payload, "failure_code"),
        ),
        _values(
            _path(record, "judge_output", "verifier_failure", "failure_code"),
            _path(record, "verifier_result", "failure_code"),
            _path(trial_payload, "verifier_result", "failure_code"),
            *[
                _path(outcome, "judge_output", "failure_code")
                for outcome in runtime_judge_outcomes
            ],
            *[
                _path(outcome, "verifier_result", "payload", "failure_code")
                for outcome in runtime_judge_outcomes
            ],
        ),
        _values(
            _path(trial_payload, "scheduler_result", "failure", "failure_code"),
            _path(trial_payload, "scheduler_result", "failure_code"),
            _path(record, "scheduler_result", "failure", "failure_code"),
            _path(record, "scheduler_result", "failure_code"),
            _path(record, "judge_output", "scheduler_failure", "failure_code"),
            *[
                _path(outcome, "verifier_input", "scheduler_result", "failure", "failure_code")
                for outcome in runtime_judge_outcomes
            ],
            *[
                _path(outcome, "verifier_input", "scheduler_result", "failure_code")
                for outcome in runtime_judge_outcomes
            ],
        ),
        _values(
            _path(trial_payload, "model_output", "failure", "failure_code"),
            _path(trial_payload, "model_output", "failure_code"),
            _path(record, "model_output", "failure", "failure_code"),
            _path(record, "model_output", "failure_code"),
            *[
                _path(outcome, "failure", "failure_code")
                for outcome in runtime_judge_outcomes
            ],
        ),
    ):
        _extend_unique(codes, source)

    return codes or [fallback]


def agentkit_trial_results(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return AgentKit trial results from live flattened and samples.jsonl shapes."""

    trials: list[Mapping[str, Any]] = []
    _extend_trial_results(trials, _path(record, "model_output", "agent_eval", "trial_results"))

    sample = _mapping(record.get("sample"))
    predict_result = sample.get("predict_result") if sample else None
    if isinstance(predict_result, list):
        for item in predict_result:
            _extend_trial_results(trials, _path(_mapping(item), "agent_eval", "trial_results"))
    elif isinstance(predict_result, Mapping):
        _extend_trial_results(trials, _path(predict_result, "agent_eval", "trial_results"))

    return trials


def first_agentkit_trial_result(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    trials = agentkit_trial_results(record)
    return trials[0] if trials else None


def _runtime_judge_outcomes(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outcomes: list[Mapping[str, Any]] = []
    _append_mapping(outcomes, _path(record, "model_output", "runtime_judge_outcome"))

    sample = _mapping(record.get("sample"))
    predict_result = sample.get("predict_result") if sample else None
    if isinstance(predict_result, list):
        for item in predict_result:
            _append_mapping(outcomes, _path(_mapping(item), "runtime_judge_outcome"))
    elif isinstance(predict_result, Mapping):
        _append_mapping(outcomes, _path(predict_result, "runtime_judge_outcome"))
    return outcomes


def _values(*values: Any) -> list[str]:
    return [_canonical_reason_code(str(value)) for value in values if value not in (None, "")]


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _path(payload: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _extend_trial_results(target: list[Mapping[str, Any]], value: Any) -> None:
    if not isinstance(value, list):
        return
    for item in value:
        if isinstance(item, Mapping):
            target.append(item)


def _append_mapping(target: list[Mapping[str, Any]], value: Any) -> None:
    if isinstance(value, Mapping):
        target.append(value)


def _canonical_reason_code(value: str) -> str:
    return _CANONICAL_REASON_CODE_ALIASES.get(value, value)


__all__ = [
    "agentkit_trial_results",
    "extract_attention_reason_codes",
    "first_agentkit_trial_result",
]
