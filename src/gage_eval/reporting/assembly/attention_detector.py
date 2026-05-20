from __future__ import annotations

from typing import Any

from gage_eval.reporting.contracts import AttentionCase, AttentionCaseScoring, ReasonCodeRegistry, Severity


SCORING_CONFIG = {
    "formula": "0.30*frequency + 0.50*impact_weight + 0.20*actionability_weight",
    "weights": {"frequency": 0.3, "impact": 0.5, "actionability": 0.2},
    "impact_weights": {
        "critical": 1.0,
        "high": 0.85,
        "medium": 0.6,
        "low": 0.3,
        "unknown": 0.4,
    },
    "actionability_weights": {
        "high": 1.0,
        "medium": 0.65,
        "low": 0.3,
        "unknown": 0.4,
    },
}

DIRECT_HIGH_REASON_CODES = frozenset(
    {"scheduler.failed", "verifier.skipped", "runtime.error", "timeout"}
)


class AttentionCaseDetector:
    def __init__(self, top_k: int = 50) -> None:
        self.top_k = top_k
        self.registry = ReasonCodeRegistry.load_builtin()

    def detect(self, candidates: list[dict[str, Any]], *, total_samples: int) -> list[AttentionCase]:
        cases = [self._case(candidate, total_samples=max(total_samples, 1)) for candidate in candidates]
        cases.sort(key=lambda item: (-(item.scoring.priority_score or 0), item.case_id or ""))
        return cases[: self.top_k]

    def _case(self, candidate: dict[str, Any], *, total_samples: int) -> AttentionCase:
        codes = list(candidate.get("reason_codes") or ["runtime.error"])
        impact = _max_level([self._impact(code) for code in codes], SCORING_CONFIG["impact_weights"])
        actionability = _max_level(
            [self._actionability(code) for code in codes],
            SCORING_CONFIG["actionability_weights"],
        )
        frequency = _frequency(candidate, total_samples=total_samples)
        score = min(
            1.0,
            frequency * SCORING_CONFIG["weights"]["frequency"]
            + _level_score(impact, SCORING_CONFIG["impact_weights"]) * SCORING_CONFIG["weights"]["impact"]
            + _level_score(actionability, SCORING_CONFIG["actionability_weights"])
            * SCORING_CONFIG["weights"]["actionability"],
        )
        return AttentionCase(
            case_id=candidate.get("case_id") or _join_case_id(candidate),
            task_id=candidate.get("task_id"),
            sample_id=candidate.get("sample_id"),
            trial_id=candidate.get("trial_id"),
            severity=_severity(score, reason_codes=codes),
            scoring=_Scoring(
                frequency=frequency,
                impact=impact,
                actionability=actionability,
                priority_score=round(score, 4),
            ),
            reason_codes=codes,
            summary=str(candidate.get("summary") or ", ".join(codes)),
            evidence_ref_ids=list(candidate.get("evidence_ref_ids") or []),
        )

    def _impact(self, code: str) -> str:
        return self.registry.reason_codes.get(code, _Default()).impact_default

    def _actionability(self, code: str) -> str:
        return self.registry.reason_codes.get(code, _Default()).actionability_default


class _Default:
    impact_default = "unknown"
    actionability_default = "unknown"


def _join_case_id(candidate: dict[str, Any]) -> str:
    return "/".join(str(candidate.get(key, "")) for key in ("task_id", "sample_id") if candidate.get(key)) or "case"


def _frequency(candidate: dict[str, Any], *, total_samples: int) -> float:
    raw_frequency = candidate.get("frequency")
    if raw_frequency is None:
        raw_frequency = 1 / total_samples
    return max(0.0, min(1.0, float(raw_frequency)))


def _level_score(level: str, weights: dict[str, float]) -> float:
    return float(weights.get(level, weights["unknown"]))


def _max_level(levels: list[str], weights: dict[str, float]) -> str:
    return max(levels, key=lambda level: _level_score(level, weights)) if levels else "unknown"


def _severity(score: float, *, reason_codes: list[str]) -> str:
    if score >= 0.85:
        return Severity.CRITICAL
    if any(code in DIRECT_HIGH_REASON_CODES for code in reason_codes):
        return Severity.HIGH
    if score >= 0.7:
        return Severity.HIGH
    if score >= 0.45:
        return Severity.MEDIUM
    if score >= 0.2:
        return Severity.LOW
    return Severity.INFO


class _Scoring(AttentionCaseScoring):
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]
