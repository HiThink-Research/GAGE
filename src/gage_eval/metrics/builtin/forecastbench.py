"""ForecastBench probability forecasting metric (rule-based Brier scoring)."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Optional

from gage_eval.metrics.base import BaseMetric, MetricContext, MetricResult
from gage_eval.metrics.utils import get_first_reference, get_text_content_of_first_predict_result
from gage_eval.registry import registry


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_STARRED_PROBABILITY = re.compile(r"\*\s*((?:0(?:\.\d+)?)|(?:1(?:\.0+)?)|(?:\.\d+))\s*\*")
_PLAIN_PROBABILITY = re.compile(r"^\s*((?:0(?:\.\d+)?)|(?:1(?:\.0+)?)|(?:\.\d+))\s*$")


def _strip_json_fences(text: str) -> str:
    match = _JSON_FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_last_json_object(text: str) -> Optional[dict[str, Any]]:
    """Return the last top-level JSON object in *text* if it parses as an object."""

    last_obj: Optional[dict[str, Any]] = None
    depth = 0
    start: Optional[int] = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                chunk = text[start : index + 1]
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    last_obj = parsed
                start = None
    return last_obj


def _extract_starred_probability(text: str) -> Optional[float]:
    matches = _STARRED_PROBABILITY.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


def _extract_plain_probability(text: str) -> Optional[float]:
    match = _PLAIN_PROBABILITY.match(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _parse_forecast_blob(raw: str) -> tuple[Optional[float], bool]:
    """Return (forecast, ok). *ok* is False when forecast must fall back."""

    text = _strip_json_fences(raw)
    if not text:
        return None, False
    starred = _extract_starred_probability(text)
    if starred is not None:
        return starred, True
    plain = _extract_plain_probability(text)
    if plain is not None:
        return plain, True
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "forecast" in data:
            return float(data["forecast"]), True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    fallback_obj = _extract_last_json_object(text)
    if isinstance(fallback_obj, dict) and "forecast" in fallback_obj:
        try:
            return float(fallback_obj["forecast"]), True
        except (TypeError, ValueError):
            return None, False
    return None, False


def _prediction_text(context: MetricContext) -> str:
    sample = context.sample if isinstance(context.sample, Mapping) else {}
    text = get_text_content_of_first_predict_result(sample)  # type: ignore[arg-type]
    if text:
        return str(text)
    model_out = context.model_output if isinstance(context.model_output, Mapping) else {}
    for key in ("answer", "text", "content"):
        val = model_out.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _resolved_reference(sample: Mapping[str, Any]) -> float:
    sample_id = str(sample.get("id", "?"))
    ref = get_first_reference(sample)
    if ref is None and "references" in sample:
        refs = sample.get("references")
        if isinstance(refs, list) and refs:
            ref = refs[0]
    if ref is None:
        label = sample.get("label")
        if label is not None and str(label).strip() != "":
            ref = label
    if ref is None:
        raise ValueError(f"Sample '{sample_id}' is missing a resolved numeric reference (references[0] or label).")
    try:
        return float(ref)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sample '{sample_id}': cannot convert reference {ref!r} to float.") from exc


@registry.asset(
    "metrics",
    "forecastbench_probability",
    desc="ForecastBench probability rule metric (Brier / calibration)",
    tags=("forecastbench", "probability", "rule"),
    default_aggregation="forecastbench_probability_summary",
)
class ForecastBenchProbabilityMetric(BaseMetric):
    """Score a single probability forecast against ``references[0]`` (resolved outcome)."""

    def compute(self, context: MetricContext) -> MetricResult:
        sample = context.sample if isinstance(context.sample, Mapping) else {}
        resolved_to = _resolved_reference(sample)

        raw_text = _prediction_text(context)
        raw_forecast, parsed_ok = _parse_forecast_blob(raw_text)
        parse_error = 0.0 if parsed_ok else 1.0

        clamp_applied = 0.0
        if not parsed_ok:
            forecast = 0.5
        else:
            forecast = float(raw_forecast)
            if forecast < 0.0 or forecast > 1.0:
                clamp_applied = 1.0
                forecast = max(0.0, min(1.0, forecast))

        brier = (forecast - resolved_to) ** 2
        brier_index_simple_case = (1.0 - math.sqrt(max(0.0, brier))) * 100.0
        abs_error = abs(forecast - resolved_to)
        accuracy_at_0_5 = 1.0 if (forecast >= 0.5) == (resolved_to >= 0.5) else 0.0

        meta_extra: dict[str, Any] = {}

        freeze_raw = None
        if isinstance(sample.get("metadata"), Mapping):
            freeze_raw = sample["metadata"].get("freeze_datetime_value")
        freeze_val: Optional[float]
        try:
            freeze_val = float(freeze_raw) if freeze_raw is not None else None
        except (TypeError, ValueError):
            freeze_val = None

        market_baseline_brier: Optional[float] = None
        model_minus_market_brier: Optional[float] = None
        if freeze_val is not None:
            market_baseline_brier = (freeze_val - resolved_to) ** 2
            model_minus_market_brier = brier - market_baseline_brier
            meta_extra["market_baseline_brier"] = market_baseline_brier
            meta_extra["model_minus_market_brier"] = model_minus_market_brier

        values = {
            "brier": float(brier),
            "brier_index_simple_case": float(brier_index_simple_case),
            "accuracy_at_0_5": float(accuracy_at_0_5),
            "abs_error": float(abs_error),
            "parse_error": float(parse_error),
            "clamp_applied": float(clamp_applied),
            "forecast": float(forecast),
            "resolved_to": float(resolved_to),
        }
        if market_baseline_brier is not None and model_minus_market_brier is not None:
            values["market_baseline_brier"] = float(market_baseline_brier)
            values["model_minus_market_brier"] = float(model_minus_market_brier)

        return MetricResult(
            sample_id=context.sample_id,
            values=values,
            metadata=meta_extra,
        )


__all__ = ["ForecastBenchProbabilityMetric"]
