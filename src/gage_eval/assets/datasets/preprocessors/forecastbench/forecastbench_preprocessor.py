"""ForecastBench preprocessor: raw joined records -> GAGE Sample."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.sample import SCHEMA_VERSION, Message, MessageContent, Sample


def _stable_sample_id(*, question_set: str, source: str, question_id: str) -> str:
    stem = Path(str(question_set)).stem
    return f"forecastbench:{stem}:{_norm_token(source)}:{_norm_token(question_id)}"


def _norm_token(value: str) -> str:
    return str(value).strip()


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _infer_today_date(*, forecast_due: str, question_set: str) -> str:
    if forecast_due.strip():
        return forecast_due.strip()
    stem = Path(str(question_set)).stem
    if len(stem) >= 10 and stem[4:5] == "-" and stem[7:8] == "-":
        return stem[:10]
    return ""


class ForecastBenchPreprocessor(BasePreprocessor):
    """Convert ForecastBench joined records into probability-forecast Samples.

    Public callers use :meth:`transform` from :class:`BasePreprocessor`; this class only
    implements :meth:`to_sample` for the shared validation and multimodal merge pipeline.
    """

    name = "forecastbench_static"

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        schema_version: str = SCHEMA_VERSION,
        **kwargs: Any,
    ) -> Sample:
        question_id = str(record.get("id", ""))
        source = str(record.get("source", ""))
        question_set = str(record.get("question_set", "unknown.json"))
        resolved_to = record.get("resolved_to")
        if resolved_to is None:
            raise ValueError("ForecastBench record missing resolved_to")

        freeze_val = _as_float(record.get("freeze_datetime_value"))
        forecast_due = str(record.get("forecast_due_date", ""))
        question_text = str(record.get("question", ""))
        resolution_criteria = str(record.get("resolution_criteria", ""))
        background = str(record.get("background", ""))
        freeze_dt = str(record.get("freeze_datetime", ""))
        resolution_date = str(
            record.get("market_info_close_datetime")
            or record.get("resolution_date")
            or record.get("close_datetime")
            or ""
        )
        today_date = _infer_today_date(forecast_due=forecast_due, question_set=question_set)

        lines = [
            "You are an expert superforecaster, familiar with the work of Tetlock and others.",
            "Make a prediction of the probability that the question will be resolved as true. "
            "You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. "
            "If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.",
            f"Question: {question_text}",
            f"Question Background: {background}",
            f"Resolution Criteria: {resolution_criteria}",
            f"Question Resolution Date: {resolution_date}",
            f"Today's Date: {today_date}",
        ]
        if freeze_val is not None:
            include_market_baseline = _coerce_bool(kwargs.get("include_market_baseline_in_prompt", True))
            if include_market_baseline:
                lines.append(f"Market value on {freeze_dt}: {freeze_val}")
        lines.extend(
            [
                "Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.",
                "Do not output anything else.",
                "Answer: { Insert answer here }",
            ]
        )
        prompt = "\n".join(lines)

        ref_float = float(resolved_to)
        metadata: Dict[str, Any] = {
            "benchmark": "forecastbench",
            "question_id": question_id,
            "forecast_due_date": forecast_due,
            "source": source,
        }
        if freeze_val is not None:
            metadata["freeze_datetime_value"] = freeze_val

        data_tag = {
            "benchmark": "forecastbench",
            "source": source,
            "question_type": "market",
        }

        sample_id = _stable_sample_id(question_set=question_set, source=source, question_id=question_id)

        message = Message(role="user", content=[MessageContent(type="text", text=prompt)])
        return Sample(
            schema_version=schema_version,
            id=sample_id,
            task_type="probability-forecast",
            messages=[message],
            references=[ref_float],
            label=str(ref_float),
            metadata=metadata,
            data_tag=data_tag,
        )
