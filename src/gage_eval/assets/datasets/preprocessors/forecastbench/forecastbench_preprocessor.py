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

        lines = [
            "You are forecasting a Polymarket-style binary market.",
            "",
            f"Question: {question_text}",
            f"Resolution criteria: {resolution_criteria}",
            f"Background: {background}",
            f"Information freeze datetime (UTC): {freeze_dt}",
        ]
        if freeze_val is not None:
            lines.append(f"Market-implied probability at freeze (if available): {freeze_val}")
        lines.extend(
            [
                f"Forecast due date: {forecast_due}",
                "",
                "Do not use web search or browse the internet. Use only the information provided above.",
                "",
                'Return JSON only with exactly this shape: {"forecast": <number between 0 and 1>, "reasoning": "<short string>"}',
                "Do not wrap the JSON in markdown fences.",
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
