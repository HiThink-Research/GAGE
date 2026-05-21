"""ForecastBench preprocessor: raw joined records -> GAGE Sample."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Optional

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.sample import SCHEMA_VERSION, Message, MessageContent, Sample


# NOTE: Known ForecastBench sources mapped to short, filesystem-friendly codes used in
# the sample id. Unknown sources fall back to the first four characters of the
# slugged normalized name (see :func:`_short_source_code`) so artifact filenames stay
# bounded without depending on the global cache sanitizer.
_SOURCE_SHORT_CODES: Dict[str, str] = {
    "polymarket": "pm",
}


def _norm_token(value: str) -> str:
    return str(value).strip()


def _short_source_code(source: str) -> str:
    """Return a short, filesystem-friendly code for *source*."""

    normalized = str(source).strip().lower()
    if not normalized:
        return "unk"
    mapped = _SOURCE_SHORT_CODES.get(normalized)
    if mapped:
        return mapped
    slug = "".join(ch for ch in normalized if ch.isalnum() or ch in {"-", "_"})
    return slug[:4] or "unk"


def _short_date_prefix(question_set: str) -> str:
    """Extract ``YYYY-MM-DD`` from the question_set stem when it looks like a date."""

    stem = Path(str(question_set)).stem
    if len(stem) >= 10 and stem[4:5] == "-" and stem[7:8] == "-":
        return stem[:10]
    return stem[:10] if stem else "unknown"


def _stable_sample_id(*, question_set: str, source: str, question_id: str) -> str:
    """Build a short, stable sample id safe for artifact filenames.

    Format: ``fb:<date>:<src>:<prefix10>_<hash8>``. Keeping the id short keeps
    per-sample artifact paths well within Linux's 255-byte filename limit and
    Windows' default 260-character ``MAX_PATH`` cap without relying on the global
    :class:`gage_eval.evaluation.cache.EvalCache` truncation hook. The original
    identifiers (``question_set``, ``source``, ``question_id``) are preserved in
    :attr:`Sample.metadata` so the legacy
    ``forecastbench:<stem>:<source>:<question_id>`` form is always reconstructable.
    """

    date_code = _short_date_prefix(question_set)
    source_code = _short_source_code(source)
    qid = _norm_token(question_id)
    digest = hashlib.sha1(qid.encode("utf-8")).hexdigest()[:8]
    prefix = qid[:10].rstrip("_") if qid else ""
    suffix = f"{prefix}_{digest}" if prefix else digest
    return f"fb:{date_code}:{source_code}:{suffix}"


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
    """Pick the prompt's ``Today's Date`` line.

    Prefer the explicit ``forecast_due_date`` on the record; otherwise extract a
    ``YYYY-MM-DD`` prefix from the question_set stem. If neither is available,
    return an empty string so the prompt does not surface a misleading date.
    """

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
    To ablate the freeze market value from prompts, set
    ``params.preprocess_kwargs.include_market_baseline_in_prompt: false`` in YAML.
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
        ref_float = _as_float(resolved_to)
        if ref_float is None or not math.isfinite(ref_float):
            raise ValueError(
                f"ForecastBench record {question_id!r} has missing or non-numeric resolved_to: {resolved_to!r}"
            )

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

        # NOTE: ``question_set`` (full original filename) and ``question_id`` (full
        # original identifier) are kept in metadata so the legacy
        # ``forecastbench:<stem>:<source>:<id>`` sample id can be reconstructed
        # downstream even though :func:`_stable_sample_id` shortens the persisted form.
        metadata: Dict[str, Any] = {
            "benchmark": "forecastbench",
            "question_id": question_id,
            "question_set": question_set,
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
