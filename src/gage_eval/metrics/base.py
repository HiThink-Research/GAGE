"""Define core metric abstractions: context, results, and base classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING, Tuple, Union, Iterable, Callable

from loguru import logger
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.utils import extract_field, normalize_text_advanced, ensure_list_of_strings, levenshtein_distance

if TYPE_CHECKING:  # Avoid circular imports.
    from gage_eval.observability.trace import ObservabilityTrace


@dataclass(frozen=True)
class MetricContext:
    """Carry runtime context passed into metric implementations."""

    sample_id: str
    sample: Mapping[str, Any]
    model_output: Mapping[str, Any]
    judge_output: Mapping[str, Any]
    args: Mapping[str, Any]
    trace: "ObservabilityTrace"

    def get(self, descriptor: Optional[str], default: Any = None) -> Any:
        """Return a nested field value using the framework's descriptor syntax."""

        return extract_field(self, descriptor, default=default)


@dataclass(frozen=True)
class MetricResult:
    """Represent the per-sample metric output."""

    sample_id: str
    values: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"sample_id": self.sample_id, "values": self.values}
        if self.metadata:
            payload["metadata"] = self.metadata
        if self.explanation:
            payload["explanation"] = self.explanation
        return payload


@dataclass(frozen=True)
class AggregatedMetric:
    """Represent an aggregated metric result across samples."""

    metric_id: str
    aggregation: str
    values: Dict[str, float]
    count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "metric_id": self.metric_id,
            "aggregation": self.aggregation,
            "values": self.values,
            "count": self.count,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class BaseMetric:
    """Define the abstract base class for all metrics."""

    def __init__(self, spec: MetricSpec) -> None:
        self.spec = spec
        self.args = dict(spec.params)
        self.setup()
        logger.debug("Metric '{}' initialized", spec.metric_id)

    def setup(self) -> None:
        """Run optional initialization (useful for loading heavy resources)."""

    def teardown(self) -> None:
        """Run optional cleanup for metric resources."""
        logger.debug("Metric '{}' teardown invoked", self.spec.metric_id)

    def compute(self, context: MetricContext) -> MetricResult:  # pragma: no cover
        raise NotImplementedError


class SimpleMetric(BaseMetric):
    """Provide a convenience base class for single-float metrics."""

    value_key: str = "score"

    def compute_value(self, context: MetricContext) -> Union[float, Tuple[float, Dict[str, Any]]]:  # pragma: no cover
        raise NotImplementedError

    def compute_metadata(self, context: MetricContext) -> Optional[Dict[str, Any]]:
        """Return optional metadata; defaults to None."""

        return None

    def compute(self, context: MetricContext) -> MetricResult:
        raw_value = self.compute_value(context)
        metadata: Dict[str, Any] = {}
        if isinstance(raw_value, tuple):
            if len(raw_value) != 2:
                raise ValueError("SimpleMetric.compute_value must return (value, metadata) when using tuple form")
            value, metadata_part = raw_value
            if metadata_part:
                metadata.update(metadata_part)
        else:
            value = raw_value
            extra_metadata = self.compute_metadata(context)
            if extra_metadata:
                metadata.update(extra_metadata)
        value = float(value)
        logger.trace("Metric '{}' computed value={}", self.spec.metric_id, value)
        return MetricResult(sample_id=context.sample_id, values={self.value_key: value}, metadata=metadata)


class ComparisonMetric(SimpleMetric):
    """Provide a convenience base class for prediction-vs-reference comparisons."""

    default_prediction_field: str = "model_output.answer"
    default_reference_field: str = "sample.label"

    def get_prediction_field(self) -> str:
        return str(self.args.get("prediction_field", self.default_prediction_field))

    def get_reference_field(self) -> str:
        return str(self.args.get("label_field", self.default_reference_field))

    def extract_prediction(self, context: MetricContext) -> Any:
        return context.get(self.get_prediction_field())

    def extract_reference(self, context: MetricContext) -> Any:
        return context.get(self.get_reference_field())

    def compare(self, prediction: Any, reference: Any) -> Tuple[float, Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def compute(self, context: MetricContext) -> MetricResult:
        prediction = self.extract_prediction(context)
        reference = self.extract_reference(context)
        score, metadata = self.compare(prediction, reference)
        merged_meta = {"prediction": prediction, "reference": reference}
        if metadata:
            merged_meta.update(metadata)
        value = float(score)
        logger.trace("ComparisonMetric '{}' computed value={}", self.spec.metric_id, value)
        return MetricResult(sample_id=context.sample_id, values={self.value_key: value}, metadata=merged_meta)


class SequenceDistanceMetric(ComparisonMetric):
    """Compute a similarity score using a pluggable distance function (defaults to Levenshtein)."""

    distance_fn: Callable[[str, str], int] = staticmethod(levenshtein_distance)

    def _normalize_inputs(self, prediction: Any, reference: Any) -> Tuple[str, str]:
        pred = normalize_text_advanced(prediction, collapse_whitespace=True) or ""
        ref = normalize_text_advanced(reference, collapse_whitespace=True) or ""
        return pred, ref

    def _compute_distance(self, prediction: str, reference: str) -> int:
        return self.distance_fn(prediction, reference)

    def _normalize_score(self, distance: int, prediction: str, reference: str) -> Tuple[float, str]:
        strategy = str(self.args.get("normalize", "anls")).lower()
        if strategy == "wer":
            denom = len(reference) or 1
            score = max(1.0 - distance / denom, 0.0)
        elif strategy == "raw":
            score = float(distance)
        else:  # Default: ANLS-style normalization.
            denom = max(len(prediction), len(reference)) or 1
            score = max(1.0 - distance / denom, 0.0)
        return score, strategy

    def compare(self, prediction: Any, reference: Any) -> Tuple[float, Dict[str, Any]]:
        pred_norm, ref_norm = self._normalize_inputs(prediction, reference)
        if not pred_norm and not ref_norm:
            return 1.0, {"distance": 0, "normalized": 1.0, "strategy": self.args.get("normalize", "anls")}
        if not pred_norm or not ref_norm:
            return 0.0, {"distance": None, "normalized": 0.0, "strategy": self.args.get("normalize", "anls"), "warning": "empty_input"}
        distance = self._compute_distance(pred_norm, ref_norm)
        score, strategy = self._normalize_score(distance, pred_norm, ref_norm)
        return score, {"distance": distance, "normalized": score, "strategy": strategy}


class MultiReferenceTextMetric(ComparisonMetric):
    """Score a prediction against multiple text references and keep the best match."""

    default_reference_field = "sample.references"

    def _normalize_prediction(self, prediction: Any) -> str:
        return normalize_text_advanced(
            prediction,
            case_sensitive=bool(self.args.get("case_sensitive", False)),
            strip=True,
            collapse_whitespace=True,
        ) or ""

    def _normalize_reference(self, reference: Any) -> str:
        return normalize_text_advanced(
            reference,
            case_sensitive=bool(self.args.get("case_sensitive", False)),
            strip=True,
            collapse_whitespace=True,
        ) or ""

    def extract_references(self, context: MetricContext) -> list[str]:
        ref_field = self.args.get("references_field", self.get_reference_field())
        raw = context.get(ref_field, default=())
        refs = ensure_list_of_strings(raw)
        sep = self.args.get("separator")
        if sep and len(refs) == 1:
            refs = [chunk for chunk in refs[0].split(sep) if chunk]
        return refs

    def score_single(self, prediction: str, reference: str) -> float:  # pragma: no cover
        return 1.0 if prediction == reference else 0.0

    def aggregate_scores(self, scores: Iterable[float], references: list[str], prediction: str) -> Tuple[float, Dict[str, Any]]:
        scores_list = list(scores)
        if not scores_list:
            return 0.0, {"warning": "empty_refs"}
        best_score = max(scores_list)
        best_ref = references[scores_list.index(best_score)] if references else None
        return best_score, {"best_score": best_score, "best_reference": best_ref}

    def compute(self, context: MetricContext) -> MetricResult:
        prediction_raw = self.extract_prediction(context)
        references_raw = self.extract_references(context)
        prediction = self._normalize_prediction(prediction_raw)
        references = [self._normalize_reference(ref) for ref in references_raw]

        metadata: Dict[str, Any] = {"prediction": prediction, "references": references}
        if not prediction or not references:
            metadata["warning"] = "empty_refs_or_pred"
            return MetricResult(sample_id=context.sample_id, values={self.value_key: 0.0}, metadata=metadata)

        scores = [self.score_single(prediction, ref) for ref in references]
        score, extra_meta = self.aggregate_scores(scores, references, prediction)
        if extra_meta:
            metadata.update(extra_meta)
        return MetricResult(sample_id=context.sample_id, values={self.value_key: float(score)}, metadata=metadata)


class NumericThresholdMetric(ComparisonMetric):
    """Compare numeric predictions with tolerance/threshold modes."""

    def _to_number(self, value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _get_threshold_config(self) -> Dict[str, Any]:
        return {
            "tolerance": float(self.args.get("tolerance", 0.0)),
            "min_value": self.args.get("min_value"),
            "max_value": self.args.get("max_value"),
            "mode": str(self.args.get("mode", "abs_diff")).lower(),
            "fallback": float(self.args.get("fallback", 0.0)),
        }

    def compare(self, prediction: Any, reference: Any) -> Tuple[float, Dict[str, Any]]:
        cfg = self._get_threshold_config()
        pred_val = self._to_number(prediction)
        ref_val = self._to_number(reference)
        metadata: Dict[str, Any] = {k: v for k, v in cfg.items() if k != "fallback"}

        if pred_val is None or (cfg["mode"] not in ("ge", "le") and ref_val is None):
            metadata["invalid_format"] = True
            return cfg["fallback"], metadata

        if cfg["mode"] == "ge":
            if ref_val is None:
                metadata["invalid_format"] = True
                return cfg["fallback"], metadata
            score = 1.0 if pred_val >= ref_val else 0.0
        elif cfg["mode"] == "le":
            if ref_val is None:
                metadata["invalid_format"] = True
                return cfg["fallback"], metadata
            score = 1.0 if pred_val <= ref_val else 0.0
        else:
            score = 1.0 if ref_val is not None and abs(ref_val - pred_val) <= cfg["tolerance"] else 0.0

        if cfg["min_value"] is not None and pred_val < float(cfg["min_value"]):
            score = 0.0
        if cfg["max_value"] is not None and pred_val > float(cfg["max_value"]):
            score = 0.0

        metadata.update({"prediction": pred_val, "reference": ref_val})
        return score, metadata
