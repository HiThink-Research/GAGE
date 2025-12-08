"""Data validation utilities based on Pydantic models."""

from __future__ import annotations

import importlib
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ValidationError

try:  # Pydantic v2+
    from pydantic import ConfigDict  # type: ignore
except ImportError:  # pragma: no cover - v1 fallback
    ConfigDict = None

from gage_eval.observability.trace import ObservabilityTrace

logger = logging.getLogger(__name__)


class ValidationMode(str, Enum):
    OFF = "off"
    WARN = "warn"
    STRICT = "strict"


class DefaultEnvelopeModel(BaseModel):
    """Minimal schema covering Sample Envelope 核心字段。"""

    id: str
    messages: list
    choices: list
    predict_result: list = Field(default_factory=list)
    eval_result: Dict[str, Any] = Field(default_factory=dict)
    data_tag: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    label: Optional[Any] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    sampling_params: Dict[str, Any] = Field(default_factory=dict)
    generation_params: Dict[str, Any] = Field(default_factory=dict)
    model_prompt_tmpl: str = ""
    model_prompt_placeholder: List[Any] = Field(default_factory=list)

    if ConfigDict is not None:  # pragma: no branch - runtime evaluated once
        model_config = ConfigDict(extra="allow", protected_namespaces=())  # type: ignore[assignment]
    else:  # pragma: no cover - Pydantic v1
        class Config:
            extra = "allow"
            protected_namespaces = ()


class SampleValidator:
    """Validates raw dataset records and/or standardized samples."""

    def __init__(
        self,
        *,
        raw_model: Optional[Type[BaseModel]] = None,
        envelope_model: Optional[Type[BaseModel]] = None,
        mode: ValidationMode = ValidationMode.WARN,
    ) -> None:
        self._raw_model = raw_model
        self._envelope_model = envelope_model or DefaultEnvelopeModel
        self._mode = mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate_raw(
        self,
        record: Dict[str, Any],
        *,
        dataset_id: str,
        index: int,
        trace: Optional[ObservabilityTrace] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._raw_model:
            return record
        try:
            instance = self._model_validate(self._raw_model, record)
            return self._model_dump(instance)
        except ValidationError as exc:
            return self._handle_failure("raw_record", exc, dataset_id=dataset_id, index=index, trace=trace)

    def validate_envelope(
        self,
        sample: Dict[str, Any],
        *,
        dataset_id: str,
        sample_id: Optional[str],
        trace: Optional[ObservabilityTrace] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._envelope_model or self._mode == ValidationMode.OFF:
            return sample
        try:
            instance = self._model_validate(self._envelope_model, sample)
            return self._model_dump(instance)
        except ValidationError as exc:
            return self._handle_failure(
                "sample_envelope",
                exc,
                dataset_id=dataset_id,
                sample_id=sample_id,
                trace=trace,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _handle_failure(
        self,
        step: str,
        error: ValidationError,
        *,
        dataset_id: str,
        index: Optional[int] = None,
        sample_id: Optional[str] = None,
        trace: Optional[ObservabilityTrace],
    ) -> Optional[Dict[str, Any]]:
        message = (
            f"Data validation failed ({step}) for dataset='{dataset_id}' "
            f"index={index} sample_id={sample_id}: {error}"
        )
        if self._mode == ValidationMode.STRICT:
            raise ValueError(message) from error
        logger.warning(message)
        if trace:
            trace.emit(
                "data_validation_failed",
                {
                    "dataset_id": dataset_id,
                    "step": step,
                    "index": index,
                    "sample_id": sample_id,
                    "errors": error.errors(),
                },
            )
        return None

    @staticmethod
    def _model_validate(model: Type[BaseModel], payload: Dict[str, Any]) -> BaseModel:
        if hasattr(model, "model_validate"):
            return model.model_validate(payload)  # type: ignore[attr-defined]
        return model.parse_obj(payload)  # type: ignore[return-value]

    @staticmethod
    def _model_dump(instance: BaseModel) -> Dict[str, Any]:
        if hasattr(instance, "model_dump"):
            return instance.model_dump()  # type: ignore[attr-defined]
        return instance.dict()  # type: ignore[return-value]


def validate_sample_schema(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure core fields are of expected container types."""

    if not isinstance(sample.get("messages"), list):
        sample["messages"] = [] if sample.get("messages") is not None else []
    if not isinstance(sample.get("choices"), list):
        sample["choices"] = []
    if not isinstance(sample.get("metadata"), dict):
        sample["metadata"] = {}
    if not isinstance(sample.get("data_tag"), dict):
        sample["data_tag"] = {}
    if not isinstance(sample.get("eval_result"), dict):
        sample["eval_result"] = {}
    inputs = sample.get("inputs")
    if not isinstance(inputs, dict):
        prompt_val = sample.get("prompt") if isinstance(sample.get("prompt"), str) else None
        sample["inputs"] = {"prompt": prompt_val} if prompt_val else {}
    return sample


def build_validator(config: Optional[Dict[str, Any]]) -> Optional[SampleValidator]:
    """Create a SampleValidator from DatasetSpec.schema 或默认 Envelope 校验。"""

    if config is None:
        return SampleValidator(envelope_model=DefaultEnvelopeModel, mode=ValidationMode.WARN)

    mode = ValidationMode(config.get("mode", "warn"))
    raw_model = _maybe_import_model(config.get("raw_model"))
    envelope_model = _maybe_import_model(config.get("envelope_model")) or DefaultEnvelopeModel

    if mode == ValidationMode.OFF and raw_model is None and config.get("envelope_model") is None:
        return None

    return SampleValidator(raw_model=raw_model, envelope_model=envelope_model, mode=mode)


def _maybe_import_model(path: Optional[str]) -> Optional[Type[BaseModel]]:
    if not path:
        return None
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    elif "." in path:
        module_name, class_name = path.rsplit(".", 1)
    else:
        raise ValueError(f"Schema model path '{path}' must include module and class name")
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    if not issubclass(model_cls, BaseModel):  # type: ignore[arg-type]
        raise TypeError(f"Schema model '{path}' must inherit from pydantic.BaseModel")
    return model_cls
