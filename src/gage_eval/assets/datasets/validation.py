"""Data validation utilities based on Pydantic models."""

from __future__ import annotations

import importlib
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationError

try:  # Pydantic v2+
    from pydantic import ConfigDict  # type: ignore
except ImportError:  # pragma: no cover - v1 fallback
    ConfigDict = None

from gage_eval.observability.trace import ObservabilityTrace
from dataclasses import dataclass, asdict, is_dataclass
from gage_eval.assets.datasets.sample import (
    Sample,
)

logger = logging.getLogger(__name__)

class ValidationMode(str, Enum):
    OFF = "off"
    WARN = "warn"
    STRICT = "strict"

class DefaultEnvelopeModel(BaseModel):
    """Minimal schema covering the core fields of the Sample envelope."""
    schema_version: str
    id: str
    messages: list

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
        record: Union[Dict[str, Any], Sample],
        *,
        dataset_id: str,
        index: int,
        trace: Optional[ObservabilityTrace] = None,
    ) -> Optional[Dict[str, Any]]:
        record_dict = asdict(record) if is_dataclass(record) else record
        if not self._raw_model:
            return record_dict
        try:
            instance = self._model_validate(self._raw_model, record_dict)
            return self._model_dump(instance)
        except ValidationError as exc:
            return self._handle_failure("raw_record", exc, dataset_id=dataset_id, index=index, trace=trace)

    def validate_envelope(
        self,
        sample: Union[Dict[str, Any], Sample],
        *,
        dataset_id: str,
        sample_id: Optional[str],
        trace: Optional[ObservabilityTrace] = None,
    ) -> Optional[Dict[str, Any]]:
        sample_dict = asdict(sample) if is_dataclass(sample) else sample
        if not self._envelope_model or self._mode == ValidationMode.OFF:
            return sample_dict
        try:
            instance = self._model_validate(self._envelope_model, sample_dict)
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

def _validate_list_field(sample, attr):
    """ Ensure each list filed of a Sample object"""
    if not hasattr(sample, attr):
        raise ValueError(f"sample should have a not null messages {attr}.")
    attr_val = getattr(sample, attr)
    if attr_val is None or not isinstance(attr_val, list):
        raise ValueError(f"sample.{attr} should be a list.")
    if len(attr_val) == 0:
        raise ValueError(f"sample.{attr} should not be empty.")

def _validate_str_field(sample, attr):
    """ Ensure each str filed of a Sample object"""
    if not hasattr(sample, attr):
        raise ValueError(f"sample should have a not null messages {attr}.")
    attr_val = getattr(sample, attr)
    if attr_val is None or not isinstance(attr_val, str):
        raise ValueError(f"sample.{attr} should be a list.")
    if attr_val == "":
        raise ValueError(f"sample.{attr} should not be empty.")

def validate_sample_schema(sample: Sample) -> Sample:
    """Ensure core fields are of expected container types."""
    for attr in ["messages"]:
        _validate_list_field(sample, attr)
    for attr in ["schema_version", "id" ]:
        _validate_str_field(sample, attr)
    return sample


def build_validator(config: Optional[Dict[str, Any]]) -> Optional[SampleValidator]:
    """Creates a SampleValidator from DatasetSpec.schema or the default envelope model."""

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
