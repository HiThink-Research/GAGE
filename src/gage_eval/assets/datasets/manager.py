"""Data management utilities shared by all pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.assets.datasets.validation import (
    SampleValidator,
    ValidationFailure,
    build_validator,
)
from gage_eval.assets.datasets.sample import (
    Sample,
    sample_from_dict,
)


@dataclass
class DataSource:
    """Represents a reusable dataset along with optional doc_to hooks."""

    dataset_id: str
    records: Iterable[Sample]
    doc_to_text: Optional[Callable[[Dict[str, Any]], str]] = None
    doc_to_visual: Optional[Callable[[Dict[str, Any]], Any]] = None
    doc_to_audio: Optional[Callable[[Dict[str, Any]], Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    validator: Optional[SampleValidator] = field(default=None, repr=False)
    streaming: bool = False


class DataManager:
    """Central entry point for dataset loading, caching and schema mapping.

    The implementation borrows ideas from LMMS' ``doc_to_*`` adapter
    pattern as well as llm-eval's ``data_server.py`` queue-based design.
    For now the class keeps things simple: datasets are assumed to be
    in-memory iterables. Future iterations can mount streaming sources
    or integrate HuggingFace datasets by extending ``register_loader``.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, DataSource] = {}

    def register_source(self, source: DataSource, trace: Optional[ObservabilityTrace] = None) -> None:
        if source.dataset_id in self._sources:
            raise ValueError(f"Dataset '{source.dataset_id}' is already registered")
        source.validator = build_validator(source.validation)
        self._sources[source.dataset_id] = source
        metadata = source.metadata or {}
        formatted_meta = ", ".join(f"{key}={value}" for key, value in metadata.items()) or "no-metadata"
        logger.info("Registered dataset '{}' ({})", source.dataset_id, formatted_meta)
        if trace:
            trace.emit(
                "dataset_registered",
                {
                    "dataset_id": source.dataset_id,
                    "metadata": metadata,
                },
            )

    def get(self, dataset_id: str) -> DataSource:
        try:
            return self._sources[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Dataset '{dataset_id}' is not registered") from exc

    def iter_samples(
        self,
        dataset_id: str,
        trace: Optional[ObservabilityTrace] = None,
        *,
        record_seen: Optional[Callable[[int], None]] = None,
        validation_reporter: Optional[Callable[[ValidationFailure], None]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield normalized samples for the requested dataset.

        Args:
            dataset_id: Identifier passed to :meth:`register_source`.
            trace: Optional observability sink used to emit per-sample events.
        """

        logger.debug("Iterating samples for dataset_id='{}'", dataset_id)
        source = self.get(dataset_id)
        validator = source.validator
        for index, record in enumerate(source.records):
            if record_seen is not None:
                record_seen(index)
            if not isinstance(record, Sample):
                if isinstance(record, dict):
                    try:
                        record = sample_from_dict(record)
                    except Exception as exc:
                        message = (
                            "Failed to convert dict record to Sample "
                            f"dataset='{dataset_id}' index={index}: {exc}"
                        )
                        logger.warning(message)
                        _report_ingress_failure(
                            validation_reporter,
                            step="raw_record",
                            dataset_id=dataset_id,
                            index=index,
                            reason="raw_record:deserialization_failed",
                            message=message,
                        )
                        continue
                else:
                    message = (
                        "Skipping non-dict record "
                        f"dataset='{dataset_id}' index={index} "
                        f"type={type(record).__name__}"
                    )
                    logger.warning(message)
                    _report_ingress_failure(
                        validation_reporter,
                        step="raw_record",
                        dataset_id=dataset_id,
                        index=index,
                        reason="raw_record:invalid_record_type",
                        message=message,
                    )
                    continue
            normalized = record
            if validator:
                validated = validator.validate_raw(
                    record,
                    dataset_id=dataset_id,
                    index=index,
                    trace=trace,
                    on_failure=validation_reporter,
                )
                if validated is None:
                    continue
                normalized = validated
            normalized_payload = (
                asdict(normalized) if is_dataclass(normalized) else dict(normalized)
            )
            normalized_payload.setdefault("_gage_source_index", index)
            normalized_payload.setdefault("_gage_dataset_id", dataset_id)
            if trace:
                trace.emit(
                    "data_sample_emitted",
                    {
                        "dataset_id": dataset_id,
                        "index": index,
                        "keys": list(normalized_payload.keys()),
                    },
                )
            logger.debug(
                "Emitted normalized sample idx={} keys={}",
                index,
                list(normalized_payload.keys()),
            )
            yield normalized_payload


def _report_ingress_failure(
    reporter: Optional[Callable[[ValidationFailure], None]],
    *,
    step: str,
    dataset_id: str,
    index: int,
    reason: str,
    message: str,
) -> None:
    """Emit a synthetic validation failure for non-validator drop paths."""

    if reporter is None:
        return
    reporter(
        ValidationFailure(
            step=step,
            dataset_id=dataset_id,
            message=message,
            reason=reason,
            index=index,
        )
    )
