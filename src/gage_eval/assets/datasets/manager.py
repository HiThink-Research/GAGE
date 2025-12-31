"""Data management utilities shared by all pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.assets.datasets.validation import SampleValidator, build_validator
from gage_eval.assets.datasets.utils.multimodal import merge_multimodal_inputs
from gage_eval.assets.datasets.sample import (
    Sample,
    Message,
    MessageContent
)
from dataclasses import dataclass, fields, asdict, is_dataclass

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

    def iter_samples(self, dataset_id: str, trace: Optional[ObservabilityTrace] = None) -> Iterator[Dict[str, Any]]:
        """Yield normalized samples for the requested dataset.

        Args:
            dataset_id: Identifier passed to :meth:`register_source`.
            trace: Optional observability sink used to emit per-sample events.
        """

        logger.debug("Iterating samples for dataset_id='{}'", dataset_id)
        source = self.get(dataset_id)
        validator = source.validator
        for index, record in enumerate(source.records):
            if not isinstance(record, Sample):
                logger.warning(
                        "Skipping non-dict record dataset='%s' index=%s type=%s",
                        dataset_id,
                        index,
                        type(record).__name__,
                )
                continue
            normalized = record
            if validator:
                validated = validator.validate_raw(record, dataset_id=dataset_id, index=index, trace=trace)
                if validated is None:
                    continue
                normalized = validated
            if trace:
                trace.emit(
                    "data_sample_emitted",
                    {
                        "dataset_id": dataset_id,
                        "index": index,
                        "keys": list(asdict(normalized).keys() if is_dataclass(normalized) else normalized.keys()),
                    },
                )
            logger.debug("Emitted normalized sample idx={} keys={}", index, list(asdict(normalized).keys() if is_dataclass(normalized) else normalized.keys()))
            # Convert Sample dataclass to dict so backends can access fields via .get()
            yield asdict(normalized) if is_dataclass(normalized) else normalized
