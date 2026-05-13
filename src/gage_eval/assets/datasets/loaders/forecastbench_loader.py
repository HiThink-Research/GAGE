"""ForecastBench dataset loader (question_set + resolution_set JSON join)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.hubs.base import DatasetHubHandle
from gage_eval.assets.datasets.loaders.base import DatasetLoader
from gage_eval.assets.datasets.loaders.loader_utils import (
    apply_default_params,
    apply_preprocess,
    build_preprocess_context,
    resolve_doc_to_callable,
)
from gage_eval.assets.datasets.manager import DataSource
from gage_eval.registry import registry


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("questions", "resolutions", "data", "items", "records"):
            inner = data.get(key)
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
    return []


def _norm_source(value: Any) -> str:
    return str(value or "").strip().lower()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (1,):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _join_records(
    questions: Sequence[Mapping[str, Any]],
    resolutions: Sequence[Mapping[str, Any]],
    *,
    question_set_name: str,
) -> List[Dict[str, Any]]:
    res_by_id: Dict[str, Dict[str, Any]] = {}
    for row in resolutions:
        rid = row.get("id")
        if rid is None:
            continue
        res_by_id[str(rid)] = dict(row)

    joined: List[Dict[str, Any]] = []
    for q in questions:
        qid = q.get("id")
        if qid is None:
            continue
        key = str(qid)
        if key not in res_by_id:
            continue
        r = res_by_id[key]
        merged: Dict[str, Any] = {**dict(q), **dict(r)}
        merged["raw_question"] = dict(q)
        merged["raw_resolution"] = dict(r)
        merged.setdefault("question_set", question_set_name)
        joined.append(merged)
    return joined


def _filter_joined(
    rows: Sequence[Dict[str, Any]],
    *,
    source_filter: Sequence[str],
    resolved_only: bool,
) -> List[Dict[str, Any]]:
    allowed = {_norm_source(s) for s in source_filter if s}
    out: List[Dict[str, Any]] = []
    for row in rows:
        if allowed and _norm_source(row.get("source")) not in allowed:
            continue
        if resolved_only:
            if not _coerce_bool(row.get("resolved")):
                continue
            if row.get("resolved_to") is None:
                continue
        out.append(row)
    return out


def _stable_sort_ids(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: str(r.get("id", "")))


def _tag_raw_records(
    records: Iterable[Dict[str, Any]],
    spec: DatasetSpec,
    *,
    data_path: Optional[str],
) -> Iterator[Dict[str, Any]]:
    for record in records:
        tagged = dict(record)
        tagged.setdefault("_dataset_id", spec.dataset_id)
        if data_path and "_dataset_metadata" not in tagged:
            tagged["_dataset_metadata"] = {"path": data_path}
        yield tagged


@registry.asset(
    "dataset_loaders",
    "forecastbench",
    desc="ForecastBench loader (join question_set.json + resolution_set.json)",
    tags=("forecastbench", "json", "local"),
    supports_streaming=False,
)
class ForecastBenchDatasetLoader(DatasetLoader):
    """Load ForecastBench-style JSON and join questions to resolutions by ``id``.

    Defaults align with ForecastBench P0: ``resolved_only`` defaults to ``True`` (only settled
    rows). When ``params.source_filter`` is omitted, only ``polymarket`` sources are kept; pass
    an empty list ``[]`` to disable source filtering.
    """

    def load(self, hub_handle: Optional[DatasetHubHandle], *, trace=None) -> DataSource:
        params = self.spec.params or {}
        question_path = params.get("question_set_path")
        resolution_path = params.get("resolution_set_path")
        if not question_path or not resolution_path:
            raise ValueError(
                f"Dataset '{self.spec.dataset_id}' requires params.question_set_path "
                "and params.resolution_set_path"
            )

        qpath = Path(str(question_path)).expanduser()
        rpath = Path(str(resolution_path)).expanduser()
        if not qpath.is_file():
            raise FileNotFoundError(f"ForecastBench question set not found: {qpath}")
        if not rpath.is_file():
            raise FileNotFoundError(f"ForecastBench resolution set not found: {rpath}")

        questions = _load_json_records(qpath)
        resolutions = _load_json_records(rpath)
        question_set_name = qpath.name

        joined = _join_records(questions, resolutions, question_set_name=question_set_name)
        raw_source_filter = params.get("source_filter")
        if raw_source_filter is None:
            source_filter: Sequence[str] = ("polymarket",)
        elif isinstance(raw_source_filter, str):
            source_filter = (raw_source_filter,)
        else:
            source_filter = tuple(str(x) for x in raw_source_filter)
        resolved_only = bool(params.get("resolved_only", True))
        filtered = _filter_joined(
            joined,
            source_filter=source_filter,
            resolved_only=resolved_only,
        )
        ordered = _stable_sort_ids(filtered)
        max_samples = params.get("max_samples")
        if max_samples is not None:
            try:
                cap = int(max_samples)
            except (TypeError, ValueError):
                cap = 0
            if cap > 0:
                ordered = ordered[:cap]

        doc_to_text = resolve_doc_to_callable(self.spec, "doc_to_text")
        doc_to_visual = resolve_doc_to_callable(self.spec, "doc_to_visual")
        doc_to_audio = resolve_doc_to_callable(self.spec, "doc_to_audio")
        data_path = str(qpath)
        preprocess_ctx = build_preprocess_context(
            self.spec,
            data_path=data_path,
            registry_lookup=self.registry_lookup,
            allow_lazy_import=self.allow_asset_lazy_import,
        )
        if preprocess_ctx:
            raw_iter: Iterable[Dict[str, Any]] = ordered
            records = apply_preprocess(
                raw_iter,
                self.spec,
                data_path=data_path,
                registry_lookup=self.registry_lookup,
                allow_lazy_import=self.allow_asset_lazy_import,
                doc_to_text=doc_to_text,
                doc_to_visual=doc_to_visual,
                doc_to_audio=doc_to_audio,
                trace=trace,
            )
        else:
            records = _tag_raw_records(ordered, self.spec, data_path=data_path)

        records = apply_default_params(records, self.spec)
        records = list(records)

        metadata = {
            "loader": "forecastbench",
            "question_set_path": str(qpath),
            "resolution_set_path": str(rpath),
            "question_set": question_set_name,
            "streaming": False,
        }

        return DataSource(
            dataset_id=self.spec.dataset_id,
            records=records,
            doc_to_text=None,
            doc_to_visual=None,
            doc_to_audio=None,
            metadata=metadata,
            validation=self.spec.schema,
            streaming=False,
        )
