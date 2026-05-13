"""Tests for ForecastBench dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.assets.datasets.loaders.forecastbench_loader import ForecastBenchDatasetLoader
from gage_eval.config.pipeline_config import DatasetSpec


def _forecastbench_fixture_dir() -> Path:
    """Resolve ``tests/fixtures/forecastbench`` by walking up from this file (any test depth)."""

    here = Path(__file__).resolve()
    marker = Path("tests") / "fixtures" / "forecastbench" / "smoke_question_set.json"
    for parent in [here.parent] + list(here.parents):
        candidate = parent / marker
        if candidate.is_file():
            return candidate.parent
    raise FileNotFoundError(f"Could not locate ForecastBench fixtures under parents of {here}")


FIXTURE_DIR = _forecastbench_fixture_dir()
QUESTION_PATH = FIXTURE_DIR / "smoke_question_set.json"
RESOLUTION_PATH = FIXTURE_DIR / "smoke_resolution_set.json"


def _write_pair(tmp_path: Path, questions: list, resolutions: list) -> tuple[Path, Path]:
    qpath = tmp_path / "q.json"
    rpath = tmp_path / "r.json"
    qpath.write_text(json.dumps(questions), encoding="utf-8")
    rpath.write_text(json.dumps(resolutions), encoding="utf-8")
    return qpath, rpath


def test_forecastbench_loader_polymarket_resolved_only() -> None:
    spec = DatasetSpec(
        dataset_id="fb_loader_p0",
        loader="forecastbench",
        hub="inline",
        params={
            "question_set_path": str(QUESTION_PATH),
            "resolution_set_path": str(RESOLUTION_PATH),
            "source_filter": ["polymarket"],
            "resolved_only": True,
        },
    )
    loader = ForecastBenchDatasetLoader(spec=spec)
    ds = loader.load(hub_handle=None)
    records = list(ds.records)
    assert len(records) == 4
    for row in records:
        assert isinstance(row, dict)
        assert row["source"] == "polymarket"
        assert row["resolved"] is True
        assert row["resolved_to"] is not None
        assert "forecast_due_date" in row
        assert "question_set" in row
    ids = {str(r["id"]) for r in records}
    assert ids == {"pm-join-1", "pm-a", "pm-b", "pm-c"}


def test_forecastbench_loader_join_by_id_ignores_resolution_date(tmp_path: Path) -> None:
    questions = [
        {
            "id": "edge-1",
            "source": "polymarket",
            "question": "Q?",
            "resolution_criteria": "rc",
            "background": "bg",
            "freeze_datetime": "2025-02-20T00:00:00+00:00",
            "forecast_due_date": "2025-03-02",
        }
    ]
    resolutions = [
        {"id": "edge-1", "resolved": True, "resolved_to": 1.0, "resolution_date": None},
    ]
    qpath, rpath = _write_pair(tmp_path, questions, resolutions)
    spec = DatasetSpec(
        dataset_id="fb_edge",
        loader="forecastbench",
        hub="inline",
        params={
            "question_set_path": str(qpath),
            "resolution_set_path": str(rpath),
            "source_filter": ["polymarket"],
            "resolved_only": True,
        },
    )
    loader = ForecastBenchDatasetLoader(spec=spec)
    ds = loader.load(hub_handle=None)
    rows = list(ds.records)
    assert len(rows) == 1
    assert rows[0]["resolved_to"] == 1.0


def test_forecastbench_loader_max_samples_stable(tmp_path: Path) -> None:
    questions = []
    resolutions = []
    for letter in ("z", "y", "x"):
        qid = f"pm-{letter}"
        questions.append(
            {
                "id": qid,
                "source": "polymarket",
                "question": f"Q {letter}",
                "resolution_criteria": "rc",
                "background": "bg",
                "freeze_datetime": "2025-02-20T00:00:00+00:00",
                "forecast_due_date": "2025-03-02",
            }
        )
        resolutions.append({"id": qid, "resolved": True, "resolved_to": 0.0, "resolution_date": None})
    qpath, rpath = _write_pair(tmp_path, questions, resolutions)
    spec = DatasetSpec(
        dataset_id="fb_max",
        loader="forecastbench",
        hub="inline",
        params={
            "question_set_path": str(qpath),
            "resolution_set_path": str(rpath),
            "source_filter": ["polymarket"],
            "resolved_only": True,
            "max_samples": 2,
        },
    )
    loader = ForecastBenchDatasetLoader(spec=spec)
    ds = loader.load(hub_handle=None)
    rows = list(ds.records)
    assert len(rows) == 2
    assert [r["id"] for r in rows] == ["pm-x", "pm-y"]


def test_forecastbench_loader_requires_paths() -> None:
    spec = DatasetSpec(
        dataset_id="fb_missing",
        loader="forecastbench",
        hub="inline",
        params={},
    )
    loader = ForecastBenchDatasetLoader(spec=spec)
    with pytest.raises(ValueError):
        loader.load(hub_handle=None)


def test_forecastbench_loader_resolved_only_defaults_true(tmp_path: Path) -> None:
    questions = [
        {
            "id": "pm-1",
            "source": "polymarket",
            "question": "Q",
            "resolution_criteria": "r",
            "background": "b",
            "freeze_datetime": "2025-02-20T00:00:00+00:00",
            "forecast_due_date": "2025-03-02",
        },
        {
            "id": "pm-2",
            "source": "polymarket",
            "question": "Q2",
            "resolution_criteria": "r",
            "background": "b",
            "freeze_datetime": "2025-02-20T00:00:00+00:00",
            "forecast_due_date": "2025-03-02",
        },
    ]
    resolutions = [
        {"id": "pm-1", "resolved": True, "resolved_to": 1.0},
        {"id": "pm-2", "resolved": False, "resolved_to": None},
    ]
    qpath, rpath = _write_pair(tmp_path, questions, resolutions)
    spec = DatasetSpec(
        dataset_id="fb_default_resolved",
        loader="forecastbench",
        hub="inline",
        params={
            "question_set_path": str(qpath),
            "resolution_set_path": str(rpath),
            "source_filter": ["polymarket"],
        },
    )
    loader = ForecastBenchDatasetLoader(spec=spec)
    rows = list(loader.load(hub_handle=None).records)
    assert len(rows) == 1
    assert rows[0]["id"] == "pm-1"


def test_forecastbench_loader_empty_source_filter_allows_all_sources(tmp_path: Path) -> None:
    questions = [
        {
            "id": "a1",
            "source": "polymarket",
            "question": "Q",
            "resolution_criteria": "r",
            "background": "b",
            "freeze_datetime": "2025-02-20T00:00:00+00:00",
            "forecast_due_date": "2025-03-02",
        },
        {
            "id": "b1",
            "source": "metaculus",
            "question": "Q2",
            "resolution_criteria": "r",
            "background": "b",
            "freeze_datetime": "2025-02-20T00:00:00+00:00",
            "forecast_due_date": "2025-03-02",
        },
    ]
    resolutions = [
        {"id": "a1", "resolved": True, "resolved_to": 0.0},
        {"id": "b1", "resolved": True, "resolved_to": 1.0},
    ]
    qpath, rpath = _write_pair(tmp_path, questions, resolutions)
    spec = DatasetSpec(
        dataset_id="fb_no_source_filter",
        loader="forecastbench",
        hub="inline",
        params={
            "question_set_path": str(qpath),
            "resolution_set_path": str(rpath),
            "source_filter": [],
            "resolved_only": True,
        },
    )
    loader = ForecastBenchDatasetLoader(spec=spec)
    rows = list(loader.load(hub_handle=None).records)
    assert len(rows) == 2
    sources = {str(r["source"]) for r in rows}
    assert sources == {"polymarket", "metaculus"}


def test_forecastbench_loader_accepts_string_resolved_on(tmp_path: Path) -> None:
    questions = [
        {
            "id": "pm-on",
            "source": "polymarket",
            "question": "Q",
            "resolution_criteria": "r",
            "background": "b",
            "freeze_datetime": "2025-02-20T00:00:00+00:00",
            "forecast_due_date": "2025-03-02",
        },
    ]
    resolutions = [{"id": "pm-on", "resolved": "on", "resolved_to": 0.0}]
    qpath, rpath = _write_pair(tmp_path, questions, resolutions)
    spec = DatasetSpec(
        dataset_id="fb_on",
        loader="forecastbench",
        hub="inline",
        params={
            "question_set_path": str(qpath),
            "resolution_set_path": str(rpath),
            "source_filter": ["polymarket"],
            "resolved_only": True,
        },
    )
    rows = list(ForecastBenchDatasetLoader(spec=spec).load(hub_handle=None).records)
    assert len(rows) == 1
    assert rows[0]["id"] == "pm-on"
