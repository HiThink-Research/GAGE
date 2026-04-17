from __future__ import annotations

import random

import pytest

from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


def _trace(run_id: str) -> ObservabilityTrace:
    return ObservabilityTrace(recorder=InMemoryRecorder(run_id=run_id), run_id=run_id)


@pytest.mark.fast
def test_sample_loop_env_max_samples_zero_is_treated_as_unbounded(monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_MAX_SAMPLES", "0")
    samples = [{"id": f"s{idx}"} for idx in range(3)]
    loop = SampleLoop(samples)

    selected = [sample["id"] for _, sample in loop._iter_samples(_trace("max-samples-zero"))]

    assert selected == ["s0", "s1", "s2"]


@pytest.mark.fast
def test_sample_loop_auto_shuffle_selects_reservoir_for_streaming_max_samples() -> None:
    loop = SampleLoop(
        ({"id": f"s{idx}"} for idx in range(10)),
        shuffle=True,
        shuffle_strategy="auto",
        shuffle_seed=7,
        max_samples=3,
        streaming=True,
    )

    selected = [sample["id"] for _, sample in loop._iter_samples(_trace("shuffle-reservoir"))]

    assert len(selected) == 3
    assert selected != ["s0", "s1", "s2"]
    assert loop.shuffle_summary["resolved"] == "reservoir"
    assert loop.shuffle_summary["reason"] == "max_samples_present"


@pytest.mark.fast
def test_sample_loop_auto_shuffle_selects_in_memory_for_small_dataset() -> None:
    samples = [{"id": f"s{idx}"} for idx in range(5)]
    loop = SampleLoop(
        samples,
        shuffle=True,
        shuffle_strategy="auto",
        shuffle_seed=11,
        shuffle_small_dataset_threshold=10,
    )

    selected = [sample["id"] for _, sample in loop._iter_samples(_trace("shuffle-in-memory"))]
    expected_indices = list(range(len(samples)))
    random.Random(11).shuffle(expected_indices)

    assert loop.shuffle_summary["resolved"] == "in_memory"
    assert selected == [samples[idx]["id"] for idx in expected_indices]


@pytest.mark.fast
def test_sample_loop_external_index_keeps_artifacts_when_requested(tmp_path) -> None:
    artifact_root = tmp_path / "shuffle-artifacts"
    loop = SampleLoop(
        ({"id": f"s{idx}"} for idx in range(6)),
        shuffle=True,
        shuffle_strategy="external_index",
        shuffle_seed=5,
        shuffle_small_dataset_threshold=2,
        streaming=True,
        keep_shuffle_artifacts=True,
        shuffle_artifact_root=artifact_root,
    )

    selected = [sample["id"] for _, sample in loop._iter_samples(_trace("shuffle-external"))]

    assert loop.shuffle_summary["resolved"] == "external_index"
    assert loop.shuffle_summary["artifact_root"] == str(artifact_root)
    assert sorted(selected) == [f"s{idx}" for idx in range(6)]
    assert artifact_root.exists()
