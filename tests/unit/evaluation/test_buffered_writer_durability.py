from __future__ import annotations

import pytest

from gage_eval.evaluation.buffered_writer import BufferedResultWriter


def _sample_payload(idx: int) -> dict:
    return {"sample_id": f"s{idx}", "value": idx}


@pytest.mark.fast
def test_always_policy_fsyncs_pending_and_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GAGE_EVAL_BUFFER_DURABILITY_POLICY", "always")
    calls: list[int] = []
    monkeypatch.setattr("gage_eval.evaluation.buffered_writer.os.fsync", lambda fd: calls.append(fd))
    writer = BufferedResultWriter(tmp_path / "always.jsonl", max_batch_size=64, flush_interval_s=60.0)

    writer.record(_sample_payload(1))
    writer.flush()

    assert writer.flush_count == 1
    assert writer.fsync_count == 2
    assert len(calls) == 2


@pytest.mark.fast
def test_interval_policy_defers_fsync_until_close(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GAGE_EVAL_BUFFER_DURABILITY_POLICY", "interval")
    monkeypatch.setenv("GAGE_EVAL_BUFFER_FSYNC_EVERY_FLUSHES", "99")
    monkeypatch.setenv("GAGE_EVAL_BUFFER_FSYNC_EVERY_S", "999")
    calls: list[int] = []
    monkeypatch.setattr("gage_eval.evaluation.buffered_writer.os.fsync", lambda fd: calls.append(fd))
    writer = BufferedResultWriter(tmp_path / "interval.jsonl", max_batch_size=64, flush_interval_s=60.0)

    writer.record(_sample_payload(1))
    writer.flush()

    assert writer.flush_count == 1
    assert writer.fsync_count == 0

    writer.close()

    assert writer.fsync_count == 1
    assert len(calls) == 1


@pytest.mark.fast
def test_never_policy_skips_fsync_even_on_close(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GAGE_EVAL_BUFFER_DURABILITY_POLICY", "never")
    calls: list[int] = []
    monkeypatch.setattr("gage_eval.evaluation.buffered_writer.os.fsync", lambda fd: calls.append(fd))
    writer = BufferedResultWriter(tmp_path / "never.jsonl", max_batch_size=64, flush_interval_s=60.0)

    writer.record(_sample_payload(1))
    writer.flush()
    writer.close()

    assert writer.flush_count == 1
    assert writer.fsync_count == 0
    assert calls == []
