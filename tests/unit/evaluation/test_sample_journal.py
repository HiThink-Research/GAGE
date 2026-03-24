from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.evaluation.sample_journal import LockedJsonlJournal


def _serialize_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


@pytest.mark.io
def test_locked_jsonl_journal_writes_complete_lines(tmp_path: Path) -> None:
    journal = LockedJsonlJournal(
        tmp_path / "samples.jsonl",
        serializer=_serialize_payload,
    )

    journal.append({"sample_id": "s1", "path": tmp_path})
    journal.append({"sample_id": "s2", "value": 2})

    lines = (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["sample_id"] == "s1"
    assert json.loads(lines[0])["path"] == str(tmp_path)
    assert json.loads(lines[1])["sample_id"] == "s2"


@pytest.mark.io
def test_locked_jsonl_journal_rejects_appends_after_close(tmp_path: Path) -> None:
    journal = LockedJsonlJournal(
        tmp_path / "samples.jsonl",
        serializer=_serialize_payload,
    )

    journal.append({"sample_id": "s1"})
    journal.close()

    with pytest.raises(RuntimeError, match="already closed"):
        journal.append({"sample_id": "s2"})
