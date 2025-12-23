from __future__ import annotations

import json
from pathlib import Path

from gage_eval.assets.datasets.preprocessors.builtin import SwebenchProStandardizer


def test_swebench_pro_standardizer_filters_smoke(tmp_path: Path):
    smoke_file = tmp_path / "ids.txt"
    smoke_file.write_text("keep-me\n", encoding="utf-8")
    pre = SwebenchProStandardizer(smoke_ids_path=str(smoke_file))
    # not in smoke -> filtered out
    assert pre.transform({"instance_id": "drop-me"}) is None
    # in smoke -> kept
    sample = {
        "instance_id": "keep-me",
        "problem_statement": "Fix bug",
        "requirements": "Do X",
        "interface": "Call Y",
        "fail_to_pass": '["a"]',
        "pass_to_pass": [],
        "selected_test_files_to_run": ["t1"],
        "base_commit": "abc",
        "repo": "org/repo",
        "test_patch": "diff --git ...",
        "before_repo_set_cmd": "echo hi",
    }
    out = pre.transform(
        sample,
        dataset_id="ds",
        dataset_metadata={"path": "p"},
    )
    assert out == sample["inputs"]
    assert sample["id"] == "keep-me"
    assert sample["_dataset_id"] == "ds"
    assert sample["_dataset_metadata"] == {"path": "p"}
    assert sample["metadata"]["base_commit"] == "abc"
    assert sample["metadata"]["fail_to_pass"] == ["a"]
    assert sample["metadata"]["selected_test_files_to_run"] == ["t1"]
    assert sample["inputs"]["prompt"].startswith("Fix bug")
    assert sample["messages"][0]["content"][0]["text"].startswith("Fix bug")


def test_swebench_pro_standardizer_parses_lists():
    pre = SwebenchProStandardizer()
    sample = {
        "instance_id": "i1",
        "problem_statement": "p",
        "fail_to_pass": "[\"x\", \"y\"]",
        "pass_to_pass": "[]",
        "selected_test_files_to_run": "[\"t\"]",
        "base_commit": "",
        "repo": "",
    }
    out = pre.transform(sample)
    assert out == sample["inputs"]
    assert sample["metadata"]["fail_to_pass"] == ["x", "y"]
    assert sample["metadata"]["pass_to_pass"] == []
    assert sample["metadata"]["selected_test_files_to_run"] == ["t"]


def test_swebench_pro_standardizer_tutanota_hotfix():
    pre = SwebenchProStandardizer()
    sample = {
        "instance_id": "tutao__tutanota-de49d486feef842101506adf040a0f00ded59519",
        "problem_statement": "p",
        "pass_to_pass": '["test/api/Suite.ts | api tests (3065 assertions)"]',
    }
    pre.transform(sample)
    assert sample["metadata"]["pass_to_pass"] == ["test/api/Suite.ts | api tests (3064 assertions)"]
