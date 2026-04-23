from __future__ import annotations

import pytest

from gage_eval.assets.datasets.preprocessors.swebench_pro_preprocessor import SwebenchProPreprocessor
from gage_eval.registry import registry
from gage_eval.observability.config import ObservabilityConfig
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


@pytest.mark.fast
def test_swebench_preprocessor_injects_sandbox() -> None:
    preprocessor = SwebenchProPreprocessor(
        dockerhub_username="tester",
        registry_prefix="registry.local/swebench",
    )
    record = {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc",
        "problem_statement": "Fix bug",
        "fail_to_pass": [],
        "pass_to_pass": [],
        "selected_test_files_to_run": [],
    }
    sample = preprocessor.transform(record)

    assert sample.sandbox is not None
    assert sample.sandbox["sandbox_id"] == "swebench_runtime"
    assert sample.sandbox["runtime"] == "docker"
    assert sample.sandbox["lifecycle"] == "per_sample"
    assert "image" in sample.sandbox
    assert sample.metadata["image_uri"].startswith("registry.local/swebench/")


@pytest.mark.fast
def test_swebench_preprocessor_preserves_sandbox_overrides() -> None:
    preprocessor = SwebenchProPreprocessor()
    record = {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc",
        "problem_statement": "Fix bug",
        "fail_to_pass": [],
        "pass_to_pass": [],
        "selected_test_files_to_run": [],
        "sandbox": {"sandbox_id": "custom", "runtime": "remote"},
    }
    sample = preprocessor.transform(record)

    assert sample.sandbox is not None
    assert sample.sandbox["sandbox_id"] == "custom"
    assert sample.sandbox["runtime"] == "remote"


@pytest.mark.fast
def test_swebench_preprocessor_formats_prompt_sections() -> None:
    preprocessor = SwebenchProPreprocessor()
    record = {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc",
        "problem_statement": "Fix the bug in the search endpoint.",
        "requirements": "Return HTTP 400 on invalid inputs.",
        "interface": "The API accepts GET /search?q=term.",
        "fail_to_pass": [],
        "pass_to_pass": [],
        "selected_test_files_to_run": [],
    }
    sample = preprocessor.transform(record)

    text = sample.messages[0].content[0].text

    assert text is not None
    assert "## Problem Statement" in text
    assert "## Requirements" in text
    assert "You MUST adhere to the following requirements" in text
    assert "## Interface Specification" in text


@pytest.mark.fast
def test_swebench_preprocessor_coerces_list_strings() -> None:
    preprocessor = SwebenchProPreprocessor()
    record = {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc",
        "problem_statement": "Fix bug",
        "fail_to_pass": "['test/foo', 'test/bar']",
        "pass_to_pass": '["test/baz"]',
        "selected_test_files_to_run": [],
        "test_patch": "diff --git a/a b/a",
        "patch": "diff --git a/a b/a",
    }
    sample = preprocessor.transform(record)

    assert sample.metadata["fail_to_pass"] == ["test/foo", "test/bar"]
    assert sample.metadata["pass_to_pass"] == ["test/baz"]
    assert "test_patch" not in sample.metadata
    assert "gold_patch" not in sample.metadata
    assert "patch" not in sample.metadata


@pytest.mark.fast
def test_swebench_preprocessor_projects_alias_prompt_fields() -> None:
    preprocessor = SwebenchProPreprocessor()
    record = {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc",
        "description": "Fix the search bug.",
        "requirements": "Keep the API stable.",
        "instructions": "Use the existing endpoint contract.",
        "fail_to_pass": [],
        "pass_to_pass": [],
        "selected_test_files_to_run": [],
    }

    sample = preprocessor.transform(record)

    assert sample.instruction == sample.prompt
    assert sample.prompt == sample.inputs["prompt"]
    assert sample.messages[0].content[0].text == sample.prompt
    assert sample.metadata["prompt_source"] == "description"
    assert sample.metadata["prompt_present"] is True
    assert sample.metadata["prompt_length_chars"] == len(sample.prompt)


@pytest.mark.fast
def test_swebench_preprocessor_fails_fast_on_missing_problem_statement() -> None:
    preprocessor = SwebenchProPreprocessor()
    record = {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc",
        "fail_to_pass": [],
        "pass_to_pass": [],
        "selected_test_files_to_run": [],
    }

    with pytest.raises(ValueError, match="input_projection\\.missing_problem_statement"):
        preprocessor.transform(record)


@pytest.mark.fast
def test_swebench_preprocessor_can_skip_missing_problem_statement_with_trace() -> None:
    preprocessor = SwebenchProPreprocessor(on_error="skip")
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="swebench-preprocess"))
    cfg = ObservabilityConfig(enabled=True)
    record = {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc",
        "fail_to_pass": [],
        "pass_to_pass": [],
        "selected_test_files_to_run": [],
    }

    sample = preprocessor.transform(record, trace=trace, observability_config=cfg)

    assert sample is None
    events = [event["event"] for event in trace.events]
    assert "preprocess_start" in events
    assert "preprocess_error" in events


@pytest.mark.fast
def test_swebench_preprocessor_applies_tutanota_hotfix_for_instance_prefix() -> None:
    preprocessor = SwebenchProPreprocessor()
    record = {
        "instance_id": "instance_tutao__tutanota-123",
        "repo": "tutao/tutanota",
        "base_commit": "abc",
        "problem_statement": "Fix bug",
        "fail_to_pass": [],
        "pass_to_pass": ["test/api/Suite.ts -> (3065 assertions)"],
        "selected_test_files_to_run": [],
    }

    sample = preprocessor.transform(record)

    assert sample.metadata["pass_to_pass"] == ["test/api/Suite.ts -> (3064 assertions)"]
    assert sample.metadata["prompt_source"] == "problem_statement"


@pytest.mark.fast
def test_swebench_standardizer_registered() -> None:
    registry.auto_discover("dataset_preprocessors", "gage_eval.assets.datasets.preprocessors.builtin")

    preprocessor_cls = registry.get("dataset_preprocessors", "swebench_pro_standardizer")

    assert issubclass(preprocessor_cls, SwebenchProPreprocessor)
