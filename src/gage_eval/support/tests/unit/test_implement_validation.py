from pathlib import Path

import pytest

from gage_eval.support.pipeline import _validate_agent_files, _assert_metric_class_paths


def test_validate_rejects_function_metric() -> None:
    files = {
        Path("src/gage_eval/metrics/builtin/foo_metric.py"): "@registry.asset('metrics','foo',desc='')\ndef foo():\n    pass\n"
    }
    with pytest.raises(RuntimeError):
        _validate_agent_files(files)


def test_validate_rejects_non_class_preprocessor() -> None:
    files = {
        Path("src/gage_eval/assets/datasets/preprocessors/foo_preprocessor.py"): "# missing class\nprint('oops')\n"
    }
    with pytest.raises(RuntimeError):
        _validate_agent_files(files)


def test_validate_passes_when_class_present() -> None:
    files = {
        Path("src/gage_eval/metrics/builtin/foo_metric.py"): "class FooMetric:\n    pass\n",
        Path("src/gage_eval/assets/datasets/preprocessors/foo_preprocessor.py"): "class FooPreprocessor:\n    pass\n",
    }
    _validate_agent_files(files)


def test_guard_metric_class_path_missing_module(tmp_path: Path) -> None:
    cfg_block = {
        "metrics": [
            {"metric_id": "m1", "implementation": "nonexistent.module:Cls", "aggregation": "mean"},
        ]
    }
    with pytest.raises(RuntimeError):
        _assert_metric_class_paths(cfg_block, tmp_path)


def test_guard_metric_class_path_existing_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a fake module path on disk to satisfy import check.
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "mod.py").write_text("class Foo: pass\n", encoding="utf-8")
    cfg_block = {
        "metrics": [
            {"metric_id": "m1", "implementation": "pkg.mod:Foo", "aggregation": "mean"},
        ]
    }
    # Should pass because module path exists on disk.
    _assert_metric_class_paths(cfg_block, tmp_path)


def test_guard_metric_class_path_generated_allows(tmp_path: Path) -> None:
    cfg_block = {
        "metrics": [
            {"metric_id": "m1", "implementation": "pkg.mod:Foo", "aggregation": "mean"},
        ]
    }
    generated_files = {Path("pkg/mod.py"): "class Foo: pass\n"}
    _assert_metric_class_paths(cfg_block, tmp_path, generated_files=generated_files)
    cfg_block = {
        "metrics": [
            {"metric_id": "m1", "implementation": "pkg.mod:Foo", "aggregation": "mean"},
        ]
    }
    with pytest.raises(RuntimeError):
        _assert_metric_class_paths(cfg_block, tmp_path)
