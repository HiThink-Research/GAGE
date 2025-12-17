from gage_eval.support.pipeline import _normalize_support_config_run_commands


def test_append_max_samples_when_missing() -> None:
    md = """
```yaml support_config
dataset_id: foo
preprocess_name: foo
tests:
  run_commands:
    - PYTHONPATH=src python run.py --config config/custom/foo_openai.yaml
```
"""
    cfg_block = {
        "preprocess_name": "foo",
        "tests": {"run_commands": ["PYTHONPATH=src python run.py --config config/custom/foo_openai.yaml"]},
    }
    normalized = _normalize_support_config_run_commands(md, cfg_block, "foo")
    assert "--max-samples 0" in normalized


def test_keep_existing_max_samples() -> None:
    md = """
```yaml support_config
dataset_id: foo
preprocess_name: foo
tests:
  run_commands:
    - PYTHONPATH=src python run.py --config config/custom/foo_vllm.yaml --max-samples 0
```
"""
    cfg_block = {
        "preprocess_name": "foo",
        "tests": {"run_commands": ["PYTHONPATH=src python run.py --config config/custom/foo_vllm.yaml --max-samples 0"]},
    }
    normalized = _normalize_support_config_run_commands(md, cfg_block, "foo")
    assert normalized.count("--max-samples 0") == 1
