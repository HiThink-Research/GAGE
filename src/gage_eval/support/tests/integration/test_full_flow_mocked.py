from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.support.agent_bridge import AgentResult
from gage_eval.support.config import PathConfig, SupportConfig
from gage_eval.support.inspector import inspect_dataset
from gage_eval.support.pipeline import run_design, run_implement
from gage_eval.support.utils import artifact_slug_from_dataset_id


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.io
def test_full_flow_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "foo.jsonl"
    _write_jsonl(data_path, ['{"id": "1", "question": "q", "answer": "a"}'])

    cfg = SupportConfig(paths=PathConfig(workspace_root=tmp_path / "dev_docs", local_datasets_root=tmp_path))
    cfg.execution.dry_run_default = False

    dataset_dir = inspect_dataset(
        dataset_name="foo",
        subset=None,
        split=None,
        max_samples=1,
        local_path=data_path,
        cfg=cfg,
    )
    slug = dataset_dir.name
    artifact = artifact_slug_from_dataset_id(slug)

    design_md = f"""# Design

```yaml support_config
dataset_id: {slug}
preprocess_name: {slug}_p
fields:
  question_field: question
  answers_field: answer
  content_field: messages
metrics:
  - metric_id: exact_match
    implementation: exact_match
tests:
  run_commands:
    - python -c "print('ok')"
```
"""

    monkeypatch.setattr(
        "gage_eval.support.pipeline.call_agent",
        lambda prompt, cfg: AgentResult(stdout=design_md, stderr="", returncode=0),
    )
    design_path = run_design(slug, cfg=cfg, force=True)
    rendered = design_path.read_text(encoding="utf-8")
    assert "<!-- support:auto_preview:start -->" in rendered
    assert f"config/custom/{artifact}_openai.yaml" in rendered
    assert f"src/gage_eval/assets/datasets/preprocessors/{artifact}_preprocessor.py" in rendered

    preproc_rel = Path("src/gage_eval/assets/datasets/preprocessors") / f"{artifact}_preprocessor.py"
    preproc_content = (
        "from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor\n"
        "class FooPreprocessor(BasePreprocessor):\n"
        "    def to_sample(self, record, **kwargs):\n"
        "        return record\n"
    )
    monkeypatch.setattr(
        "gage_eval.support.pipeline.render_files_from_agent",
        lambda prompt, cfg: {preproc_rel: preproc_content},
    )

    executed: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        executed.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("gage_eval.support.pipeline.subprocess.run", fake_run)

    run_implement(slug, cfg=cfg, dry_run=False, force=True, skip_tests=False)

    assert preproc_rel.exists()
    assert (Path("config/custom") / f"{artifact}_openai.yaml").exists()
    assert (Path("config/custom") / f"{artifact}_vllm.yaml").exists()
    assert (Path("src/gage_eval/assets/datasets/preprocessors/custom.py")).read_text(encoding="utf-8").find(
        f"\"{slug}_p\""
    ) != -1
    assert any("python -c" in c for c in executed)


@pytest.mark.io
def test_full_flow_multimodal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "mm.jsonl"
    _write_jsonl(
        data_path,
        [
            '{"id": "1", "messages": [{"role":"user","content":[{"type":"text","text":"q"}, {"type":"image_url","image_url":{"url":"x.png"}}]}], "answer": "a"}'
        ],
    )

    cfg = SupportConfig(paths=PathConfig(workspace_root=tmp_path / "dev_docs", local_datasets_root=tmp_path))
    cfg.execution.dry_run_default = False

    dataset_dir = inspect_dataset(
        dataset_name="mm",
        subset=None,
        split=None,
        max_samples=1,
        local_path=data_path,
        cfg=cfg,
    )
    slug = dataset_dir.name
    artifact = artifact_slug_from_dataset_id(slug)

    design_md = f"""# Design
```yaml support_config
dataset_id: {slug}
preprocess_name: {slug}_p
fields:
  question_field: messages.0.content.0.text
  answers_field: answer
  content_field: messages.0.content
modalities: [text, image]
doc_to_visual: gage_eval.assets.datasets.utils.multimodal:embed_local_message_images
metrics:
  - metric_id: exact_match
    implementation: exact_match
tests:
  run_commands: []
```
"""
    monkeypatch.setattr(
        "gage_eval.support.pipeline.call_agent",
        lambda prompt, cfg: AgentResult(stdout=design_md, stderr="", returncode=0),
    )
    run_design(slug, cfg=cfg, force=True)

    preproc_rel = Path("src/gage_eval/assets/datasets/preprocessors") / f"{artifact}_preprocessor.py"
    monkeypatch.setattr(
        "gage_eval.support.pipeline.render_files_from_agent",
        lambda prompt, cfg: {
            preproc_rel: (
                "from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor\n"
                "class FooPreprocessor(BasePreprocessor):\n"
                "    def to_sample(self, record, **kwargs):\n"
                "        return record\n"
            )
        },
    )
    run_implement(slug, cfg=cfg, dry_run=False, force=True, skip_tests=True)
    assert preproc_rel.exists()


@pytest.mark.io
def test_full_flow_with_custom_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "metric.jsonl"
    _write_jsonl(data_path, ['{"id": "1", "question": "q", "answer": "a"}'])

    cfg = SupportConfig(paths=PathConfig(workspace_root=tmp_path / "dev_docs", local_datasets_root=tmp_path))
    cfg.execution.dry_run_default = False

    dataset_dir = inspect_dataset(
        dataset_name="metric",
        subset=None,
        split=None,
        max_samples=1,
        local_path=data_path,
        cfg=cfg,
    )
    slug = dataset_dir.name
    artifact = artifact_slug_from_dataset_id(slug)

    design_md = f"""# Design
```yaml support_config
dataset_id: {slug}
preprocess_name: {slug}_p
fields:
  question_field: question
  answers_field: answer
  content_field: messages
metrics:
  - metric_id: custom_metric
    implementation: gage_eval.metrics.builtin.custom_metric:CustomMetric
tests:
  run_commands: []
```
"""
    monkeypatch.setattr(
        "gage_eval.support.pipeline.call_agent",
        lambda prompt, cfg: AgentResult(stdout=design_md, stderr="", returncode=0),
    )
    run_design(slug, cfg=cfg, force=True)

    preproc_rel = Path("src/gage_eval/assets/datasets/preprocessors") / f"{artifact}_preprocessor.py"
    metric_rel = Path("src/gage_eval/metrics/builtin/custom_metric.py")
    monkeypatch.setattr(
        "gage_eval.support.pipeline.render_files_from_agent",
        lambda prompt, cfg: {
            preproc_rel: (
                "from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor\n"
                "class FooPreprocessor(BasePreprocessor):\n"
                "    def to_sample(self, record, **kwargs):\n"
                "        return record\n"
            ),
            metric_rel: "from gage_eval.metrics.base import BaseMetric\nclass CustomMetric(BaseMetric):\n    def compute(self, context):\n        return {}\n",
        },
    )
    run_implement(slug, cfg=cfg, dry_run=False, force=True, skip_tests=True)
    assert preproc_rel.exists()
    assert metric_rel.exists()


@pytest.mark.io
def test_full_flow_with_custom_metric_registry_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "metric_registry.jsonl"
    _write_jsonl(data_path, ['{"id": "1", "question": "q", "answer": "a"}'])

    cfg = SupportConfig(paths=PathConfig(workspace_root=tmp_path / "dev_docs", local_datasets_root=tmp_path))
    cfg.execution.dry_run_default = False

    dataset_dir = inspect_dataset(
        dataset_name="metric_registry",
        subset=None,
        split=None,
        max_samples=1,
        local_path=data_path,
        cfg=cfg,
    )
    slug = dataset_dir.name
    artifact = artifact_slug_from_dataset_id(slug)

    design_md = f"""# Design
```yaml support_config
dataset_id: {slug}
preprocess_name: {slug}_p
fields:
  question_field: question
  answers_field: answer
  content_field: messages
metrics:
  - metric_id: custom_metric
    implementation: custom_impl
tests:
  run_commands: []
```
"""
    monkeypatch.setattr(
        "gage_eval.support.pipeline.call_agent",
        lambda prompt, cfg: AgentResult(stdout=design_md, stderr="", returncode=0),
    )
    design_path = run_design(slug, cfg=cfg, force=True)
    rendered = design_path.read_text(encoding="utf-8")
    assert "src/gage_eval/metrics/builtin/custom_impl.py" in rendered

    preproc_rel = Path("src/gage_eval/assets/datasets/preprocessors") / f"{artifact}_preprocessor.py"
    metric_rel = Path("src/gage_eval/metrics/builtin/custom_impl.py")
    monkeypatch.setattr(
        "gage_eval.support.pipeline.render_files_from_agent",
        lambda prompt, cfg: {
            preproc_rel: (
                "from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor\n"
                "class FooPreprocessor(BasePreprocessor):\n"
                "    def to_sample(self, record, **kwargs):\n"
                "        return record\n"
            ),
            metric_rel: "from gage_eval.metrics.base import BaseMetric\nclass CustomImpl(BaseMetric):\n    def compute(self, context):\n        return {}\n",
        },
    )
    run_implement(slug, cfg=cfg, dry_run=False, force=True, skip_tests=True)
    assert preproc_rel.exists()
    assert metric_rel.exists()
