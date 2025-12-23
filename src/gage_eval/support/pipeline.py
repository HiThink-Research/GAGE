from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from .agent_bridge import call_agent, render_files_from_agent
from .config import SupportConfig
from .utils import (
    artifact_slug_from_dataset_id,
    class_name_from_slug,
    ensure_git_clean,
    guard_commands,
    iter_test_commands,
    normalize_preprocess_name,
    parse_support_config,
    slugify_dataset_name,
)


TEMPLATE_DIR = Path(__file__).parent / "templates"

_SUPPORT_BLOCK_RE = re.compile(
    r"```yaml\s+support_config\s*\n.*?```",
    flags=re.DOTALL | re.IGNORECASE,
)
_SUPPORT_BLOCK_CONTENT_RE = re.compile(
    r"```yaml\s+support_config\s*\n(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)
_AUTO_PREVIEW_START = "<!-- support:auto_preview:start -->"
_AUTO_PREVIEW_END = "<!-- support:auto_preview:end -->"

def _load_reference_text(path: Path, *, max_chars: int = 6000) -> str:
    """Read reference file with truncation to keep prompt stable."""

    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...<truncated {len(text) - max_chars} chars>\n"
    return text


def _replace_support_config_scalar(md: str, *, key: str, value: str) -> str:
    match = _SUPPORT_BLOCK_CONTENT_RE.search(md)
    if not match:
        return md
    content = match.group(1)
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", flags=re.MULTILINE)
    if pattern.search(content):
        updated = pattern.sub(f"{key}: {value}", content, count=1)
    else:
        updated = content.rstrip() + f"\n{key}: {value}\n"
    if not updated.endswith("\n"):
        updated += "\n"
    return md[: match.start(1)] + updated + md[match.end(1) :]


def _render_template(name: str, **context: Any) -> str:
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Bench-Support requires jinja2 to render templates.") from exc
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template(name)
    return tmpl.render(**context)


def _resolve_dataset_dir(dataset: str, cfg: SupportConfig) -> Path:
    root = cfg.paths.workspace_root
    cand = root / dataset
    if cand.exists():
        return cand
    slug = slugify_dataset_name(dataset)
    cand = root / slug
    if cand.exists():
        return cand
    legacy_slug = dataset.replace("/", "_")
    legacy = root / legacy_slug
    if legacy.exists():
        return legacy
    # Backward compatibility: locate an existing dir whose slugify matches.
    for p in root.glob("*"):
        if p.is_dir() and slugify_dataset_name(p.name) == slug:
            return p
    return cand


def _strip_auto_preview(md: str) -> str:
    if _AUTO_PREVIEW_START not in md or _AUTO_PREVIEW_END not in md:
        return md
    pattern = re.compile(
        re.escape(_AUTO_PREVIEW_START) + r".*?" + re.escape(_AUTO_PREVIEW_END),
        flags=re.DOTALL,
    )
    return pattern.sub("", md).strip() + "\n"


def _normalize_flowchart_nodes(mermaid: str) -> str:
    """Best-effort rewrite for invalid `A Label --> B Label` flowchart lines.

    Mermaid 节点 id 不能包含空格；历史模板曾用 `A Inspect --> B SampleJson` 形式，部分渲染器会报错。
    这里将其转换为 `Inspect --> SampleJson`，只对简单直连边做重写。
    """

    rewritten: list[str] = []
    id_to_label: dict[str, str] = {}
    last_rhs_label: Optional[str] = None

    both_re = re.compile(r"^(\s*)(\w+)\s+(\w+)\s*(-->|---)\s*(\w+)\s+(\w+)\s*$")
    right_labeled_re = re.compile(r"^(\s*)(\w+)\s*(-->|---)\s*(\w+)\s+(\w+)\s*$")
    left_labeled_re = re.compile(r"^(\s*)(\w+)\s+(\w+)\s*(-->|---)\s*(\w+)\s*$")
    plain_re = re.compile(r"^(\s*)(\w+)\s*(-->|---)\s*(\w+)\s*$")

    for line in mermaid.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("flowchart"):
            rewritten.append(line)
            continue

        m = both_re.match(line)
        if m:
            indent, lhs_id, lhs_label, arrow, rhs_id, rhs_label = m.groups()
            id_to_label.setdefault(lhs_id, lhs_label)
            id_to_label.setdefault(rhs_id, rhs_label)
            rewritten.append(f"{indent}{lhs_label} {arrow} {rhs_label}")
            last_rhs_label = rhs_label
            continue

        m = right_labeled_re.match(line)
        if m:
            indent, lhs_id, arrow, rhs_id, rhs_label = m.groups()
            id_to_label.setdefault(rhs_id, rhs_label)
            if re.fullmatch(r"[A-Z]", lhs_id) and lhs_id not in id_to_label:
                if last_rhs_label and not re.fullmatch(r"[A-Z]", last_rhs_label):
                    id_to_label[lhs_id] = last_rhs_label
            lhs_label = id_to_label.get(lhs_id, lhs_id)
            rewritten.append(f"{indent}{lhs_label} {arrow} {rhs_label}")
            last_rhs_label = rhs_label
            continue

        m = left_labeled_re.match(line)
        if m:
            indent, lhs_id, lhs_label, arrow, rhs_id = m.groups()
            id_to_label.setdefault(lhs_id, lhs_label)
            rhs_label = id_to_label.get(rhs_id, rhs_id)
            rewritten.append(f"{indent}{lhs_label} {arrow} {rhs_label}")
            last_rhs_label = rhs_label
            continue

        m = plain_re.match(line)
        if m:
            indent, lhs_id, arrow, rhs_id = m.groups()
            if re.fullmatch(r"[A-Z]", lhs_id) and lhs_id not in id_to_label:
                if last_rhs_label and not re.fullmatch(r"[A-Z]", last_rhs_label):
                    id_to_label[lhs_id] = last_rhs_label
            lhs_label = id_to_label.get(lhs_id, lhs_id)
            rhs_label = id_to_label.get(rhs_id, rhs_id)
            rewritten.append(f"{indent}{lhs_label} {arrow} {rhs_label}")
            last_rhs_label = rhs_label
            continue

        rewritten.append(line)
    return "\n".join(rewritten)


def _normalize_mermaid_blocks(md: str) -> str:
    """Normalize Mermaid code blocks for better renderer compatibility."""

    def _replace(match: re.Match[str]) -> str:
        block = match.group(0)
        header, body, footer = match.group(1), match.group(2), match.group(3)
        body = _normalize_flowchart_nodes(body)
        if body and not body.endswith("\n"):
            body += "\n"
        return f"{header}{body}{footer}"

    pattern = re.compile(r"(```mermaid\s*\n)(.*?)(```)", flags=re.DOTALL | re.IGNORECASE)
    return pattern.sub(_replace, md)


def _render_preprocessor_preview(cfg_block: Dict[str, Any], slug: str) -> str:
    fields = cfg_block.get("fields") or {}
    question_field = str(fields.get("question_field") or "")
    answers_field = str(fields.get("answers_field") or "")
    dataset_id = str(cfg_block.get("dataset_id") or slug)
    preprocess_name = str(cfg_block.get("preprocess_name") or "")
    preprocessor_class_name = class_name_from_slug(slug) + "Preprocessor"
    modalities = cfg_block.get("modalities") or ["text"]

    to_sample_body = f"""
question_field = str(merged.get("question_field") or "{question_field}")
answers_field = str(merged.get("answers_field") or "{answers_field}")

question = extract_field(record, question_field)
answer = extract_field(record, answers_field)
if question is None:
    raise ValueError(f"Missing question_field: {{question_field}}")

# Prefer stable ids when available (normalize_sample 会兜底生成哈希 id)。
stable_id = record.get("unique_id") or record.get("id") or record.get("_id")
if stable_id:
    sample["id"] = str(stable_id)

sample["messages"] = [
    {{
        "role": "system",
        "content": [{{"type": "text", "text": "请只输出最终答案，不要输出推导过程。"}}],
    }},
    {{
        "role": "user",
        "content": [{{"type": "text", "text": str(question)}}],
    }},
]

meta = dict(sample.get("metadata") or {{}})
for key in ("subject", "level", "unique_id"):
    if record.get(key) is not None:
        meta[key] = record.get(key)
if record.get("solution") is not None:
    meta["reference_solution"] = record.get("solution")
sample["metadata"] = meta

answers: list[str] = []
if answer is not None:
    answers.append(str(answer))
if answers:
    sample["answers"] = answers
    sample["label"] = answers[0]
    meta = dict(sample.get("metadata") or {{}})
    meta["answers"] = answers
    sample["metadata"] = meta
""".strip()

    return _render_template(
        "preprocessor.py.j2",
        dataset_slug=slug,
        dataset_name=dataset_id,
        dataset_id=dataset_id,
        preprocess_name=preprocess_name,
        preprocessor_class_name=preprocessor_class_name,
        to_sample_body=to_sample_body,
        modalities=modalities,
    )


def _render_preprocessor_registry_wrapper_preview(cfg_block: Dict[str, Any], slug: str) -> str:
    preprocess_name = str(cfg_block.get("preprocess_name") or "")
    dataset_id = str(cfg_block.get("dataset_id") or slug)
    class_name = class_name_from_slug(slug) + "Preprocessor"
    import_path = f"gage_eval.assets.datasets.preprocessors.{slug}_preprocessor"
    return (
        '"""Auto-generated preprocessors (Bench-Support)."""\n\n'
        "from __future__ import annotations\n\n"
        "from gage_eval.registry import registry\n"
        f"from {import_path} import {class_name} as _Impl\n\n"
        "@registry.asset(\n"
        '    "dataset_preprocessors",\n'
        f'    "{preprocess_name}",\n'
        f'    desc="{dataset_id} Preprocessor",\n'
        ")\n"
        f"class {class_name}(_Impl):\n"
        "    pass\n"
    )


_BUILTIN_METRIC_IDS = {
    "exact_match",
    "contains",
    "numeric_match",
    "regex_match",
    "judge_threshold",
    "text_length",
    "latency",
    "multi_choice_accuracy",
    "docvqa_anls",
    "mmmu_accuracy",
    "mathvista_accuracy",
    "likelihood",
    "ranking",
}


def _render_metric_stub_preview(*, metric_registry_id: str, metric_id: str) -> Tuple[Path, str]:
    """Render a minimal metric stub preview.

    - metric_registry_id: registry 中用于加载的实现名（support_config.metrics[*].implementation）
    - metric_id: 报告侧展示/聚合使用的指标标识（support_config.metrics[*].metric_id）
    """

    class_name = class_name_from_slug(metric_registry_id) + "Metric"
    path = Path("src/gage_eval/metrics/builtin") / f"{metric_registry_id}.py"
    code = f"""from __future__ import annotations

import re
from typing import Any

from gage_eval.metrics.base import ComparisonMetric
from gage_eval.metrics.utils import normalize_text_advanced
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "{metric_registry_id}",
    desc="Auto-generated metric impl: {metric_registry_id} (metric_id={metric_id})",
    tags=("auto",),
    default_aggregation="mean",
)
class {class_name}(ComparisonMetric):
    \"\"\"简单等价匹配（自动生成骨架，可按需增强）。\"\"\"

    default_reference_field = "sample.label"
    default_prediction_field = "model_output.answer"

    def compare(self, prediction: Any, reference: Any) -> tuple[float, dict]:
        pred = _normalize(prediction)
        ref = _normalize(reference)
        if not pred or not ref:
            return 0.0, {{"pred_norm": pred, "ref_norm": ref, "warning": "empty_input"}}
        return (1.0 if pred == ref else 0.0), {{"pred_norm": pred, "ref_norm": ref}}


_BOXED_RE = re.compile(r"\\\\boxed\\{{(.*?)\\}}")


def _normalize(value: Any) -> str:
    text = normalize_text_advanced(value, strip=True, collapse_whitespace=True) or ""
    if not text:
        return ""
    # Prefer boxed answer when present.
    m = _BOXED_RE.search(text)
    if m:
        text = m.group(1)
    # Common LaTeX cleanup.
    text = text.replace("\\\\left", "").replace("\\\\right", "")
    text = text.replace("$", "")
    text = re.sub(r"\\s+", "", text)
    return text
"""
    return path, code.rstrip() + "\n"


def _build_auto_preview_section(
    cfg_block: Dict[str, Any],
    slug: str,
    *,
    cfg: Optional[SupportConfig] = None,
    project_root: Optional[Path] = None,
    dataset_meta: Optional[Dict[str, Any]] = None,
) -> str:
    resolved_project_root: Optional[Path] = None
    if project_root is not None:
        resolved_project_root = project_root.expanduser()
        if not resolved_project_root.is_absolute():
            resolved_project_root = (Path.cwd() / resolved_project_root).resolve()
        else:
            resolved_project_root = resolved_project_root.resolve()

    configs = _render_pipeline_configs(cfg_block, slug, cfg=cfg, project_root=resolved_project_root, dataset_meta=dataset_meta)
    preproc_rel = Path("src/gage_eval/assets/datasets/preprocessors") / f"{slug}_preprocessor.py"
    if resolved_project_root and (resolved_project_root / preproc_rel).exists():
        preproc = (resolved_project_root / preproc_rel).read_text(encoding="utf-8").rstrip()
    else:
        preproc = _render_preprocessor_preview(cfg_block, slug).rstrip()

    wrapper = _render_preprocessor_registry_wrapper_preview(cfg_block, slug).rstrip()

    openai_rel = Path("config/custom") / f"{slug}_openai.yaml"
    vllm_rel = Path("config/custom") / f"{slug}_vllm.yaml"
    if resolved_project_root and (resolved_project_root / openai_rel).exists():
        openai_yaml = (resolved_project_root / openai_rel).read_text(encoding="utf-8").rstrip()
    else:
        openai_yaml = configs["openai"].rstrip()
    if resolved_project_root and (resolved_project_root / vllm_rel).exists():
        vllm_yaml = (resolved_project_root / vllm_rel).read_text(encoding="utf-8").rstrip()
    else:
        vllm_yaml = configs["vllm"].rstrip()

    metrics = cfg_block.get("metrics") or []
    stubs: list[Tuple[Path, str]] = []
    if isinstance(metrics, list):
        for m in metrics:
            if not isinstance(m, dict):
                continue
            metric_id = str(m.get("metric_id") or "").strip()
            impl = str(m.get("implementation") or metric_id).strip()
            if not metric_id:
                continue
            if ":" in impl or "." in impl:
                continue
            if impl in _BUILTIN_METRIC_IDS:
                continue
            metric_rel = Path("src/gage_eval/metrics/builtin") / f"{impl}.py"
            if resolved_project_root and (resolved_project_root / metric_rel).exists():
                stubs.append((metric_rel, (resolved_project_root / metric_rel).read_text(encoding="utf-8").rstrip() + "\n"))
            else:
                stubs.append(_render_metric_stub_preview(metric_registry_id=impl, metric_id=metric_id))

    parts: list[str] = [
        _AUTO_PREVIEW_START,
        "### 核心产物预览（自动生成）",
        "",
        "> 本节由 Support 根据文末 `support_config` 自动生成，用于评审；实际落地以 `implement` 产物为准。",
        "> 预览代码已按 sample.py 结构补齐 answers/label/metadata/modalities，避免 TODO 占位。",
        "",
        "#### 预处理器",
        f"目标文件：`src/gage_eval/assets/datasets/preprocessors/{slug}_preprocessor.py`",
        "",
        "```python",
        preproc,
        "```",
        "",
        "目标文件：`src/gage_eval/assets/datasets/preprocessors/custom.py`（registry wrapper）",
        "",
        "```python",
        wrapper,
        "```",
        "",
        "#### 配置文件（PipelineConfig）",
        f"目标文件：`config/custom/{slug}_openai.yaml`",
        "",
        "```yaml",
        openai_yaml,
        "```",
        "",
        f"目标文件：`config/custom/{slug}_vllm.yaml`",
        "",
        "```yaml",
        vllm_yaml,
        "```",
    ]
    if stubs:
        parts += [
            "",
            "#### 指标代码（如需新增）",
            "",
            "以下为根据 `support_config.metrics` 推断的新增指标骨架（可按需增强）：",
        ]
        for path, code in stubs:
            parts += [
                "",
                f"目标文件：`{path}`",
                "",
                "```python",
                code.rstrip(),
                "```",
            ]
    parts.append(_AUTO_PREVIEW_END)
    return "\n".join(parts).strip()

def _iter_custom_metric_impls(cfg_block: Dict[str, Any]) -> list[str]:
    metrics = cfg_block.get("metrics") or []
    impls: list[str] = []
    if not isinstance(metrics, list):
        return impls
    for m in metrics:
        if not isinstance(m, dict):
            continue
        metric_id = str(m.get("metric_id") or "").strip()
        impl = str(m.get("implementation") or metric_id).strip()
        if not impl:
            continue
        # class_path 或 module path 不需要生成文件
        if ":" in impl or "." in impl:
            continue
        if impl in _BUILTIN_METRIC_IDS:
            continue
        impls.append(impl)
    # stable unique order
    seen: set[str] = set()
    out: list[str] = []
    for impl in impls:
        if impl in seen:
            continue
        seen.add(impl)
        out.append(impl)
    return out


def _render_landing_plan_table(cfg_block: Dict[str, Any], slug: str) -> str:
    preprocess_name = str(cfg_block.get("preprocess_name") or "<preprocess_name>")
    metric_impls = _iter_custom_metric_impls(cfg_block)
    metric_deliverables = (
        "；".join(f"`src/gage_eval/metrics/builtin/{impl}.py`" for impl in metric_impls)
        if metric_impls
        else "复用内置指标（无需新增代码）"
    )

    rows = [
        (
            "1",
            "确认并固化 support_config 真相源",
            "`dev_docs/<dataset_dir>/design.md`（文末唯一 `yaml support_config`）",
            "`python -m gage_eval.support design <dataset> --force`",
            "未开始",
        ),
        (
            "2",
            "生成预处理器逻辑（BasePreprocessor）",
            f"`src/gage_eval/assets/datasets/preprocessors/{slug}_preprocessor.py`",
            "`python -m gage_eval.support implement <dataset> --force`",
            "未开始",
        ),
        (
            "3",
            "注册预处理器到 Registry",
            f"`src/gage_eval/assets/datasets/preprocessors/custom.py`（注册名：`{preprocess_name}`）",
            "`python -c \"import gage_eval; from gage_eval.registry import registry; registry.get('dataset_preprocessors','"
            + preprocess_name
            + "')\"`",
            "未开始",
        ),
        (
            "4",
            "生成两份 PipelineConfig（OpenAI/vLLM）",
            f"`config/custom/{slug}_openai.yaml`；`config/custom/{slug}_vllm.yaml`",
            "执行 `tests.run_commands`",
            "未开始",
        ),
        (
            "5",
            "接入/实现指标（metrics）",
            metric_deliverables,
            "`pytest src/gage_eval/support/tests -q`（或补齐对应 unit test）",
            "未开始",
        ),
        (
            "6",
            "多组集成测试覆盖全部变更",
            "`src/gage_eval/support/tests/integration/*`",
            "`pytest -m \"not gpu and not network\" src/gage_eval/support/tests -q`",
            "未开始",
        ),
    ]

    lines = [
        "| 序号 | 任务 | 交付物 | 验证/测试 | 完成状态 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for idx, task, deliverable, verify, status in rows:
        lines.append(f"| {idx} | {task} | {deliverable} | {verify} | {status} |")
    return "\n".join(lines)


def _rewrite_landing_plan_table(md: str, cfg_block: Dict[str, Any], slug: str) -> str:
    """Rewrite the landing plan table to match actual implement outputs.

    设计文档的落地计划属于“人读”内容，但路径必须与 Support 实际写盘一致，避免误导实现。
    """

    lines = md.splitlines()
    section_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip() == "## 5. 落地计划":
            section_idx = i
            break
    if section_idx is None:
        return md

    table_start: Optional[int] = None
    for i in range(section_idx, len(lines)):
        if lines[i].lstrip().startswith("| 序号 |") and "交付物" in lines[i]:
            table_start = i
            break
    if table_start is None:
        return md

    table_end = len(lines)
    for i in range(table_start + 1, len(lines)):
        if lines[i].lstrip().startswith("|"):
            continue
        table_end = i
        break

    new_table = _render_landing_plan_table(cfg_block, slug).splitlines()
    out = lines[:table_start] + new_table + lines[table_end:]
    return "\n".join(out).rstrip() + "\n"


def _normalize_support_config_run_commands(md: str, cfg_block: Dict[str, Any], slug: str) -> str:
    preprocess_name = str(cfg_block.get("preprocess_name") or "").strip()
    if not preprocess_name:
        return md

    # Support 产物文件名统一由 slug 控制（而非 preprocess_name）。
    md = md.replace(
        f"config/custom/{preprocess_name}_openai.yaml",
        f"config/custom/{slug}_openai.yaml",
    ).replace(
        f"config/custom/{preprocess_name}_vllm.yaml",
        f"config/custom/{slug}_vllm.yaml",
    )

    # Self-test commands should validate the pipeline wiring without running inference by default.
    # Users can run `run.py` without `--max-samples 0` for real evaluations.
    openai_cmd = f"PYTHONPATH=src python run.py --config config/custom/{slug}_openai.yaml"
    vllm_cmd = f"PYTHONPATH=src python run.py --config config/custom/{slug}_vllm.yaml"

    def _ensure_max_samples(text: str, base_cmd: str) -> str:
        marker = f"{base_cmd} --max-samples 0"
        if marker in text:
            return text
        if base_cmd in text:
            return text.replace(base_cmd, marker)
        # Fallback: line-wise prefix match，补上 --max-samples 0。
        lines = text.splitlines()
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(base_cmd) and "--max-samples" not in stripped and "--max_samples" not in stripped:
                out.append(line.replace(base_cmd, marker, 1))
            else:
                out.append(line)
        return "\n".join(out)

    md = _ensure_max_samples(md, openai_cmd)
    md = _ensure_max_samples(md, vllm_cmd)
    return md


def _normalize_design_doc_artifact_refs(md: str, *, slug: str, legacy_slug: str) -> str:
    """Best-effort rewrite for legacy artifact names inside design.md.

    目标：design.md 中的人读内容/示例代码/路径描述要与 Support 实际写盘一致。
    """

    legacy = str(legacy_slug or "").strip()
    if not legacy or legacy == slug:
        return md
    updated = md.replace(f"`{legacy}`", f"`{slug}`")
    updated = updated.replace(
        f"src/gage_eval/assets/datasets/preprocessors/{legacy}_preprocessor.py",
        f"src/gage_eval/assets/datasets/preprocessors/{slug}_preprocessor.py",
    )
    updated = updated.replace(
        f"gage_eval.assets.datasets.preprocessors.{legacy}_preprocessor",
        f"gage_eval.assets.datasets.preprocessors.{slug}_preprocessor",
    )
    updated = updated.replace(
        f"config/custom/{legacy}_openai.yaml",
        f"config/custom/{slug}_openai.yaml",
    )
    updated = updated.replace(
        f"config/custom/{legacy}_vllm.yaml",
        f"config/custom/{slug}_vllm.yaml",
    )
    return updated


def _inject_auto_preview(
    md: str,
    cfg_block: Dict[str, Any],
    slug: str,
    *,
    cfg: Optional[SupportConfig] = None,
    project_root: Optional[Path] = None,
    dataset_meta: Optional[Dict[str, Any]] = None,
) -> str:
    md = _strip_auto_preview(md)
    match = _SUPPORT_BLOCK_RE.search(md)
    if not match:
        return md
    prefix = md[: match.start()].rstrip()
    support_block = md[match.start() : match.end()].strip()
    suffix = md[match.end() :].strip()

    preview = _build_auto_preview_section(cfg_block, slug, cfg=cfg, project_root=project_root, dataset_meta=dataset_meta)
    merged = prefix + "\n\n" + preview + "\n\n" + support_block
    if suffix:
        logger.warning("design.md 的 support_config 后存在额外内容；auto preview 将插入到其之前。")
        merged += "\n\n" + suffix
    return merged.rstrip() + "\n"


def _extract_support_config_yaml(md: str) -> str:
    matches = _SUPPORT_BLOCK_CONTENT_RE.findall(md)
    if len(matches) != 1:
        raise ValueError(f"design.md must contain exactly one yaml support_config block (found {len(matches)})")
    return str(matches[0]).strip() + "\n"


def run_design(dataset: str, *, cfg: SupportConfig, force: bool = False) -> Path:
    dataset_dir = _resolve_dataset_dir(dataset, cfg)
    dataset_meta = _read_inspect_meta(dataset_dir)
    dataset_slug_hint = artifact_slug_from_dataset_id(str(dataset_meta.get("hub_id") or dataset))
    legacy_slug = slugify_dataset_name(dataset_dir.name)
    design_path = dataset_dir / "design.md"
    if design_path.exists() and not force:
        # Refresh existing design.md deterministically (no agent call).
        design_md = design_path.read_text(encoding="utf-8")
        design_md = _normalize_mermaid_blocks(design_md)
        try:
            cfg_block = parse_support_config(design_md)
            dataset_slug = artifact_slug_from_dataset_id(str(cfg_block.get("dataset_id") or dataset_slug_hint))
            design_md = _normalize_support_config_run_commands(design_md, cfg_block, dataset_slug)
            cfg_block = parse_support_config(design_md)
            normalized_preprocess_name = normalize_preprocess_name(
                str(cfg_block.get("preprocess_name") or ""),
                artifact_slug=dataset_slug,
            )
            if normalized_preprocess_name != str(cfg_block.get("preprocess_name") or ""):
                design_md = _replace_support_config_scalar(design_md, key="preprocess_name", value=normalized_preprocess_name)
                cfg_block = parse_support_config(design_md)
            design_md = _normalize_design_doc_artifact_refs(design_md, slug=dataset_slug, legacy_slug=legacy_slug)
            design_md = _rewrite_landing_plan_table(design_md, cfg_block, dataset_slug)
            design_md = _inject_auto_preview(
                design_md,
                cfg_block,
                dataset_slug,
                cfg=cfg,
                project_root=cfg.paths.project_root,
                dataset_meta=dataset_meta,
            )
        except Exception as exc:
            logger.warning(f"design.md refresh skipped auto preview due to invalid support_config: {exc}")
        design_path.write_text(design_md, encoding="utf-8")
        logger.info(f"Refreshed design.md at {design_path} (no agent call). Use --force to regenerate.")
        return design_path

    sample_path = dataset_dir / "sample.json"
    schema_path = dataset_dir / "schema.json"
    meta_path = dataset_dir / "meta.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"sample.json not found in {dataset_dir}, run inspect first.")

    def _truncate_for_prompt(value: Any, *, max_depth: int = 6) -> Any:
        if max_depth <= 0:
            return "<truncated>"
        if isinstance(value, str):
            max_len = 800
            if len(value) <= max_len:
                return value
            return value[:max_len] + f"...<truncated {len(value) - max_len} chars>"
        if isinstance(value, list):
            max_items = 20
            out = [_truncate_for_prompt(v, max_depth=max_depth - 1) for v in value[:max_items]]
            if len(value) > max_items:
                out.append(f"<truncated {len(value) - max_items} items>")
            return out
        if isinstance(value, dict):
            return {k: _truncate_for_prompt(v, max_depth=max_depth - 1) for k, v in value.items()}
        return value

    sample_json_raw = sample_path.read_text(encoding="utf-8")
    try:
        sample_obj = json.loads(sample_json_raw)
        if isinstance(sample_obj, list):
            sample_obj = sample_obj[:2]
        sample_json = json.dumps(_truncate_for_prompt(sample_obj), ensure_ascii=False, indent=2)
    except Exception:
        sample_json = sample_json_raw[:20_000]

    schema_json_raw = schema_path.read_text(encoding="utf-8") if schema_path.exists() else "{}"
    try:
        schema_obj = json.loads(schema_json_raw)
        schema_json = json.dumps(_truncate_for_prompt(schema_obj), ensure_ascii=False, indent=2)
    except Exception:
        schema_json = schema_json_raw[:20_000]

    meta_json = meta_path.read_text(encoding="utf-8") if meta_path.exists() else "{}"

    # Optional few-shot references.
    ref_preproc = _load_reference_text(TEMPLATE_DIR / "reference_preprocessor.py.txt")
    ref_cfg = _load_reference_text(TEMPLATE_DIR / "reference_config.yaml.txt")
    sample_format_doc = _load_reference_text(cfg.paths.project_root / "src/gage_eval/assets/datasets/sample.py")
    metrics_base_doc = _load_reference_text(cfg.paths.project_root / "src/gage_eval/metrics/base.py")

    prompt = _render_template(
        "prompt_design.j2",
        dataset_name=dataset,
        meta_json=meta_json,
        sample_json=sample_json,
        schema_json=schema_json,
        reference_preprocessor_code=ref_preproc,
        reference_config_yaml=ref_cfg,
        sample_format_doc=sample_format_doc,
        metrics_base_doc=metrics_base_doc,
        language=cfg.language,
        design_doc_template=_render_template(
            "design_doc.md.j2",
            dataset_name=dataset,
            dataset_slug=dataset_slug_hint,
            language=cfg.language,
        ),
    )

    if design_path.exists() and not force:
        raise FileExistsError(f"{design_path} exists, use --force to overwrite.")

    def _generate_design_md(*, use_stdin: bool) -> str:
        result = call_agent(prompt, cfg, prefer_stdin=use_stdin)
        if result.returncode != 0 and not result.stdout.strip():
            raise RuntimeError(f"Agent failed: {result.stderr.strip()}")
        design_md_local = _normalize_mermaid_blocks(result.stdout)
        try:
            cfg_block = parse_support_config(design_md_local)
            dataset_slug = artifact_slug_from_dataset_id(str(cfg_block.get("dataset_id") or dataset_slug_hint))
            design_md_local = _normalize_support_config_run_commands(design_md_local, cfg_block, dataset_slug)
            cfg_block = parse_support_config(design_md_local)
            normalized_preprocess_name = normalize_preprocess_name(
                str(cfg_block.get("preprocess_name") or ""),
                artifact_slug=dataset_slug,
            )
            if normalized_preprocess_name != str(cfg_block.get("preprocess_name") or ""):
                design_md_local = _replace_support_config_scalar(
                    design_md_local, key="preprocess_name", value=normalized_preprocess_name
                )
                cfg_block = parse_support_config(design_md_local)
            design_md_local = _normalize_design_doc_artifact_refs(design_md_local, slug=dataset_slug, legacy_slug=legacy_slug)
            design_md_local = _rewrite_landing_plan_table(design_md_local, cfg_block, dataset_slug)
            design_md_local = _inject_auto_preview(
                design_md_local,
                cfg_block,
                dataset_slug,
                cfg=cfg,
                project_root=cfg.paths.project_root,
                dataset_meta=dataset_meta,
            )
        except Exception as exc:
            raise RuntimeError(f"design.md 缺少或解析 support_config 失败：{exc}") from exc
        return design_md_local

    try:
        design_md = _generate_design_md(use_stdin=False)
    except Exception as exc:
        if cfg.agent.type == "gemini":
            logger.warning(f"Gemini 生成 design.md 首次校验失败，改用 stdin 传递 Prompt 重试：{exc}")
            try:
                design_md = _generate_design_md(use_stdin=True)
            except Exception:
                if design_path.exists() and force:
                    design_path.unlink(missing_ok=True)
                raise
        else:
            if design_path.exists() and force:
                design_path.unlink(missing_ok=True)
            raise

    design_path.write_text(design_md, encoding="utf-8")

    logger.info(f"Generated design.md at {design_path}")
    return design_path


def _append_custom_wrapper(
    *,
    slug: str,
    preprocess_name: str,
    dataset_name: str,
    custom_path: Path,
) -> None:
    class_name = class_name_from_slug(slug) + "Preprocessor"
    import_path = f"gage_eval.assets.datasets.preprocessors.{slug}_preprocessor"
    header = '"""Auto-generated preprocessors (Bench-Support)."""\n\nfrom __future__ import annotations\n\n'
    if not custom_path.exists():
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom_path.write_text(header, encoding="utf-8")
    text = custom_path.read_text(encoding="utf-8")
    if preprocess_name in text:
        return
    block = (
        f"\nfrom gage_eval.registry import registry\n"
        f"from {import_path} import {class_name} as _Impl\n\n"
        f"@registry.asset(\n"
        f"    \"dataset_preprocessors\",\n"
        f"    \"{preprocess_name}\",\n"
        f"    desc=\"{dataset_name} Preprocessor\",\n"
        f")\n"
        f"class {class_name}(_Impl):\n"
        f"    pass\n"
    )
    custom_path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


def _ensure_preprocessors_custom_import(init_path: Path) -> None:
    """Ensure preprocessors/__init__.py imports custom wrappers when present.

    设计约定：Support 生成的注册包装写在 `preprocessors/custom.py`，需要在包导入时触发注册。
    `gage_eval` 的 auto_discover 会 import `preprocessors/builtin.py`，因此在 package __init__ 中 try-import custom 即可生效。
    """

    init_path.parent.mkdir(parents=True, exist_ok=True)
    snippet = (
        "try:\n"
        "    import importlib\n"
        "    importlib.import_module(\"gage_eval.assets.datasets.preprocessors.custom\")\n"
        "except ModuleNotFoundError as exc:\n"
        "    if exc.name != \"gage_eval.assets.datasets.preprocessors.custom\":\n"
        "        raise\n"
        "\n"
    )
    if init_path.exists():
        text = init_path.read_text(encoding="utf-8")
    else:
        text = '"""Preprocessor utilities."""\n\n__all__ = []\n'

    if "gage_eval.assets.datasets.preprocessors.custom" in text:
        return

    lines = text.splitlines()
    out: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and line.strip().startswith("__all__"):
            out.extend(snippet.rstrip("\n").splitlines())
            inserted = True
        out.append(line)
    if not inserted:
        if out and out[-1].strip():
            out.append("")
        out.extend(snippet.rstrip("\n").splitlines())
    init_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def _append_custom_metric_imports(*, metric_impls: list[str], custom_path: Path) -> None:
    """Ensure metrics/builtin/custom.py imports generated metric modules.

    Auto-discovery imports `gage_eval.metrics.builtin` which try-imports this module;
    we import modules for side effects so `@registry.asset("metrics", ...)` is registered.
    """

    if not metric_impls:
        return
    header = '"""Auto-generated metrics (Bench-Support)."""\n\nfrom __future__ import annotations\n\n'
    custom_path.parent.mkdir(parents=True, exist_ok=True)
    if not custom_path.exists():
        custom_path.write_text(header + "import importlib\n", encoding="utf-8")
    text = custom_path.read_text(encoding="utf-8")
    if "import importlib" not in text:
        text = text.rstrip() + "\n\nimport importlib\n"
    updated = False
    for impl in metric_impls:
        module = f"gage_eval.metrics.builtin.{impl}"
        if module in text:
            continue
        text = text.rstrip() + f"\nimportlib.import_module(\"{module}\")\n"
        updated = True
    if updated:
        custom_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _ensure_metrics_custom_import(init_path: Path) -> None:
    """Ensure metrics/builtin/__init__.py imports custom wrappers when present."""

    init_path.parent.mkdir(parents=True, exist_ok=True)
    snippet = (
        "try:\n"
        "    import importlib\n"
        "    importlib.import_module(\"gage_eval.metrics.builtin.custom\")\n"
        "except ModuleNotFoundError as exc:\n"
        "    if exc.name != \"gage_eval.metrics.builtin.custom\":\n"
        "        raise\n"
        "\n"
    )
    text = init_path.read_text(encoding="utf-8") if init_path.exists() else ""
    if "gage_eval.metrics.builtin.custom" in text:
        return
    init_path.write_text(text.rstrip() + "\n\n" + snippet, encoding="utf-8")


def _render_pipeline_configs(
    cfg_block: Dict[str, Any],
    slug: str,
    *,
    cfg: Optional[SupportConfig] = None,
    project_root: Optional[Path] = None,
    dataset_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    return _render_pipeline_configs_with_context(cfg_block, slug, cfg=cfg, project_root=project_root, dataset_meta=dataset_meta)


_LOCAL_DATA_EXTS = (".jsonl", ".json", ".csv", ".parquet")


def _read_inspect_meta(dataset_dir: Path) -> Dict[str, Any]:
    """Read inspector meta.json (best-effort)."""

    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _looks_like_hf_hub_id(value: str) -> bool:
    """Heuristic: distinguish HF hub id from local filesystem path."""

    v = str(value or "").strip()
    if not v or "/" not in v:
        return False
    if v.startswith(("/", "./", "../", "~")):
        return False
    if "\\" in v:
        return False
    lowered = v.lower()
    if any(lowered.endswith(ext) for ext in _LOCAL_DATA_EXTS):
        return False
    return True


def _infer_dataset_context(
    cfg_block: Dict[str, Any],
    *,
    slug: str,
    cfg: Optional[SupportConfig],
    project_root: Optional[Path],
    dataset_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Derive dataset loader/hub/path context for config templates."""

    dataset_id = str(cfg_block.get("dataset_id") or slug)
    explicit_loader = str(cfg_block.get("loader") or "").strip()
    explicit_hub = str(cfg_block.get("hub") or "").strip()
    explicit_data_path = cfg_block.get("data_path") or cfg_block.get("path")
    data_path = str(explicit_data_path).strip() if isinstance(explicit_data_path, str) else ""

    meta_hub_id = str((dataset_meta or {}).get("hub_id") or "").strip()
    meta_subset = (dataset_meta or {}).get("subset")
    meta_split = (dataset_meta or {}).get("split")

    hub_id = str(cfg_block.get("hub_id") or meta_hub_id or dataset_id).strip()
    subset = cfg_block.get("subset", meta_subset)
    split = cfg_block.get("split", meta_split) or "train"

    trust_remote_code = bool(cfg_block.get("trust_remote_code", False))

    ctx: Dict[str, Any] = {}

    # If user explicitly declares loader, respect it and only fill missing fields.
    if explicit_loader:
        ctx["loader"] = explicit_loader
        if explicit_loader in {"hf_hub", "modelscope"}:
            ctx["hub"] = explicit_hub or "huggingface"
            ctx["hub_id"] = hub_id
            ctx["subset"] = subset
            ctx["split"] = split
            ctx["trust_remote_code"] = trust_remote_code
        else:
            if data_path:
                ctx["data_path"] = data_path
        return ctx

    # If data_path explicitly provided, default to jsonl.
    if data_path:
        ctx["loader"] = "jsonl"
        ctx["data_path"] = data_path
        return ctx

    # Try resolve as local path from meta.hub_id (when inspect used --local-path).
    if meta_hub_id:
        candidate = Path(meta_hub_id).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists():
            ctx["loader"] = "jsonl"
            if project_root:
                ctx["data_path"] = os.path.relpath(candidate, project_root)
            else:
                ctx["data_path"] = str(candidate)
            return ctx

    # Try local-datasets/<slug>.jsonl when configured.
    if cfg and project_root:
        local_root = cfg.paths.local_datasets_root
        abs_root = local_root.expanduser()
        if not abs_root.is_absolute():
            abs_root = (Path.cwd() / abs_root).resolve()
        else:
            abs_root = abs_root.resolve()
        default_abs = abs_root / f"{slug}.jsonl"
        if default_abs.exists():
            ctx["loader"] = "jsonl"
            ctx["data_path"] = os.path.relpath(default_abs, project_root)
            return ctx

    # Fallback: treat as HF hub dataset if it looks like one.
    if _looks_like_hf_hub_id(hub_id):
        ctx["loader"] = "hf_hub"
        ctx["hub"] = explicit_hub or "huggingface"
        ctx["hub_id"] = hub_id
        ctx["subset"] = subset
        ctx["split"] = split
        ctx["trust_remote_code"] = trust_remote_code
        return ctx

    # Last resort: jsonl with conventional relative path.
    if cfg and project_root:
        local_root = cfg.paths.local_datasets_root
        abs_root = local_root.expanduser()
        if not abs_root.is_absolute():
            abs_root = (Path.cwd() / abs_root).resolve()
        else:
            abs_root = abs_root.resolve()
        default_abs = abs_root / f"{slug}.jsonl"
        ctx["loader"] = "jsonl"
        ctx["data_path"] = os.path.relpath(default_abs, project_root)
        return ctx

    ctx["loader"] = "jsonl"
    return ctx


def _render_pipeline_configs_with_context(
    cfg_block: Dict[str, Any],
    slug: str,
    *,
    cfg: Optional[SupportConfig] = None,
    project_root: Optional[Path] = None,
    dataset_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    common = dict(cfg_block)
    common.setdefault("modalities", ["text"])
    common.update(
        _infer_dataset_context(
            cfg_block,
            slug=slug,
            cfg=cfg,
            project_root=project_root,
            dataset_meta=dataset_meta,
        )
    )
    return {
        "openai": _render_template("openai_config.j2", dataset_slug=slug, **common),
        "vllm": _render_template("vllm_config.j2", dataset_slug=slug, **common),
    }


def _assert_metric_class_paths(
    cfg_block: Dict[str, Any],
    project_root: Path,
    *,
    generated_files: Optional[Dict[Path, str]] = None,
) -> None:
    """Reject metrics implementations that point to non-existent class paths.

    - registry id（无冒号/点）不检查
    - class_path 需要可导入，或在本次生成列表中出现
    """

    import importlib.util
    import sys

    metrics = cfg_block.get("metrics") or []
    if not isinstance(metrics, list):
        return

    generated_paths: set[Path] = set()
    if generated_files:
        for p in generated_files:
            try:
                resolved = (p if p.is_absolute() else (project_root / p)).resolve()
                generated_paths.add(resolved)
            except Exception:
                continue

    for m in metrics:
        if not isinstance(m, dict):
            continue
        impl = str(m.get("implementation") or "").strip()
        if not impl or (":" not in impl and "." not in impl):
            continue
        module_name = impl.split(":")[0] if ":" in impl else impl.rsplit(".", 1)[0]
        rel_parts = module_name.split(".")
        candidate = project_root.joinpath(*rel_parts)
        candidate_src = project_root.joinpath("src", *rel_parts)
        if (
            candidate.with_suffix(".py").exists()
            or (candidate / "__init__.py").exists()
            or candidate_src.with_suffix(".py").exists()
            or (candidate_src / "__init__.py").exists()
        ):
            continue
        # If the file is planned to be generated in this run, allow.
        generated_match = (
            candidate.with_suffix(".py").resolve() in generated_paths
            or candidate_src.with_suffix(".py").resolve() in generated_paths
        )
        if generated_match:
            continue
        spec = None
        try:
            # Temporarily allow project_root for import detection.
            sys_path_orig = list(sys.path)
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        finally:
            sys.path = sys_path_orig
        if spec:
            continue
        raise RuntimeError(
            f"Metric implementation '{impl}' 不可导入（未找到模块 {module_name}），且本次未生成对应文件。"
            "请改用已注册的 registry 名，或先实现对应模块后再运行 implement。"
        )

def _validate_agent_files(files: Dict[Path, str]) -> None:
    """Lightweight validation to catch common generation mistakes early."""

    for path, content in files.items():
        norm = str(path).replace("\\", "/")
        if "metrics/builtin" in norm:
            if "class " not in content:
                raise RuntimeError(
                    f"生成的指标文件缺少类定义（需要继承 ComparisonMetric）: {path}"
                )
        if "preprocessors" in norm:
            if "class " not in content:
                raise RuntimeError(f"生成的预处理器文件缺少类定义: {path}")


def run_implement(
    dataset: str,
    *,
    cfg: SupportConfig,
    dry_run: bool = False,
    force: bool = False,
    skip_tests: bool = False,
) -> None:
    dataset_dir = _resolve_dataset_dir(dataset, cfg)
    dataset_meta = _read_inspect_meta(dataset_dir)
    legacy_slug = slugify_dataset_name(dataset_dir.name)
    design_path = dataset_dir / "design.md"
    if not design_path.exists():
        raise FileNotFoundError(f"design.md not found in {dataset_dir}, run design first.")

    design_md = design_path.read_text(encoding="utf-8")
    design_md_for_agent = _strip_auto_preview(design_md)
    cfg_block = parse_support_config(design_md_for_agent)
    slug = artifact_slug_from_dataset_id(str(cfg_block.get("dataset_id") or dataset_meta.get("hub_id") or dataset))
    design_md_for_agent = _normalize_support_config_run_commands(design_md_for_agent, cfg_block, slug)
    cfg_block = parse_support_config(design_md_for_agent)
    normalized_preprocess_name = normalize_preprocess_name(
        str(cfg_block.get("preprocess_name") or ""),
        artifact_slug=slug,
    )
    if normalized_preprocess_name != str(cfg_block.get("preprocess_name") or ""):
        design_md_for_agent = _replace_support_config_scalar(design_md_for_agent, key="preprocess_name", value=normalized_preprocess_name)
        cfg_block = parse_support_config(design_md_for_agent)
    design_md_for_agent = _normalize_design_doc_artifact_refs(design_md_for_agent, slug=slug, legacy_slug=legacy_slug)
    language = cfg_block.get("language") or cfg.language
    project_root = cfg.paths.project_root
    if not project_root.is_absolute():
        project_root = (Path.cwd() / project_root).resolve()
    support_config_yaml = _extract_support_config_yaml(design_md_for_agent)

    # Default behavior: implement is dry-run unless user explicitly opts in (docs use --force).
    effective_dry_run = bool(dry_run or (cfg.execution.dry_run_default and not force))

    ref_preproc = (
        (TEMPLATE_DIR / "reference_preprocessor.py.txt").read_text(encoding="utf-8")
        if (TEMPLATE_DIR / "reference_preprocessor.py.txt").exists()
        else ""
    )
    ref_cfg = (
        (TEMPLATE_DIR / "reference_config.yaml.txt").read_text(encoding="utf-8")
        if (TEMPLATE_DIR / "reference_config.yaml.txt").exists()
        else ""
    )
    sample_format_doc = _load_reference_text(cfg.paths.project_root / "src/gage_eval/assets/datasets/sample.py")
    metrics_base_doc = _load_reference_text(cfg.paths.project_root / "src/gage_eval/metrics/base.py")

    if effective_dry_run:
        planned_paths = [
            str(project_root / "src/gage_eval/assets/datasets/preprocessors" / f"{slug}_preprocessor.py"),
            str(project_root / "src/gage_eval/assets/datasets/preprocessors/custom.py"),
            str(project_root / "src/gage_eval/assets/datasets/preprocessors/__init__.py"),
            str(project_root / "config/custom" / f"{slug}_openai.yaml"),
            str(project_root / "config/custom" / f"{slug}_vllm.yaml"),
        ]
        logger.info("Dry-run: planned file writes:\n" + "\n".join(planned_paths))
        commands = iter_test_commands(cfg_block)
        if commands and not skip_tests:
            logger.info("Dry-run: planned test commands:\n" + "\n".join(commands))
        logger.info("如需实际写入文件，请在命令中添加 --force（同时跳过 git guard）。")
        logger.info(f"Summary (implement dry-run): files={len(planned_paths)}, tests=skipped")
        return

    prompt = _render_template(
        "prompt_implement.j2",
        dataset_slug=slug,
        preprocess_name=cfg_block["preprocess_name"],
        preprocessor_class_name=class_name_from_slug(slug) + "Preprocessor",
        dataset_name=cfg_block.get("dataset_id", slug),
        support_config_yaml=support_config_yaml,
        builtin_metric_ids=sorted(_BUILTIN_METRIC_IDS),
        reference_preprocessor_code=ref_preproc,
        reference_config_yaml=ref_cfg,
        sample_format_doc=sample_format_doc,
        metrics_base_doc=metrics_base_doc,
        language=language,
    )

    files = render_files_from_agent(prompt, cfg)
    _validate_agent_files(files)
    _assert_metric_class_paths(cfg_block, project_root, generated_files=files)

    ensure_git_clean(force=force, cwd=project_root)

    written_files: list[Path] = []

    for rel_path, content in files.items():
        path = rel_path if rel_path.is_absolute() else project_root / rel_path
        resolved = path.resolve()
        if not resolved.is_relative_to(project_root):
            raise RuntimeError(f"Refuse to write outside project_root: {path}")
        if resolved.exists() and not force:
            raise FileExistsError(f"{resolved} exists, use --force to overwrite.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote {path}")
        written_files.append(resolved)

    # Ensure registry wrapper in custom.py
    custom_path = project_root / "src/gage_eval/assets/datasets/preprocessors/custom.py"
    _append_custom_wrapper(
        slug=slug,
        preprocess_name=cfg_block["preprocess_name"],
        dataset_name=str(cfg_block.get("dataset_id") or slug),
        custom_path=custom_path,
    )
    _ensure_preprocessors_custom_import(custom_path.parent / "__init__.py")
    written_files.append(custom_path.resolve())
    written_files.append((custom_path.parent / "__init__.py").resolve())

    metric_impls = _iter_custom_metric_impls(cfg_block)
    if metric_impls:
        metrics_custom_path = project_root / "src/gage_eval/metrics/builtin/custom.py"
        _append_custom_metric_imports(metric_impls=metric_impls, custom_path=metrics_custom_path)
        _ensure_metrics_custom_import(metrics_custom_path.parent / "__init__.py")
        written_files.append(metrics_custom_path.resolve())
        written_files.append((metrics_custom_path.parent / "__init__.py").resolve())

    # Render pipeline configs
    configs = _render_pipeline_configs(cfg_block, slug, cfg=cfg, project_root=project_root, dataset_meta=dataset_meta)
    openai_path = project_root / "config/custom" / f"{slug}_openai.yaml"
    vllm_path = project_root / "config/custom" / f"{slug}_vllm.yaml"
    openai_path.parent.mkdir(parents=True, exist_ok=True)
    if openai_path.exists() and not force:
        raise FileExistsError(f"{openai_path} exists, use --force to overwrite.")
    if vllm_path.exists() and not force:
        raise FileExistsError(f"{vllm_path} exists, use --force to overwrite.")
    openai_path.write_text(configs["openai"].rstrip() + "\n", encoding="utf-8")
    vllm_path.write_text(configs["vllm"].rstrip() + "\n", encoding="utf-8")

    logger.info(f"Wrote configs: {openai_path}, {vllm_path}")
    written_files.extend([openai_path.resolve(), vllm_path.resolve()])

    # 去重，方便最终摘要。
    written_files = list(dict.fromkeys(written_files))

    # Refresh design.md with updated paths + auto preview (no agent call).
    try:
        refreshed = _normalize_mermaid_blocks(design_md_for_agent)
        cfg_block_for_doc = parse_support_config(refreshed)
        refreshed = _normalize_design_doc_artifact_refs(refreshed, slug=slug, legacy_slug=legacy_slug)
        refreshed = _rewrite_landing_plan_table(refreshed, cfg_block_for_doc, slug)
        refreshed = _inject_auto_preview(
            refreshed,
            cfg_block_for_doc,
            slug,
            cfg=cfg,
            project_root=project_root,
            dataset_meta=dataset_meta,
        )
        design_path.write_text(refreshed, encoding="utf-8")
    except Exception as exc:
        logger.warning(f"design.md refresh skipped after implement due to invalid support_config: {exc}")

    if skip_tests:
        logger.info(
            f"Summary (implement): wrote {len(written_files)} files:\n"
            + "\n".join(str(p) for p in written_files)
            + "\nTests: skipped (--skip-tests)."
        )
        return
    commands = iter_test_commands(cfg_block)
    # If tests were generated under project_root/tests but no commands provided, add default pytest.
    has_tests_generated = any(
        str(p).startswith(str((project_root / "tests").resolve())) for p in written_files
    )
    if not commands and has_tests_generated:
        commands = ["pytest -q tests"]
    safe_commands = guard_commands(commands, cfg, confirm=True)
    test_results: list[tuple[str, int]] = []
    for cmd in safe_commands:
        logger.info(f"Executing test command: {cmd}")
        result = subprocess.run(cmd, shell=True, check=False, cwd=str(project_root))
        test_results.append((cmd, result.returncode))

    if not safe_commands:
        test_summary = "no commands configured"
    else:
        failed = [c for c, rc in test_results if rc != 0]
        if failed:
            test_summary = f"failed: {', '.join(failed)}"
        else:
            test_summary = "all passed"

    logger.info(
        f"Summary (implement): wrote {len(written_files)} files:\n"
        + "\n".join(str(p) for p in written_files)
        + f"\nTests: {test_summary}"
    )
