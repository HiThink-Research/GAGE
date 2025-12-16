from __future__ import annotations

from gage_eval.support.pipeline import _render_template


def test_prompt_design_includes_design_doc_template() -> None:
    rendered = _render_template(
        "prompt_design.j2",
        dataset_name="dummy",
        meta_json="{}",
        sample_json="[]",
        schema_json="{}",
        reference_preprocessor_code="",
        reference_config_yaml="",
        sample_format_doc="sample-format",
        metrics_base_doc="metrics-base",
        language="zh",
        design_doc_template="__TEMPLATE_MARK__",
    )
    assert "__TEMPLATE_MARK__" in rendered
