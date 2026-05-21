from __future__ import annotations

from typing import Any

from gage_eval.reporting.contracts import SummaryGeneratorResult


class SummaryExtensionRunner:
    def run(self, generators: list[Any], context: dict[str, Any]) -> SummaryGeneratorResult:
        combined = SummaryGeneratorResult(diagnostics={"warnings": [], "errors": []})
        seen_sections: set[str] = set()
        for generator in generators:
            generator_id = str(getattr(generator, "name", generator.__class__.__name__))
            try:
                result = generator.generate(context)
            except Exception as exc:  # pragma: no cover - exact exception type is extension-owned
                combined.diagnostics.setdefault("warnings", []).append(
                    {
                        "code": "report_pack.summary_generator_failed",
                        "generator_id": generator_id,
                        "message": str(exc),
                    }
                )
                continue
            if result is None:
                continue
            if isinstance(result, dict):
                _merge_legacy_payload(combined.legacy_payload, result)
                continue
            generator_id = str(getattr(result, "generator_id", None) or result.extra.get("generator_id") or generator_id)
            _merge_legacy_payload(combined.legacy_payload, getattr(result, "legacy_payload", None) or result.extra.get("legacy_payload"))
            for section in result.summary_sections:
                canonical = dict(section)
                raw_id = str(canonical.get("section_id") or "section")
                section_id = raw_id if raw_id.startswith(f"{generator_id}/") else f"{generator_id}/{raw_id}"
                canonical["section_id"] = section_id
                if section_id in seen_sections:
                    combined.diagnostics.setdefault("warnings", []).append(
                        {
                            "code": "report_pack.section_id_duplicate",
                            "generator_id": generator_id,
                            "section_id": section_id,
                        }
                    )
                    continue
                seen_sections.add(section_id)
                combined.summary_sections.append(canonical)
            combined.attention_cases.extend(result.attention_cases)
            combined.outliers.extend(result.outliers)
            combined.case_details.update(result.case_details)
            combined.failure_clusters.extend(result.failure_clusters)
            combined.evidence_ref_ids.extend(result.evidence_ref_ids)
        return combined


def _merge_legacy_payload(target: dict[str, Any], payload: Any) -> None:
    if isinstance(payload, dict):
        target.update(payload)
