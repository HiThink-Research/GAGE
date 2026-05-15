from __future__ import annotations

from pathlib import Path
from typing import Any

from gage_eval.reporting.contracts import EvidenceRef, ReportContext
from gage_eval.reporting.persistence.manifest_writer import ManifestWriter
from gage_eval.reporting.persistence.summary_writer import SummaryWriter
from gage_eval.reporting.privacy import SecretFilter
from gage_eval.reporting.rendering._context import atomic_write_text, deterministic_json, normalize_context
from gage_eval.reporting.rendering.markdown_renderer import MarkdownRenderer
from gage_eval.reporting.rendering.prompt_renderer import PromptRenderer
from gage_eval.reporting.rendering.static_renderer import StaticReportRenderer


_SECRET_FILTER = SecretFilter()


class ReportPackBuilder:
    """Builds the static report pack assets for a run directory."""

    def __init__(
        self,
        *,
        markdown_renderer: MarkdownRenderer | None = None,
        html_renderer: StaticReportRenderer | None = None,
        prompt_renderer: PromptRenderer | None = None,
        manifest_writer: ManifestWriter | None = None,
        summary_writer: SummaryWriter | None = None,
    ) -> None:
        self._markdown_renderer = markdown_renderer or MarkdownRenderer()
        self._html_renderer = html_renderer or StaticReportRenderer()
        self._prompt_renderer = prompt_renderer or PromptRenderer()
        self._manifest_writer = manifest_writer or ManifestWriter()
        self._summary_writer = summary_writer or SummaryWriter()

    def write(
        self,
        run_dir: str | Path,
        context: ReportContext | dict[str, Any],
        *,
        enabled: bool = True,
        cache: Any | None = None,
        summary_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not enabled:
            diagnostics = {"report_pack_status": "disabled", "report_pack_path": None}
            if cache is not None and summary_payload is not None:
                self._summary_writer.write(
                    cache,
                    summary_payload,
                    report_pack_diagnostics=diagnostics,
                )
            return diagnostics

        base = Path(run_dir)
        pack = base / "report_pack"
        payload = normalize_context(context)
        diagnostics = dict(payload.get("diagnostics") or {})
        status = str(diagnostics.get("report_pack_status") or "completed")
        diagnostics["report_pack_status"] = status
        diagnostics["report_pack_path"] = str(pack)
        payload["diagnostics"] = diagnostics

        payload = _safe_value(payload, diagnostics, path="report_context.json")
        payload["diagnostics"] = diagnostics
        markdown = _safe_text(
            self._markdown_renderer.render(payload),
            diagnostics,
            path="report_context.md",
        )
        html = _safe_text(
            self._html_renderer.render(payload),
            diagnostics,
            path="report.html",
        )
        prompt = _safe_text(
            self._prompt_renderer.render(payload),
            diagnostics,
            path="prompt.txt",
        )
        payload["diagnostics"] = diagnostics

        context_path = atomic_write_text(
            pack / "report_context.json",
            deterministic_json(payload),
        )
        markdown_path = atomic_write_text(pack / "report_context.md", markdown)
        html_path = atomic_write_text(pack / "report.html", html)
        prompt_path = atomic_write_text(pack / "prompt.txt", prompt)
        diagnostics_path = atomic_write_text(
            pack / "diagnostics.json",
            deterministic_json(diagnostics),
        )

        evidence_refs = [
            EvidenceRef.from_dict(item) if isinstance(item, dict) else item
            for item in payload.get("evidence_refs", [])
        ]
        self._manifest_writer.write(
            pack,
            run_id=str((payload.get("run") or {}).get("run_id") or "unknown"),
            evidence_refs=evidence_refs,
            files=[
                context_path,
                markdown_path,
                html_path,
                prompt_path,
                diagnostics_path,
            ],
        )
        return diagnostics


def _safe_value(value: Any, diagnostics: dict[str, Any], *, path: str) -> Any:
    result = _SECRET_FILTER.redact(value)
    if result.redacted:
        _append_redaction_warning(diagnostics, path=path, finding_count=len(result.findings))
    return result.value


def _safe_text(text: str, diagnostics: dict[str, Any], *, path: str) -> str:
    result = _SECRET_FILTER.redact(text)
    if result.redacted:
        _append_redaction_warning(diagnostics, path=path, finding_count=len(result.findings))
    return str(result.value)


def _append_redaction_warning(diagnostics: dict[str, Any], *, path: str, finding_count: int) -> None:
    warnings = diagnostics.setdefault("warnings", [])
    warnings.append(
        {
            "code": "report_pack.secret_redacted",
            "path": path,
            "finding_count": finding_count,
        }
    )
