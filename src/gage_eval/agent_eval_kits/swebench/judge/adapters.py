from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gage_eval.agent_runtime.verifier.contracts import VerifierInput, VerifierResult

from .executor import SwebenchExecutionRequest, execute_swebench_verifier
from .failure_categories import resolve_swebench_failure_category
from .patch_extraction import resolve_patch


@dataclass
class SwebenchVerifierAdapter:
    """Kit-owned SWE-bench verifier adapter."""

    scripts_dir: str | None = None
    test_timeout_s: int = 900
    judge_source: str = "swebench.verifier_adapter.run"
    swebench_pro_mode: bool = True

    def preflight(self, verifier_input: VerifierInput) -> VerifierResult | None:
        metadata_failure = _validate_metadata(dict(verifier_input.sample or {}))
        if metadata_failure is None:
            return None
        failure_code, failure_reason = metadata_failure
        return _failed(failure_code, failure_reason=failure_reason)

    def run(self, verifier_input: VerifierInput) -> VerifierResult:
        return _run_coroutine_sync(self._arun(verifier_input))

    async def _arun(self, verifier_input: VerifierInput) -> VerifierResult:
        sample = dict(verifier_input.sample or {})
        metadata_failure = _validate_metadata(sample)
        if metadata_failure is not None:
            failure_code, failure_reason = metadata_failure
            return _failed(failure_code, failure_reason=failure_reason)

        model_output = verifier_input.scheduler_result.get("agent_output") or {}
        if not isinstance(model_output, Mapping):
            model_output = {}
        runtime_context = dict(verifier_input.runtime_context or {})
        environment = _resolve_environment(runtime_context)
        if environment is None:
            return _failed(
                "environment.unavailable",
                failure_reason="sandbox_judge_error",
                summary="SWE-bench verifier environment is unavailable",
            )

        trace = runtime_context.get("trace")
        patch = await resolve_patch(
            model_output=model_output,
            sample=sample,
            environment=environment,
            trace=trace,
        )
        if patch.failure_code:
            return _failed(
                patch.failure_code,
                failure_reason="missing_patch",
                summary="SWE-bench submission patch is missing",
            )

        scripts = _load_run_scripts(
            Path(str(verifier_input.verifier_resources.get("scripts_dir") or self.scripts_dir or _default_scripts_dir())),
            _resolve_instance_id(sample),
        )
        if scripts is None:
            return _failed(
                "input_projection.workflow.prepare_failed",
                failure_reason="missing_run_scripts",
                summary="SWE-bench run scripts are missing",
            )

        timeout_s = int(verifier_input.verifier_resources.get("test_timeout_s") or self.test_timeout_s)
        result = await execute_swebench_verifier(
            environment=environment,
            request=SwebenchExecutionRequest(
                sample=sample,
                patch=patch.patch,
                run_script=scripts["run_script"],
                parser_script=scripts["parser_script"],
                test_patch=None if self.swebench_pro_mode else _get_meta(sample, "test_patch"),
                timeout_s=timeout_s,
                dockerfiles_dir=verifier_input.verifier_resources.get("dockerfiles_dir"),
                strict_patch_apply=self.swebench_pro_mode,
            ),
        )
        result.setdefault("status", "completed")
        result.setdefault("failure_category", resolve_swebench_failure_category(result))
        result.setdefault("metric", _metric(result))
        result.setdefault("artifact_refs", _artifact_refs(result))
        _emit_verifier_result(trace, verifier_input, result)
        return VerifierResult(status="completed" if result.get("resolved") else "failed", payload=result)


def _resolve_environment(runtime_context: Mapping[str, Any]) -> Any | None:
    lease = runtime_context.get("environment_lease")
    if lease is not None:
        return lease
    environment = runtime_context.get("environment")
    if environment is not None:
        return environment
    return None


def _validate_metadata(sample: Mapping[str, Any]) -> tuple[str, str] | None:
    if not _get_meta(sample, "base_commit"):
        return ("config.kit_schema.validation_failed", "missing_base_commit")
    if not _resolve_instance_id(sample) or not _get_meta(sample, "repo"):
        return ("input_projection.workflow.prepare_failed", "missing_metadata")
    return None


def _load_run_scripts(scripts_dir: Path, instance_id: str) -> dict[str, str] | None:
    run_dir = scripts_dir / instance_id
    if not run_dir.exists() and not instance_id.startswith("instance_"):
        run_dir = scripts_dir / f"instance_{instance_id}"
    run_script = run_dir / "run_script.sh"
    parser_script = run_dir / "parser.py"
    if not run_script.exists() or not parser_script.exists():
        return None
    return {
        "run_script": run_script.read_text(encoding="utf-8", errors="replace"),
        "parser_script": parser_script.read_text(encoding="utf-8", errors="replace"),
    }


def _failed(
    failure_code: str,
    *,
    failure_reason: str,
    summary: str | None = None,
) -> VerifierResult:
    payload = {
        "status": "failed",
        "resolved": False,
        "score": 0.0,
        "failure_reason": failure_reason,
        "failure_code": failure_code,
        "summary": summary or failure_reason,
        "failure_category": resolve_swebench_failure_category(
            {"failure_code": failure_code, "failure_reason": failure_reason}
        ),
        "metric": {"score": 0.0, "resolved": False},
        "artifact_refs": [],
    }
    return VerifierResult(status="failed", payload=payload)


def _metric(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "score": float(result.get("score") or (1.0 if result.get("resolved") else 0.0)),
        "resolved": bool(result.get("resolved")),
        "failure_reason": result.get("failure_reason"),
    }


def _artifact_refs(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = result.get("artifact_refs")
    if isinstance(refs, list) and refs:
        return [dict(ref) for ref in refs if isinstance(ref, Mapping)]
    return [{"owner": "verifier", "name": "verifier_result.json", "path": "verifier/result.json"}]


def _emit_verifier_result(trace: Any, verifier_input: VerifierInput, result: Mapping[str, Any]) -> None:
    emit = getattr(trace, "emit", None)
    if not callable(emit):
        return
    emit(
        "verifier.result",
        {
            "metric": dict(result.get("metric") or _metric(result)),
            "verifier_result": dict(result),
            "artifact_refs": _artifact_refs(result),
        },
        sample_id=verifier_input.sample_id,
    )


def _get_meta(sample: Mapping[str, Any], key: str) -> Any:
    metadata = sample.get("metadata") or {}
    if isinstance(metadata, Mapping):
        value = metadata.get(key)
        if value is not None:
            return value
    return sample.get(key)


def _resolve_instance_id(sample: Mapping[str, Any]) -> str:
    return str(_get_meta(sample, "instance_id") or sample.get("id") or "")


def _default_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[5] / "third_party" / "swebench_pro" / "run_scripts"


def _run_coroutine_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()
