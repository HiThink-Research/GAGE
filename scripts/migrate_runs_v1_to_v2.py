#!/usr/bin/env python3
"""One-time migration from legacy run outputs to AgentKit v2 artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Literal


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.agent_runtime.trace_schema import ArtifactRef, TrialResult
from gage_eval.agent_runtime.trials import aggregate_trial_results


class RunMigrationResult(NamedTuple):
    ok: bool
    output_run_dir: Path | None
    migrated_samples: int
    manual_fixes: list[str]


@dataclass(frozen=True)
class LegacySample:
    record: dict[str, Any]
    index: int
    sample_id: str
    task_id: str
    source_path: Path | None


def migrate_run(
    input_run_dir: Path,
    *,
    output_base_dir: Path,
    run_id: str | None = None,
) -> RunMigrationResult:
    """Migrate one legacy run directory into v2 sample/trial artifacts."""

    input_run_dir = input_run_dir.resolve()
    output_base_dir = output_base_dir.resolve()
    resolved_run_id = _safe_segment(run_id or input_run_dir.name)

    try:
        raw_samples = _read_legacy_samples(input_run_dir)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return RunMigrationResult(
            ok=False,
            output_run_dir=None,
            migrated_samples=0,
            manual_fixes=[f"manual migration required: failed to read legacy run samples: {exc}"],
        )

    samples, manual_fixes = _normalize_legacy_samples(raw_samples)
    if manual_fixes:
        return RunMigrationResult(
            ok=False,
            output_run_dir=None,
            migrated_samples=0,
            manual_fixes=manual_fixes,
        )
    if not samples:
        return RunMigrationResult(
            ok=False,
            output_run_dir=None,
            migrated_samples=0,
            manual_fixes=["manual migration required: no legacy samples found"],
        )

    output_run_dir = output_base_dir / resolved_run_id
    try:
        artifact_sink = RuntimeArtifactSink(base_dir=str(output_base_dir))
        sample_jsonl_records: list[dict[str, Any]] = []
        for sample in samples:
            sample_jsonl_records.append(
                _migrate_sample(
                    sample,
                    artifact_sink=artifact_sink,
                    input_run_dir=input_run_dir,
                    run_id=resolved_run_id,
                )
            )
        _write_root_samples_jsonl(output_run_dir, sample_jsonl_records)
        _write_migration_manifest(
            output_run_dir,
            input_run_dir=input_run_dir,
            migrated_samples=len(sample_jsonl_records),
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _cleanup_partial_output(output_run_dir, output_base_dir=output_base_dir)
        return RunMigrationResult(
            ok=False,
            output_run_dir=None,
            migrated_samples=0,
            manual_fixes=[f"manual migration required: failed to write migrated run: {exc}"],
        )
    return RunMigrationResult(
        ok=True,
        output_run_dir=output_run_dir,
        migrated_samples=len(sample_jsonl_records),
        manual_fixes=[],
    )


def _read_legacy_samples(input_run_dir: Path) -> list[tuple[dict[str, Any], Path | None]]:
    samples_jsonl = input_run_dir / "samples.jsonl"
    if samples_jsonl.exists():
        records: list[tuple[dict[str, Any], Path | None]] = []
        for line in samples_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("samples.jsonl entries must be objects")
            records.append((payload, None))
        return records

    samples_dir = input_run_dir / "samples"
    records = []
    if samples_dir.exists():
        for path in sorted(samples_dir.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: expected JSON object")
            records.append((payload, path))
    return records


def _normalize_legacy_samples(
    raw_samples: list[tuple[dict[str, Any], Path | None]],
) -> tuple[list[LegacySample], list[str]]:
    samples: list[LegacySample] = []
    manual_fixes: list[str] = []
    sanitized_id_sources: dict[tuple[str, str], tuple[str, str, int]] = {}
    for index, (record, source_path) in enumerate(raw_samples, start=1):
        raw_sample_id = _string_or_none(record.get("sample_id"))
        if raw_sample_id is None and source_path is not None:
            raw_sample_id = source_path.stem
        if raw_sample_id is None:
            manual_fixes.append(
                f"manual migration required: samples[{index}] missing sample_id and no file name fallback exists"
            )
            continue

        raw_task_id = _string_or_none(record.get("task_id")) or _string_or_none(record.get("namespace"))
        if raw_task_id is None and source_path is not None and source_path.parent.name != "samples":
            raw_task_id = source_path.parent.name
        if raw_task_id is None:
            manual_fixes.append(
                f"manual migration required: samples[{index}] sample_id={raw_sample_id} missing task_id/namespace"
            )
            continue
        if not _has_legacy_agent_output(record):
            manual_fixes.append(
                f"manual migration required: samples[{index}] sample_id={raw_sample_id} missing agent output"
            )
        if not _has_legacy_verifier_output(record):
            manual_fixes.append(
                f"manual migration required: samples[{index}] sample_id={raw_sample_id} missing verifier output"
            )

        sample_id = _safe_segment(raw_sample_id)
        task_id = _safe_segment(raw_task_id)
        key = (task_id, sample_id)
        previous = sanitized_id_sources.get(key)
        if previous is not None and previous[:2] != (raw_task_id, raw_sample_id):
            manual_fixes.append(
                "manual migration required: sanitized sample/task id collision "
                f"for task_id={task_id!r}, sample_id={sample_id!r}: "
                f"samples[{previous[2]}] raw=({previous[0]!r}, {previous[1]!r}) and "
                f"samples[{index}] raw=({raw_task_id!r}, {raw_sample_id!r})"
            )
        sanitized_id_sources.setdefault(key, (raw_task_id, raw_sample_id, index))

        samples.append(
            LegacySample(
                record=record,
                index=index,
                sample_id=sample_id,
                task_id=task_id,
                source_path=source_path,
            )
        )
    return samples, manual_fixes


def _has_legacy_agent_output(record: dict[str, Any]) -> bool:
    for key in ("scheduler_result", "runtime_result", "model_output", "agent_output"):
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, dict):
            if value:
                return True
        elif value not in (None, ""):
            return True
    return False


def _has_legacy_verifier_output(record: dict[str, Any]) -> bool:
    if any(
        key in record and isinstance(record[key], dict) and record[key]
        for key in ("verifier_result", "judge_output")
    ):
        return True
    if any(
        key in record and record[key] is not None
        for key in ("reward", "score", "resolved", "passed", "pass")
    ):
        return True
    metrics = record.get("metrics")
    return isinstance(metrics, dict) and bool(metrics)


def _migrate_sample(
    sample: LegacySample,
    *,
    artifact_sink: RuntimeArtifactSink,
    input_run_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    record = dict(sample.record)
    dut_id = _safe_segment(
        _string_or_none(record.get("dut_id"))
        or _string_or_none(record.get("adapter_id"))
        or "migrated_dut"
    )
    trial_id = "trial_0001"
    effective_config_ref = artifact_sink.write_effective_config(
        run_id=run_id,
        task_id=sample.task_id,
        sample_id=sample.sample_id,
        final_config={
            "migration": {
                "source": "legacy_run",
                "input_run_dir": str(input_run_dir),
                "source_sample_path": str(sample.source_path) if sample.source_path else "samples.jsonl",
            }
        },
        source_layers=[
            {
                "name": "legacy_run_sample",
                "values": {"keys": sorted(str(key) for key in record)},
            }
        ],
    )
    trace_ref = artifact_sink.append_trace_event(
        run_id=run_id,
        task_id=sample.task_id,
        sample_id=sample.sample_id,
        trial_id=trial_id,
        actor="runtime",
        event_type="legacy_run.migrated",
        payload={
            "source_run": input_run_dir.name,
            "source_sample_index": sample.index,
            "legacy_keys": sorted(str(key) for key in record),
        },
    )
    scheduler_result = _scheduler_result_from_legacy_record(record)
    verifier_result = _verifier_result_from_legacy_record(record)
    scheduler_ref = artifact_sink.write_artifact(
        run_id=run_id,
        task_id=sample.task_id,
        sample_id=sample.sample_id,
        trial_id=trial_id,
        owner="agent",
        name="scheduler_result.json",
        content=scheduler_result,
        mime_type="application/json",
    )
    verifier_ref = artifact_sink.write_artifact(
        run_id=run_id,
        task_id=sample.task_id,
        sample_id=sample.sample_id,
        trial_id=trial_id,
        owner="verifier",
        name="verifier_result.json",
        content=verifier_result,
        mime_type="application/json",
    )
    legacy_ref = artifact_sink.write_artifact(
        run_id=run_id,
        task_id=sample.task_id,
        sample_id=sample.sample_id,
        trial_id=trial_id,
        owner="infra",
        name="legacy_sample.json",
        content=record,
        mime_type="application/json",
    )
    artifact_refs = _artifact_refs(scheduler_ref, verifier_ref, legacy_ref)
    failure = _failure_from_results(scheduler_result, verifier_result)
    trial_result = TrialResult(
        trial_id=trial_id,
        status=_status_from_results(scheduler_result, verifier_result, failure),
        scheduler_result=to_json_compatible(scheduler_result),
        verifier_result=to_json_compatible(verifier_result),
        environment_descriptor=to_json_compatible(_environment_descriptor_from_legacy_record(record)),
        artifact_refs=artifact_refs,
        trace_ref=trace_ref,
        failure=to_json_compatible(failure) if failure is not None else None,
    )
    trial_record_ref = artifact_sink.write_trial_record(
        run_id=run_id,
        task_id=sample.task_id,
        sample_id=sample.sample_id,
        trial_id=trial_id,
        trial_result=trial_result,
    )
    trial_result.artifact_refs.append(trial_record_ref)
    aggregate = aggregate_trial_results([trial_result], aggregation="single")
    aggregate_ref = artifact_sink.write_trial_aggregate(
        run_id=run_id,
        task_id=sample.task_id,
        sample_id=sample.sample_id,
        aggregate=aggregate,
    )
    sample_artifacts = [
        effective_config_ref,
        trace_ref,
        scheduler_ref,
        verifier_ref,
        legacy_ref,
        trial_record_ref,
        aggregate_ref,
    ]
    sample_record = {
        "run_id": run_id,
        "task_id": sample.task_id,
        "sample_id": sample.sample_id,
        "dut_id": dut_id,
        "input_ref": {
            "source": "legacy_run",
            "source_run": input_run_dir.name,
            "source_sample_index": sample.index,
            "source_sample_path": str(sample.source_path) if sample.source_path else "samples.jsonl",
        },
        "trial_policy": {
            "trials": 1,
            "environment_scope": "per_trial",
            "parallelism": 1,
            "aggregation": "single",
        },
        "trial_results": [trial_result.model_dump(mode="python")],
        "aggregate_result": aggregate.to_dict(),
        "scheduler_result": trial_result.scheduler_result,
        "verifier_result": trial_result.verifier_result,
        "environment_descriptor": trial_result.environment_descriptor,
        "effective_config_ref": effective_config_ref.model_dump(mode="python"),
        "artifacts": [ref.model_dump(mode="python") for ref in sample_artifacts],
        "status": trial_result.status,
        "failure": trial_result.failure,
    }
    sample_record_ref = artifact_sink.write_artifact(
        run_id=run_id,
        task_id=sample.task_id,
        sample_id=sample.sample_id,
        owner="infra",
        name="sample_record.json",
        content=sample_record,
        mime_type="application/json",
    )
    if not isinstance(sample_record_ref, ArtifactRef):  # pragma: no cover - guarded by v2 inputs
        raise TypeError("sample_record_ref must be an ArtifactRef")
    return {
        "sample_id": sample.sample_id,
        "task_id": sample.task_id,
        "dut_id": dut_id,
        "status": trial_result.status,
        **_flatten_samples_projection(aggregate.samples_jsonl_projection),
        "sample_record_ref": sample_record_ref.model_dump(mode="python"),
    }


def _scheduler_result_from_legacy_record(record: dict[str, Any]) -> dict[str, Any]:
    scheduler_result = record.get("scheduler_result") or record.get("runtime_result")
    if isinstance(scheduler_result, dict):
        result = dict(scheduler_result)
    else:
        model_output = record.get("model_output") or record.get("agent_output") or {}
        result = {
            "status": "completed",
            "agent_output": to_json_compatible(model_output if isinstance(model_output, dict) else {"answer": model_output}),
        }
    result.setdefault("status", "completed")
    return to_json_compatible(result)


def _verifier_result_from_legacy_record(record: dict[str, Any]) -> dict[str, Any]:
    verifier_result = record.get("verifier_result") or record.get("judge_output")
    result = dict(verifier_result) if isinstance(verifier_result, dict) else {}
    for key in ("reward", "score", "resolved", "passed", "pass"):
        if key in record and key not in result:
            result[key] = record[key]
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for metric_payload in metrics.values():
            if not isinstance(metric_payload, dict):
                continue
            value = metric_payload.get("value")
            metric_id = metric_payload.get("metric_id")
            if metric_id in {"reward", "score", "resolved", "passed"} and metric_id not in result:
                result[str(metric_id)] = value
    result.setdefault("status", "completed")
    return to_json_compatible(result)


def _environment_descriptor_from_legacy_record(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("environment_descriptor", "resource_lease", "sandbox"):
        value = record.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _failure_from_results(
    scheduler_result: dict[str, Any],
    verifier_result: dict[str, Any],
) -> dict[str, Any] | None:
    for payload in (verifier_result, scheduler_result):
        failure = payload.get("failure")
        if isinstance(failure, dict):
            return dict(failure)
        failure_code = payload.get("failure_code")
        if failure_code:
            return {
                "failure_code": str(failure_code),
                "failure_domain": payload.get("failure_domain") or "legacy_run",
            }
    return None


def _status_from_results(
    scheduler_result: dict[str, Any],
    verifier_result: dict[str, Any],
    failure: dict[str, Any] | None,
) -> Literal["completed", "failed", "aborted"]:
    for payload in (verifier_result, scheduler_result):
        status = payload.get("status")
        if status in {"completed", "failed", "aborted"}:
            return status  # type: ignore[return-value]
    return "failed" if failure is not None else "completed"


def _artifact_refs(*values: Any) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for value in values:
        if isinstance(value, ArtifactRef):
            refs.append(value)
    return refs


def _flatten_samples_projection(projection: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in projection.items():
        if isinstance(value, dict) and "value" in value and "source_trial_id" in value:
            flattened[key] = value["value"]
            flattened[f"{key}_source_trial_id"] = value["source_trial_id"]
        else:
            flattened[key] = value
    return to_json_compatible(flattened)


def _write_root_samples_jsonl(output_run_dir: Path, records: list[dict[str, Any]]) -> None:
    output_run_dir.mkdir(parents=True, exist_ok=True)
    with (output_run_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_json_compatible(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_migration_manifest(
    output_run_dir: Path,
    *,
    input_run_dir: Path,
    migrated_samples: int,
) -> None:
    payload = {
        "source": "legacy_run",
        "input_run_dir": str(input_run_dir),
        "migrated_samples": migrated_samples,
    }
    (output_run_dir / "migration_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cleanup_partial_output(output_run_dir: Path, *, output_base_dir: Path) -> None:
    try:
        output_run_dir = output_run_dir.resolve()
        output_base_dir = output_base_dir.resolve()
    except OSError:
        return
    if output_run_dir == output_base_dir or output_base_dir not in output_run_dir.parents:
        return
    if output_run_dir.exists() and output_run_dir.is_dir():
        shutil.rmtree(output_run_dir, ignore_errors=True)


def _safe_segment(value: str) -> str:
    sanitized = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in str(value))
    sanitized = sanitized.strip("._")
    return sanitized or "generated"


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True, type=Path, help="Legacy run directory")
    parser.add_argument("--output-base", required=True, type=Path, help="Destination runs base directory")
    parser.add_argument("--run-id", help="Optional output run id. Defaults to input run directory name.")
    args = parser.parse_args(argv)

    result = migrate_run(args.input_run, output_base_dir=args.output_base, run_id=args.run_id)
    if result.ok:
        print(f"migrated {result.migrated_samples} sample(s) to {result.output_run_dir}")
        return 0
    for message in result.manual_fixes:
        print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
