#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SAMPLE_INFRA = ("effective_config.json", "sample_record.json", "trial_aggregate.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AgentKit v2 run artifact layout.")
    parser.add_argument("run", help="Run id or path to a run directory.")
    parser.add_argument("--runs-dir", default="runs", help="Base runs directory when RUN is a run id.")
    parser.add_argument("--require-no-failure", action="store_true", help="Fail if sample.failed events are present.")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run, Path(args.runs_dir))
    findings = validate_run(run_dir, require_no_failure=args.require_no_failure)
    if findings:
        for finding in findings:
            print(f"[FAIL] {finding}")
        return 1
    print(f"[OK] AgentKit v2 layout validated: {run_dir}")
    return 0


def validate_run(run_dir: Path, *, require_no_failure: bool = False) -> list[str]:
    findings: list[str] = []
    if not run_dir.exists():
        return [f"run directory does not exist: {run_dir}"]
    if (run_dir / "samples" / "runtime").exists():
        findings.append("legacy runtime artifact path exists: samples/runtime")

    artifacts_root = run_dir / "artifacts"
    if not artifacts_root.exists():
        findings.append("artifacts/ directory is missing")
        return findings

    sample_dirs = [
        sample_dir
        for task_dir in artifacts_root.iterdir()
        if task_dir.is_dir()
        for sample_dir in task_dir.iterdir()
        if sample_dir.is_dir()
    ]
    if not sample_dirs:
        findings.append("no sample artifact directories found under artifacts/")
        return findings

    for sample_dir in sample_dirs:
        for child in sample_dir.iterdir():
            if child.name not in {"infra", "trials"}:
                findings.append(f"unexpected sample-root artifact: {child}")
        infra = sample_dir / "infra"
        for name in REQUIRED_SAMPLE_INFRA:
            if not (infra / name).is_file():
                findings.append(f"missing sample infra artifact: {infra / name}")
        sample_record_path = infra / "sample_record.json"
        if sample_record_path.is_file():
            record = _read_json(sample_record_path, findings)
            if isinstance(record, dict):
                trial_results = record.get("trial_results")
                if not isinstance(trial_results, list) or not trial_results:
                    findings.append(f"sample_record has no trial_results: {sample_record_path}")
        trials_dir = sample_dir / "trials"
        if not trials_dir.exists():
            findings.append(f"missing trials directory: {trials_dir}")
            continue
        for trial_dir in sorted(path for path in trials_dir.iterdir() if path.is_dir()):
            for relative in (
                Path("infra") / "trace.jsonl",
                Path("infra") / "trial_result.json",
                Path("verifier") / "verifier_result.json",
            ):
                if not (trial_dir / relative).is_file():
                    findings.append(f"missing trial artifact: {trial_dir / relative}")

    event_failures = _run_event_failure_codes(run_dir / "events.jsonl")
    if require_no_failure and event_failures:
        findings.append(f"run events contain failure signal(s): {sorted(event_failures)}")
    return findings


def _resolve_run_dir(value: str, runs_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.exists() or path.parent != Path("."):
        return path
    return runs_dir.expanduser() / value


def _read_json(path: Path, findings: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        findings.append(f"invalid JSON at {path}: {type(exc).__name__}: {exc}")
        return None


def _run_event_failure_codes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    codes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "sample.failed":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            for key in ("failure_code", "verifier_failure_code", "scheduler_failure_code"):
                value = payload.get(key)
                if value:
                    codes.add(str(value))
    return codes


if __name__ == "__main__":
    raise SystemExit(main())
