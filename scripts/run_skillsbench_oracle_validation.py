"""Run full SkillsBench oracle validation using official solutions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import shutil
import shlex
import subprocess
from typing import Any, Dict, List
import uuid

from loguru import logger

from gage_eval.assets.datasets.loaders.skillsbench_loader import SkillsBenchHarborLoader
from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.role.judge.skillsbench_evaluate import SkillsBenchEvaluate
from gage_eval.sandbox.docker_runtime import (
    build_docker_run_command,
    ensure_docker_image,
    normalize_runtime_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SkillsBench end-to-end by replaying official solutions through the GAGE verifier."
    )
    parser.add_argument(
        "--output-dir",
        default="runs/skillsbench_oracle_validation",
        help="Directory to store validation artifacts.",
    )
    parser.add_argument(
        "--local-repo-dir",
        default=None,
        help="Optional Harbor checkout root. Defaults to the loader's standard local path.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Harbor revision to use when auto-downloading the dataset.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    parser.add_argument(
        "--include-task",
        action="append",
        default=[],
        help="Optional task filter. Repeat for multiple tasks.",
    )
    parser.add_argument(
        "--docker-bin",
        default="docker",
        help="Docker binary to use for image builds and verifier execution.",
    )
    parser.add_argument(
        "--default-timeout-s",
        type=int,
        default=1800,
        help="Fallback verifier timeout.",
    )
    parser.add_argument(
        "--min-build-timeout-s",
        type=int,
        default=3600,
        help="Raise task-declared build timeout to at least this many seconds.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for parallel oracle validation.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total shard count for parallel oracle validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(args)
    samples = _select_shard(
        samples,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    logger.info("Loaded {} SkillsBench tasks for oracle validation", len(samples))

    judge = SkillsBenchEvaluate(
        docker_bin=args.docker_bin,
        default_timeout_s=args.default_timeout_s,
    )
    results: List[Dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    skip_count = 0

    for index, sample in enumerate(samples, start=1):
        task_id = str(sample.get("id") or sample.get("sample_id") or index)
        logger.info("[{}/{}] validating {}", index, len(samples), task_id)
        sample_dir = output_dir / "samples" / task_id
        agent_workspace_dir = sample_dir / "agent" / "workspace"
        verifier_logs_dir = sample_dir / "verifier" / "logs"
        verifier_workspace_dir = sample_dir / "verifier" / "workspace"
        verifier_stdout_file = sample_dir / "verifier" / "stdout.log"
        verifier_stderr_file = sample_dir / "verifier" / "stderr.log"
        result_file = sample_dir / "verifier" / "result.json"
        sample_dir.mkdir(parents=True, exist_ok=True)

        skillsbench_meta = ((sample.get("metadata") or {}).get("skillsbench") or {})
        solution_dir = Path(str(skillsbench_meta.get("solution_dir") or "")).expanduser()
        if not solution_dir.exists():
            result = {
                "task_id": task_id,
                "status": "skipped",
                "failure_reason": "missing_solution_dir",
                "solution_dir": str(solution_dir),
            }
            skip_count += 1
            _write_json(result_file, result)
            results.append(result)
            continue

        prepared = _materialize_solution_workspace(
            sample=sample,
            solution_dir=solution_dir,
            agent_workspace_dir=agent_workspace_dir,
            docker_bin=args.docker_bin,
            default_timeout_s=args.default_timeout_s,
        )
        if prepared is not None:
            fail_count += 1
            _write_json(result_file, prepared)
            results.append(prepared)
            continue
        payload = {
            "sample": sample,
            "params": {
                "docker_bin": args.docker_bin,
                "timeout_sec": args.default_timeout_s,
            },
            "artifact_paths": {
                "agent_workspace_dir": str(agent_workspace_dir),
                "verifier_logs_dir": str(verifier_logs_dir),
                "verifier_workspace_dir": str(verifier_workspace_dir),
                "verifier_stdout_file": str(verifier_stdout_file),
                "verifier_stderr_file": str(verifier_stderr_file),
            },
        }
        result = dict(judge.invoke(payload))
        result["task_id"] = task_id
        if result.get("resolved") is True:
            result["status"] = "pass"
            pass_count += 1
        else:
            result["status"] = "fail"
            fail_count += 1
        _write_json(result_file, result)
        results.append(result)

    summary = {
        "task_count": len(samples),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "pass_rate": (pass_count / len(samples)) if samples else 0.0,
        "output_dir": str(output_dir),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "results": results,
    }
    _write_json(output_dir / "summary.json", summary)
    logger.info(
        "SkillsBench oracle validation complete: pass={} fail={} skip={} output={}",
        pass_count,
        fail_count,
        skip_count,
        output_dir,
    )
    return 0 if fail_count == 0 else 1


def _load_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "auto_download": True,
        "revision": args.revision,
    }
    if args.local_repo_dir:
        params["local_repo_dir"] = args.local_repo_dir
    if args.max_tasks:
        params["limit"] = args.max_tasks
    if args.include_task:
        params["include_tasks"] = list(args.include_task)
    if args.min_build_timeout_s:
        params["min_build_timeout_sec"] = args.min_build_timeout_s
    spec = DatasetSpec(
        dataset_id="skillsbench_oracle_validation",
        loader="skillsbench_harbor",
        params=params,
    )
    loader = SkillsBenchHarborLoader(spec)
    source = loader.load(None)
    return [_sample_to_dict(record) for record in source.records]


def _select_shard(
    samples: List[Dict[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
) -> List[Dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if shard_count == 1:
        return samples
    return samples[shard_index::shard_count]


def _prepare_workspace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _materialize_solution_workspace(
    *,
    sample: Dict[str, Any],
    solution_dir: Path,
    agent_workspace_dir: Path,
    docker_bin: str,
    default_timeout_s: int,
) -> Dict[str, Any] | None:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    skillsbench_meta = metadata.get("skillsbench") if isinstance(metadata.get("skillsbench"), dict) else {}
    task_id = str(skillsbench_meta.get("task_id") or sample.get("id") or "unknown")
    runtime_configs = normalize_runtime_configs(
        ((sample.get("sandbox") or {}).get("runtime_configs") or {})
    )
    image = str(runtime_configs.get("image") or skillsbench_meta.get("image") or "")
    workdir = str(runtime_configs.get("workdir") or skillsbench_meta.get("workdir") or "/app")
    timeout_sec = int(skillsbench_meta.get("agent_timeout_sec") or default_timeout_s)
    if not image:
        return {
            "task_id": task_id,
            "status": "fail",
            "failure_reason": "missing_image",
        }
    try:
        ensure_docker_image(runtime_configs)
    except Exception as exc:
        return {
            "task_id": task_id,
            "status": "fail",
            "failure_reason": "image_build_failed",
            "error": str(exc),
        }

    container_name = f"gage-skillsbench-oracle-{task_id}-{uuid.uuid4().hex[:8]}"
    started = subprocess.run(
        build_docker_run_command(
            image=image,
            container_name=container_name,
            runtime_configs={
                **runtime_configs,
                "docker_bin": docker_bin,
                "detach": True,
                "auto_remove": True,
                "command": ["sleep", "infinity"],
                "workdir": workdir,
            },
            resources=sample.get("resources") if isinstance(sample.get("resources"), dict) else {},
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if started.returncode != 0:
        return {
            "task_id": task_id,
            "status": "fail",
            "failure_reason": "container_start_failed",
            "error": (started.stderr or started.stdout).strip(),
        }
    container_id = started.stdout.strip() or container_name
    try:
        _docker_exec(
            docker_bin=docker_bin,
            container=container_id,
            command=_prepare_solution_mounts_command(workdir),
            timeout_sec=min(120, timeout_sec),
            workdir=workdir,
        )
        _docker_cp_to_container(
            docker_bin=docker_bin,
            source=solution_dir,
            container=container_id,
            destination="/solution",
        )
        completed = _docker_exec(
            docker_bin=docker_bin,
            container=container_id,
            command="bash /solution/solve.sh",
            timeout_sec=timeout_sec,
            workdir=workdir,
            capture_output=True,
        )
        if completed.returncode != 0:
            return {
                "task_id": task_id,
                "status": "fail",
                "failure_reason": "solution_execution_failed",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        _prepare_empty_dir(agent_workspace_dir)
        copied = subprocess.run(
            [docker_bin, "cp", f"{container_id}:{workdir}/.", str(agent_workspace_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if copied.returncode != 0:
            return {
                "task_id": task_id,
                "status": "fail",
                "failure_reason": "workspace_export_failed",
                "error": (copied.stderr or copied.stdout).strip(),
            }
        return None
    except subprocess.TimeoutExpired:
        return {
            "task_id": task_id,
            "status": "fail",
            "failure_reason": "solution_timeout",
        }
    finally:
        subprocess.run(
            [docker_bin, "rm", "-f", container_id],
            capture_output=True,
            text=True,
            check=False,
        )


def _docker_exec(
    *,
    docker_bin: str,
    container: str,
    command: str,
    timeout_sec: int,
    workdir: str,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [docker_bin, "exec", "-w", workdir, container, "/bin/sh", "-lc", command],
        capture_output=capture_output,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def _docker_cp_to_container(
    *,
    docker_bin: str,
    source: Path,
    container: str,
    destination: str,
) -> None:
    subprocess.run(
        [docker_bin, "cp", f"{source}/.", f"{container}:{destination}"],
        capture_output=True,
        text=True,
        check=True,
    )


def _prepare_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _prepare_solution_mounts_command(workdir: str) -> str:
    output_dir = _resolve_output_dir(workdir)
    quoted_output_dir = shlex.quote(output_dir)
    return (
        "rm -rf /solution && "
        "mkdir -p /solution && "
        f"mkdir -p {quoted_output_dir} && "
        f"if [ {quoted_output_dir} != /output ]; then rm -rf /output && ln -sfn {quoted_output_dir} /output; fi"
    )


def _resolve_output_dir(workdir: str) -> str:
    normalized = (workdir or "").strip() or "/app"
    if normalized == "/":
        return "/output"
    return f"{normalized.rstrip('/')}/output"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_to_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    if is_dataclass(record):
        return asdict(record)
    return dict(vars(record))


if __name__ == "__main__":
    raise SystemExit(main())
