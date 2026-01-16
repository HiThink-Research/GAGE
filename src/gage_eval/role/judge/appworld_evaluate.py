"""AppWorld judge implementation using container-side CLI evaluation."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.judge.base import JudgeImplementation

_TEST_SUBSETS = {"test_normal", "test_challenge"}


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


@registry.asset(
    "judge_impls",
    "appworld_evaluate",
    desc="AppWorld CLI evaluation via container execution",
    tags=("appworld", "judge"),
)
class AppWorldEvaluate(JudgeImplementation):
    def __init__(
        self,
        *,
        docker_bin: str = "docker",
        appworld_root: str = "/run",
        experiment_name: str = "default",
        eval_timeout_s: int = 120,
        output_format: Optional[str] = None,
        result_path_template: Optional[str] = None,
        output_dir_template: Optional[str] = None,
        export_outputs: bool = False,
        export_dir: Optional[str] = None,
        redact_test_details: bool = True,
        redact_trace: bool = True,
    ) -> None:
        self._docker_bin = docker_bin
        self._appworld_root = appworld_root
        self._experiment_name = experiment_name
        self._eval_timeout_s = max(1, int(eval_timeout_s))
        self._output_format = output_format
        self._result_path_template = (
            result_path_template
            or "{root}/experiments/outputs/{experiment_name}/evaluations/on_only_{task_id}.json"
        )
        self._output_dir_template = (
            output_dir_template
            or "{root}/experiments/outputs/{experiment_name}/tasks/{task_id}"
        )
        self._export_outputs = bool(export_outputs)
        self._export_dir = export_dir
        self._redact_test_details = bool(redact_test_details)
        self._redact_trace = bool(redact_trace)

    def invoke(self, payload: Dict[str, Any], state: Any = None) -> Dict[str, Any]:
        params = payload.get("params") or {}
        sample = payload.get("sample") or {}
        runtime_handle = payload.get("runtime_handle") or {}

        appworld_meta = _resolve_appworld_meta(sample)
        task_id = _resolve_task_id(sample, appworld_meta)
        subset = _resolve_subset(appworld_meta, params)
        experiment_name = _resolve_experiment_name(appworld_meta, params, self._experiment_name)
        appworld_root = str(params.get("appworld_root") or self._appworld_root)
        container = _resolve_container(runtime_handle, params)
        if not container:
            return {"appworld": _build_error_payload(task_id, subset, experiment_name, "missing_container")}

        command = _resolve_eval_command(
            task_id=task_id,
            experiment_name=experiment_name,
            appworld_root=appworld_root,
            output_format=params.get("output_format", self._output_format),
            eval_command=params.get("eval_command"),
            eval_args=params.get("eval_args"),
        )
        result = _run_container_command(
            docker_bin=str(params.get("docker_bin") or self._docker_bin),
            container=container,
            command=command,
            timeout_s=int(params.get("eval_timeout_s", self._eval_timeout_s)),
        )
        if result.returncode != 0:
            logger.warning("AppWorld evaluate failed: {}", result.stderr.strip())
            return {"appworld": _build_error_payload(task_id, subset, experiment_name, "evaluate_failed")}

        eval_payload = _extract_eval_payload(
            result.stdout,
            docker_bin=str(params.get("docker_bin") or self._docker_bin),
            container=container,
            result_path_template=params.get("result_path_template", self._result_path_template),
            appworld_root=appworld_root,
            experiment_name=experiment_name,
            task_id=task_id,
        )
        if eval_payload is None:
            return {"appworld": _build_error_payload(task_id, subset, experiment_name, "missing_eval_output")}

        normalized = _normalize_eval_payload(eval_payload, task_id=task_id)
        output_dir = _resolve_output_dir(
            appworld_root=appworld_root,
            experiment_name=experiment_name,
            task_id=task_id,
            output_dir_template=params.get("output_dir_template", self._output_dir_template),
        )
        export_dir = _resolve_export_dir(
            base_dir=params.get("export_dir", self._export_dir),
            payload=payload,
            task_id=task_id,
            experiment_name=experiment_name,
            default_enabled=self._export_outputs or bool(params.get("export_outputs")),
        )
        if output_dir and export_dir and container:
            _export_container_dir(
                docker_bin=str(params.get("docker_bin") or self._docker_bin),
                container=container,
                source_dir=output_dir,
                export_dir=export_dir,
                task_id=task_id,
            )

        appworld_output = {
            "task_id": task_id,
            "subset": subset,
            "experiment_name": experiment_name,
            **normalized,
        }
        if export_dir:
            appworld_output["export_dir"] = export_dir
        if output_dir:
            appworld_output["output_dir"] = output_dir
        if subset in _TEST_SUBSETS and self._redact_test_details:
            appworld_output = _redact_test_output(appworld_output)
        if subset in _TEST_SUBSETS and self._redact_trace:
            _redact_agent_trace(sample)
        return {"appworld": appworld_output}


def _resolve_appworld_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    appworld_meta = metadata.get("appworld") if isinstance(metadata.get("appworld"), dict) else {}
    return dict(appworld_meta)


def _resolve_task_id(sample: Dict[str, Any], meta: Dict[str, Any]) -> str:
    task_id = meta.get("task_id") or sample.get("task_id") or sample.get("id")
    if not task_id:
        raise ValueError("AppWorld judge requires appworld.task_id")
    return str(task_id)


def _resolve_subset(meta: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    subset = meta.get("subset") or params.get("subset")
    if subset:
        return str(subset).strip().lower()
    return None


def _resolve_experiment_name(meta: Dict[str, Any], params: Dict[str, Any], default: str) -> str:
    name = meta.get("experiment_name") or params.get("experiment_name") or default
    return str(name or "default")


def _resolve_container(runtime_handle: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    for key in ("container_name", "container_id"):
        value = params.get(key) or runtime_handle.get(key)
        if value:
            return str(value)
    return None


def _resolve_eval_command(
    *,
    task_id: str,
    experiment_name: str,
    appworld_root: str,
    output_format: Optional[str],
    eval_command: Any,
    eval_args: Any,
) -> str:
    format_vars = {
        "task_id": task_id,
        "experiment_name": experiment_name,
        "root": appworld_root,
    }
    if eval_command:
        if isinstance(eval_command, str):
            return eval_command.format(**format_vars)
        if isinstance(eval_command, (list, tuple)):
            parts = [str(part).format(**format_vars) for part in eval_command]
            return shlex.join(parts)
    parts = [
        "appworld",
        "evaluate",
        experiment_name,
        "--task-id",
        task_id,
        "--root",
        appworld_root,
    ]
    if output_format:
        parts.extend(["--output-format", str(output_format)])
    if isinstance(eval_args, (list, tuple)):
        parts.extend([str(item) for item in eval_args])
    return shlex.join(parts)


def _run_container_command(*, docker_bin: str, container: str, command: str, timeout_s: int) -> CommandResult:
    args = [docker_bin, "exec", container, "/bin/sh", "-lc", command]
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"docker binary not found: {docker_bin}") from exc
    return CommandResult(stdout=completed.stdout, stderr=completed.stderr, returncode=completed.returncode)


def _extract_eval_payload(
    stdout: str,
    *,
    docker_bin: str,
    container: str,
    result_path_template: Optional[str],
    appworld_root: str,
    experiment_name: str,
    task_id: str,
) -> Optional[Dict[str, Any]]:
    payload = _parse_json_payload(stdout)
    if payload is not None:
        return payload
    if result_path_template:
        result_path = result_path_template.format(
            root=appworld_root,
            experiment_name=experiment_name,
            task_id=task_id,
        )
        file_payload = _read_container_json(
            docker_bin=docker_bin,
            container=container,
            path=result_path,
        )
        if file_payload is not None:
            return file_payload
    return None


def _parse_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = stripped[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _read_container_json(*, docker_bin: str, container: str, path: str) -> Optional[Dict[str, Any]]:
    command = f"cat {shlex.quote(path)}"
    result = _run_container_command(docker_bin=docker_bin, container=container, command=command, timeout_s=30)
    if result.returncode != 0:
        return None
    return _parse_json_payload(result.stdout)


def _normalize_eval_payload(payload: Dict[str, Any], *, task_id: Optional[str]) -> Dict[str, Any]:
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    individual = payload.get("individual") if isinstance(payload.get("individual"), dict) else {}
    task_payload: Dict[str, Any] = {}
    if task_id and task_id in individual and isinstance(individual[task_id], dict):
        task_payload = individual[task_id]

    tests = _resolve_tests_payload(task_payload) or _resolve_tests_payload(payload)
    normalized: Dict[str, Any] = {}
    tgc = _coerce_float(_pick(payload, ("tgc", "task_goal_completion", "task_completion")))
    if tgc is None:
        tgc = _coerce_float(_pick(aggregate, ("tgc", "task_goal_completion", "task_completion")))
    if tgc is not None:
        normalized["tgc"] = tgc
    sgc = _coerce_float(_pick(payload, ("sgc", "scenario_goal_completion")))
    if sgc is None:
        sgc = _coerce_float(_pick(aggregate, ("sgc", "scenario_goal_completion")))
    if sgc is not None:
        normalized["sgc"] = sgc
    if tests:
        normalized["tests"] = tests
    difficulty = task_payload.get("difficulty")
    if difficulty is None:
        difficulty = payload.get("difficulty")
    if difficulty is not None:
        normalized["difficulty"] = difficulty
    return normalized


def _resolve_tests_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tests = payload.get("tests")
    if isinstance(tests, dict):
        return dict(tests)
    passes = payload.get("passes")
    fails = payload.get("fails")
    if fails is None:
        fails = payload.get("failures")
    if passes is None and fails is None:
        return None
    result: Dict[str, Any] = {}
    if passes is not None:
        result["passes"] = passes
    if fails is not None:
        result["fails"] = fails
    return result or None


def _pick(payload: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _redact_test_output(appworld_output: Dict[str, Any]) -> Dict[str, Any]:
    tgc = appworld_output.get("tgc")
    minimal = {"tgc": tgc} if tgc is not None else {}
    for key in ("task_id", "subset", "experiment_name", "status", "output_dir", "export_dir"):
        if key in appworld_output:
            minimal[key] = appworld_output[key]
    return minimal


def _redact_agent_trace(sample: Dict[str, Any]) -> None:
    predict_result = sample.get("predict_result")
    if not isinstance(predict_result, list):
        return
    for entry in predict_result:
        if not isinstance(entry, dict):
            continue
        trace = entry.get("agent_trace")
        if not isinstance(trace, list):
            continue
        sanitized = []
        for step in trace:
            if not isinstance(step, dict):
                continue
            sanitized.append(_sanitize_trace_step(step))
        entry["agent_trace"] = sanitized


def _sanitize_trace_step(step: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = {
        "step_index",
        "role",
        "name",
        "status",
        "latency_ms",
        "turn_index",
    }
    sanitized = {key: step.get(key) for key in keep_keys if key in step}
    if "usage" in step:
        sanitized["usage"] = step.get("usage")
    sanitized["redacted"] = True
    return sanitized


def _resolve_output_dir(
    *,
    appworld_root: str,
    experiment_name: str,
    task_id: str,
    output_dir_template: Optional[str],
) -> Optional[str]:
    template = output_dir_template or "{root}/experiments/outputs/{experiment_name}/tasks/{task_id}"
    try:
        return template.format(root=appworld_root, experiment_name=experiment_name, task_id=task_id)
    except KeyError:
        return None


def _resolve_export_dir(
    *,
    base_dir: Optional[str],
    payload: Dict[str, Any],
    task_id: str,
    experiment_name: str,
    default_enabled: bool,
) -> Optional[str]:
    if base_dir:
        run_id = _resolve_run_id(payload)
        return _format_path_template(
            base_dir,
            run_id=run_id,
            task_id=task_id,
            experiment_name=experiment_name,
        )
    if not default_enabled:
        return None
    run_id = _resolve_run_id(payload)
    base = Path(os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs")) / run_id / "appworld_artifacts"
    return str(base)


def _resolve_run_id(payload: Dict[str, Any]) -> str:
    trace = payload.get("trace")
    run_id = getattr(trace, "run_id", None)
    if run_id:
        return str(run_id)
    return "run-unknown"


def _format_path_template(template: str, **kwargs: str) -> str:
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def _export_container_dir(
    *,
    docker_bin: str,
    container: str,
    source_dir: str,
    export_dir: str,
    task_id: str,
) -> None:
    target_dir = Path(export_dir) / task_id
    target_dir.mkdir(parents=True, exist_ok=True)
    source = f"{container}:{source_dir}/."
    args = [docker_bin, "cp", source, str(target_dir)]
    try:
        subprocess.run(args, capture_output=True, text=True, check=False)
    except Exception as exc:
        logger.warning("AppWorld export failed: {}", exc)


def _build_error_payload(
    task_id: str,
    subset: Optional[str],
    experiment_name: str,
    reason: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "experiment_name": experiment_name,
        "status": "error",
        "failure_reason": reason,
    }
    if subset:
        payload["subset"] = subset
    return payload


__all__ = ["AppWorldEvaluate"]
