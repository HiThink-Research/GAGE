from __future__ import annotations

import json
from pathlib import Path
import subprocess
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from gage_eval.config.loader import load_pipeline_config_payload
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.config.registry import ConfigRegistry
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.resource_profile import NodeResource, ResourceProfile


PYTHON = Path("/Users/panke/miniconda3/envs/new-gage/bin/python")
TASK_PATH = Path("/Users/panke/.cache/harbor/tasks/CGqXnFmcaVcQCTbCZMpg2T/gpt2-codegolf")
DOCKER_IMAGE = "alexgshaw/gpt2-codegolf:20251031"
LMSTUDIO_MODELS_URL = "http://127.0.0.1:1234/v1/models"
FIXTURE = Path("config/custom/external_harness_kits/harbor_terminal_bench2_lmstudio_1case.yaml")


@pytest.mark.live
@pytest.mark.external_harness
def test_terminal_bench_1case_live_lmstudio_harbor_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_if_preflight_missing()
    run_id = "task16-live-terminal-bench-1case"
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    config = PipelineConfig.from_dict(load_pipeline_config_payload(FIXTURE))

    runtime = build_runtime(
        config,
        ConfigRegistry(),
        _resource_profile(),
        trace=ObservabilityTrace(run_id=run_id),
    )

    runtime.run()

    run_dir = tmp_path / run_id
    raw_root = run_dir / "external_harness"
    manifest = json.loads((raw_root / "manifest.json").read_text(encoding="utf-8"))
    raw_entry = manifest["entries"][0]
    provider_root = raw_root / "tb2_one_case" / "harbor_tb2"
    launcher_result_path = provider_root / "launcher_result.json"
    launcher_result = json.loads(launcher_result_path.read_text(encoding="utf-8"))
    job_dir = Path(launcher_result["job_dir"])
    trial_results = sorted(job_dir.glob("*/result.json"))
    samples = _read_jsonl(run_dir / "samples.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    sample = samples[0]["sample"]
    trajectory = sample["prediction"]["trajectory"]
    verifier = sample["evaluation"]["raw_verifier_result"]
    harbor_summary = summary["external_harness"]["harbor"]

    assert manifest["schema_version"] == "gage.external_harness.raw_archive.v1"
    assert raw_entry["artifacts"]["workdir_ref"] == "tb2_one_case/harbor_tb2"
    assert raw_entry["artifacts"]["jobs_dir_ref"] == "tb2_one_case/harbor_tb2/jobs"
    assert raw_entry["artifacts"]["launcher_result_ref"] == "tb2_one_case/harbor_tb2/launcher_result.json"
    assert raw_entry["artifacts"]["job_config_ref"] == "tb2_one_case/harbor_tb2/job_config.json"
    assert (provider_root / "invocation.json").exists()
    assert (provider_root / "job_config.json").exists()
    assert launcher_result_path.exists()
    assert launcher_result["schema_version"] == "gage.harbor_launcher_result.v1"
    assert launcher_result["exit_code"] == 0
    assert trial_results
    assert len(samples) == 1
    assert any(_is_tool_call(event) for event in trajectory)
    assert verifier.get("rewards")
    assert "harbor_resolve_rate" in harbor_summary
    assert "harbor_score_mean" in harbor_summary


def _skip_if_preflight_missing() -> None:
    checks = (
        _check_python,
        _check_harbor_import,
        _check_docker_image,
        _check_docker_ps,
        _check_lmstudio,
        _check_task_path,
    )
    for check in checks:
        reason = check()
        if reason:
            pytest.skip(reason)


def _check_python() -> str | None:
    if not PYTHON.exists():
        return f"new-gage Python is missing: {PYTHON}"
    result = _run([str(PYTHON), "--version"])
    return None if result.returncode == 0 else f"new-gage Python failed: {result.stderr or result.stdout}"


def _check_harbor_import() -> str | None:
    result = _run(
        [
            str(PYTHON),
            "-c",
            "from harbor.job import Job; from harbor.models.job.config import JobConfig; import harbor; print(harbor.__file__)",
        ]
    )
    return None if result.returncode == 0 else f"Harbor import failed: {result.stderr or result.stdout}"


def _check_docker_image() -> str | None:
    result = _run(["docker", "image", "inspect", DOCKER_IMAGE])
    return None if result.returncode == 0 else f"Docker image missing or unavailable: {DOCKER_IMAGE}"


def _check_docker_ps() -> str | None:
    result = _run(["docker", "ps"])
    return None if result.returncode == 0 else f"Docker daemon unavailable: {result.stderr or result.stdout}"


def _check_lmstudio() -> str | None:
    try:
        with urlopen(LMSTUDIO_MODELS_URL, timeout=5) as response:
            if response.status >= 400:
                return f"LM Studio models endpoint returned HTTP {response.status}"
            response.read(1)
    except (OSError, URLError) as exc:
        return f"LM Studio models endpoint unreachable: {LMSTUDIO_MODELS_URL} ({exc})"
    return None


def _check_task_path() -> str | None:
    return None if TASK_PATH.is_dir() else f"Terminal-Bench task path missing: {TASK_PATH}"


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(argv, returncode=127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            argv,
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or "timeout"),
        )


def _resource_profile() -> ResourceProfile:
    return ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)])


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _is_tool_call(event: object) -> bool:
    if not isinstance(event, dict):
        return False
    if event.get("type") == "tool_call":
        return True
    return any(key in event for key in ("tool_name", "arguments", "output"))
