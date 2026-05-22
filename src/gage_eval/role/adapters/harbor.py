"""Harbor external harness adapter translation layer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Literal

from gage_eval.external_harness_kits.errors import ExternalHarnessError
from gage_eval.external_harness_kits.harbor import launcher as harbor_launcher
from gage_eval.external_harness_kits.harbor.environment import build_harbor_environment_binding
from gage_eval.external_harness_kits.harbor.trace_translation import HarborATIFTranslator
from gage_eval.external_harness_kits.secret_redaction import (
    SecretRedactionContext,
    to_invocation_artifact,
)
from gage_eval.reporting.privacy import SecretFilter
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.external_harness_kits.base import (
    TaskBatchHarnessHandle,
    TaskBatchHarnessPlan,
    TaskBatchHarnessRequest,
    TaskBatchHarnessResult,
)

Probe = Callable[..., bool]

_LM_STUDIO_MODEL_PREFIX = "lm_studio/"
_HOSTED_VLLM_MODEL_PREFIX = "hosted_vllm/"
_HOSTED_VLLM_MODEL_INFO_KEYS = {
    "max_input_tokens",
    "max_output_tokens",
    "input_cost_per_token",
    "output_cost_per_token",
}
_SECRET_VALUE_PREFIXES = ("sk-", "ak-", "pk-")
_ENV_REF_RE = re.compile(
    r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-?(?P<default>.*))?\}$"
)


@dataclass(frozen=True)
class HarborInvocation:
    job_name: str
    jobs_dir: Path
    job_config_path: Path
    job_config: dict[str, Any]
    launcher_mode: Literal["python_subprocess"]
    launcher_argv: list[str]
    environ: dict[str, str]
    workdir: Path
    expected_total_trials: int | None

    def to_artifact_dict(self) -> dict[str, Any]:
        return to_invocation_artifact(
            self,
            context=SecretRedactionContext.from_environ(self.environ),
        )


class HarborAdapter(RoleAdapter):
    """Translate GAGE task-batch specs into Harbor JobConfig payloads."""

    _trace_translator = HarborATIFTranslator()

    def __init__(
        self,
        adapter_id: str,
        *,
        role_type: str = "external_harness",
        capabilities: tuple[str, ...] = ("task_batch_harness",),
        backend_id: str | None = None,
        backend: Any | None = None,
        env_id: str | None = None,
        trial_policy: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        resource_requirement: Mapping[str, Any] | None = None,
        sandbox_config: Mapping[str, Any] | None = None,
        registry_probe: Probe | None = None,
        installed_client_probe: Probe | None = None,
        local_path_visible_probe: Probe | None = None,
        model_endpoint_probe: Probe | None = None,
        docker_probe: Probe | None = None,
        e2b_probe: Probe | None = None,
        artifact_pull_probe: Probe | None = None,
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type=role_type,
            capabilities=capabilities,
            resource_requirement=dict(resource_requirement or {}),
            sandbox_config=dict(sandbox_config or {}),
        )
        self.backend_id = backend_id
        self.backend = backend
        self.env_id = env_id
        self.trial_policy = dict(trial_policy or {})
        self.params = dict(params or {})
        self._registry_probe = registry_probe
        self._installed_client_probe = installed_client_probe
        self._local_path_visible_probe = local_path_visible_probe
        self._model_endpoint_probe = model_endpoint_probe
        self._docker_probe = docker_probe
        self._e2b_probe = e2b_probe
        self._artifact_pull_probe = artifact_pull_probe
        self._active_invocations: dict[str, HarborInvocation] = {}

    def translate(self, request: TaskBatchHarnessRequest) -> TaskBatchHarnessPlan:
        payload = dict(request.payload)
        role_adapter = _mapping(payload.get("role_adapter"))
        task = _mapping(payload.get("task"))
        backend = self._resolve_backend(payload, role_adapter)
        dataset = self._resolve_dataset(payload, task)
        harness = self._harness(role_adapter)
        trial_policy = _mapping(role_adapter.get("trial_policy")) or self.trial_policy

        model = self._validated_model(backend)
        api_key = _resolve_secret_value(_mapping(backend.get("config")).get("api_key"))
        environment_binding = self._environment_binding(payload, role_adapter, harness)
        n_attempts = _trial_attempts(trial_policy)
        n_concurrent = self._n_concurrent(task, harness)
        datasets, tasks, expected_tasks = self._dataset_configs(dataset, task)
        agent_config = self._agent_config(
            backend=backend,
            harness=harness,
            model=model,
        )

        workdir = _path(payload.get("workdir"), default=Path(".gage") / "external_harness")
        jobs_dir = _path(payload.get("jobs_dir"), default=workdir / "jobs")
        job_config_path = _path(payload.get("job_config_path"), default=workdir / "harbor_job.json")
        job_name = _safe_job_name(f"{payload.get('run_id') or 'run'}__{task.get('task_id') or 'task'}")
        job_config = {
            "job_name": job_name,
            "jobs_dir": str(jobs_dir),
            "n_attempts": n_attempts,
            "n_concurrent_trials": n_concurrent,
            "timeout_multiplier": _timeout_multiplier(trial_policy, harness),
            "retry": {"max_retries": _max_retries(task, harness)},
            "environment": environment_binding.harbor_environment,
            "agents": [agent_config],
            "datasets": datasets,
            "tasks": tasks,
        }
        _apply_optional_job_options(job_config, harness)
        self._raise_if_job_config_contains_secret(job_config, {api_key} if api_key else set())

        invocation = HarborInvocation(
            job_name=job_name,
            jobs_dir=jobs_dir,
            job_config_path=job_config_path,
            job_config=job_config,
            launcher_mode="python_subprocess",
            launcher_argv=[
                str(payload.get("python") or sys.executable),
                "-m",
                "gage_eval.external_harness_kits.harbor.launcher",
                "--config",
                str(job_config_path),
                "--result-file",
                str(workdir / "launcher_result.json"),
            ],
            environ=_invocation_environ(api_key, agent_env=_mapping(agent_config.get("env"))),
            workdir=workdir,
            expected_total_trials=_expected_total_trials(expected_tasks, n_attempts),
        )
        plan_payload = {
            "job_config": job_config,
            "invocation": invocation,
            "adapter_projection": {"n_concurrent": n_concurrent},
            "environment_preflight_notes": environment_binding.preflight_notes,
            "dry_run": {
                "job_config": job_config,
                "launcher_argv": list(invocation.launcher_argv),
                "invocation": invocation.to_artifact_dict(),
            },
        }
        return TaskBatchHarnessPlan(adapter_id=self.adapter_id, payload=plan_payload)

    def launch(self, plan: TaskBatchHarnessPlan) -> TaskBatchHarnessHandle:
        invocation = _invocation_from_plan(plan)
        invocation.workdir.mkdir(parents=True, exist_ok=True)
        invocation.jobs_dir.mkdir(parents=True, exist_ok=True)
        _stage_local_registry_mirrors(invocation)
        _write_launcher_input(invocation, adapter_id=self.adapter_id)
        _write_raw_input_artifacts(invocation)
        self._active_invocations[invocation.job_name] = invocation
        try:
            launcher_result = harbor_launcher.run_launcher_subprocess(
                config_path=invocation.job_config_path,
                result_file=invocation.launcher_argv[-1],
                timeout_s=_launcher_timeout_s(self.params),
                environ=invocation.environ,
                python=invocation.launcher_argv[0] if invocation.launcher_argv else None,
                workdir=invocation.workdir,
                live_log=_launcher_live_log(self.params),
                job_log_path=invocation.jobs_dir / invocation.job_name / harbor_launcher.JOB_LOG_FILENAME,
            )
        finally:
            self._active_invocations.pop(invocation.job_name, None)
        return TaskBatchHarnessHandle(
            adapter_id=self.adapter_id,
            payload={
                "invocation": invocation,
                "launcher_subprocess": _launcher_subprocess_payload(launcher_result),
            },
        )

    def poll_until_done(self, handle: TaskBatchHarnessHandle) -> TaskBatchHarnessResult:
        payload = dict(handle.payload or {})
        invocation = payload.get("invocation")
        if not isinstance(invocation, HarborInvocation):
            raise ExternalHarnessError(
                "external_harness.runtime.harbor_api_incompatible",
                "Harbor launch handle does not contain a HarborInvocation",
            )
        subprocess_payload = _mapping(payload.get("launcher_subprocess"))
        launcher_result_path = Path(
            str(subprocess_payload.get("result_file") or _launcher_result_path_from_invocation(invocation))
        )
        launcher_result = _read_json_if_exists(launcher_result_path)
        job_dir = Path(str(launcher_result.get("job_dir") or invocation.jobs_dir / invocation.job_name))
        return TaskBatchHarnessResult(
            adapter_id=handle.adapter_id,
            payload={
                "launcher_result": launcher_result,
                "launcher_result_path": str(launcher_result_path),
                "job_name": invocation.job_name,
                "jobs_dir": str(invocation.jobs_dir),
                "job_dir": str(job_dir),
            },
        )

    def parse_results(
        self,
        result: TaskBatchHarnessResult,
        *,
        context: Any = None,
        handle: Any = None,
    ) -> Iterable[Any]:
        from gage_eval.external_harness_kits.harbor.results import parse_harbor_results

        trial_policy = self.trial_policy or _mapping(result.payload.get("trial_policy"))
        bundle = parse_harbor_results(
            result,
            context=context,
            handle=handle,
            reward_key=str(
                _mapping(_mapping(self.params.get("harness")).get("result")).get("reward_key")
                or _mapping(result.payload.get("harness_result")).get("reward_key")
                or result.payload.get("reward_key")
                or "reward"
            ),
            aggregation=str(trial_policy.get("aggregation")) if trial_policy.get("aggregation") else None,
            expected_trials=_expected_parse_trials(result.payload, trial_policy),
            dut_id=self.adapter_id,
            trace_translator=self._trace_translator,
        )
        return bundle.samples

    def shutdown(self) -> None:
        for invocation in list(self._active_invocations.values()):
            _write_cancelled_marker(invocation, reason="adapter_shutdown")

    async def ainvoke(self, payload: dict[str, Any], state: RoleAdapterState) -> dict[str, Any]:
        raise NotImplementedError("HarborAdapter is not a sample-loop role adapter")

    def _initialize(self, plan: TaskBatchHarnessPlan | None = None) -> None:
        context = dict(plan.payload if plan else {})
        checks = (
            (
                "Harbor Python package import",
                "_preflight_harbor_import",
                "external_harness.runtime.harbor_unavailable",
            ),
            (
                "Harbor Job API constructible",
                "_preflight_harbor_job_api",
                "external_harness.runtime.harbor_api_incompatible",
            ),
            (
                "Harbor registry ref resolvable",
                "_preflight_registry_ref",
                "external_harness.runtime.registry_not_found",
            ),
            (
                "JobConfig write/read",
                "_preflight_job_config_io",
                "external_harness.environment.io_unusable",
            ),
            (
                "provider compatibility",
                "_preflight_provider_compatibility",
                "external_harness.environment.provider_mismatch",
            ),
            (
                "model endpoint reachable",
                "_preflight_model_endpoint",
                "external_harness.environment.model_endpoint_unreachable",
            ),
            (
                "Docker trial usable",
                "_preflight_docker_trial",
                "external_harness.environment.docker_unavailable",
            ),
            (
                "E2B trial usable",
                "_preflight_e2b_trial",
                "external_harness.environment.e2b_unavailable",
            ),
            (
                "jobs_dir archivable",
                "_preflight_jobs_dir_artifact_pull",
                "external_harness.environment.artifact_pull_failed",
            ),
        )
        if plan is None:
            checks = checks[:2]
        for label, method_name, code in checks:
            try:
                getattr(self, method_name)(context)
            except ExternalHarnessError as exc:
                if exc.code == code:
                    raise
                raise ExternalHarnessError(code, f"{label} preflight failed: {exc}") from exc
            except Exception as exc:
                raise ExternalHarnessError(code, f"{label} preflight failed: {exc}") from exc

    def initialize(self) -> None:
        self._initialize()

    def _preflight_harbor_import(self, context: dict[str, Any]) -> None:
        del context
        from harbor.job import Job  # noqa: F401
        from harbor.models.job.config import JobConfig  # noqa: F401

    def _preflight_harbor_job_api(self, context: dict[str, Any]) -> None:
        from harbor.job import Job
        from harbor.models.job.config import JobConfig

        create = getattr(Job, "create", None)
        if not callable(create) or not inspect.iscoroutinefunction(create):
            raise RuntimeError("Job.create is not callable")
        config_payload = {
            "job_name": "gage_preflight",
            "environment": {"type": "docker"},
            "agents": [{"name": "nop"}],
            "datasets": [],
            "tasks": [{"path": "/tmp/gage-preflight"}],
        }
        preflight_root = _preflight_root(context)
        if preflight_root is not None:
            config_payload["jobs_dir"] = str(preflight_root / "jobs")
        config = JobConfig.model_validate(config_payload)
        if preflight_root is None:
            return
        try:
            _run_async(Job.create(config))
        finally:
            shutil.rmtree(preflight_root, ignore_errors=True)

    def _preflight_registry_ref(self, context: dict[str, Any]) -> None:
        for dataset in context.get("job_config", {}).get("datasets", []):
            name = dataset.get("name")
            version = dataset.get("version") or dataset.get("ref")
            if not name or not version:
                continue
            if self._registry_probe:
                exists = self._registry_probe(name, version, dataset)
            else:
                exists = _registry_ref_exists(dataset)
            if not exists:
                raise ExternalHarnessError(
                    "external_harness.runtime.registry_not_found",
                    f"Harbor registry ref '{name}@{version}' was not found",
                )

    def _preflight_job_config_io(self, context: dict[str, Any]) -> None:
        invocation = context.get("invocation")
        if not isinstance(invocation, HarborInvocation):
            return
        invocation.job_config_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(invocation.job_config, sort_keys=True)
        invocation.job_config_path.write_text(serialized, encoding="utf-8")
        if invocation.job_config_path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("JobConfig write/read hash mismatch")

    def _preflight_provider_compatibility(self, context: dict[str, Any]) -> None:
        job_config = context.get("job_config")
        if not isinstance(job_config, Mapping):
            return
        environment = _mapping(job_config.get("environment"))
        if environment.get("type") not in {"docker", "e2b"}:
            raise ExternalHarnessError(
                "external_harness.environment.provider_mismatch",
                f"Harbor environment type '{environment.get('type')}' is unsupported",
            )

    def _preflight_model_endpoint(self, context: dict[str, Any]) -> None:
        agent = _first(context.get("job_config", {}).get("agents"))
        api_base = _mapping(_mapping(agent).get("kwargs")).get("api_base")
        if not api_base:
            return
        if self._model_endpoint_probe:
            reachable = self._model_endpoint_probe(api_base=api_base, agent=agent)
        else:
            reachable = _model_endpoint_reachable(str(api_base))
        if not reachable:
            raise ExternalHarnessError(
                "external_harness.environment.model_endpoint_unreachable",
                f"model endpoint '{api_base}' is unreachable from Harbor launcher",
            )

    def _preflight_docker_trial(self, context: dict[str, Any]) -> None:
        if context.get("job_config", {}).get("environment", {}).get("type") != "docker":
            return
        if self._docker_probe:
            usable = self._docker_probe()
        else:
            usable = _docker_info_available()
        if not usable:
            raise ExternalHarnessError(
                "external_harness.environment.docker_unavailable",
                "docker trial environment is unavailable from Harbor launcher host",
            )

    def _preflight_e2b_trial(self, context: dict[str, Any]) -> None:
        if context.get("job_config", {}).get("environment", {}).get("type") != "e2b":
            return
        if self._e2b_probe:
            usable = self._e2b_probe()
        else:
            environment = _mapping(context.get("job_config", {}).get("environment"))
            kwargs = _mapping(environment.get("kwargs"))
            usable = bool(kwargs.get("template_id") and os.environ.get("E2B_API_KEY"))
        if not usable:
            raise ExternalHarnessError(
                "external_harness.environment.e2b_unavailable",
                "E2B trial environment is unavailable",
            )

    def _preflight_jobs_dir_artifact_pull(self, context: dict[str, Any]) -> None:
        invocation = context.get("invocation")
        jobs_dir = invocation.jobs_dir if isinstance(invocation, HarborInvocation) else None
        if self._artifact_pull_probe:
            usable = self._artifact_pull_probe(jobs_dir=jobs_dir)
        else:
            usable = _jobs_dir_archivable(jobs_dir)
        if not usable:
            raise ExternalHarnessError(
                "external_harness.environment.artifact_pull_failed",
                "jobs_dir artifact sink is not readable/writable",
            )

    def _resolve_backend(self, payload: Mapping[str, Any], role_adapter: Mapping[str, Any]) -> Mapping[str, Any]:
        backend = payload.get("backend")
        if isinstance(backend, Mapping):
            return backend
        backend_id = role_adapter.get("backend_id") or self.backend_id
        for candidate in payload.get("backends") or ():
            if isinstance(candidate, Mapping) and candidate.get("backend_id") == backend_id:
                return candidate
        raise ExternalHarnessError(
            "external_harness.translate.backend_missing",
            f"backend_id '{backend_id}' is not available for Harbor translation",
        )

    def _resolve_dataset(self, payload: Mapping[str, Any], task: Mapping[str, Any]) -> Mapping[str, Any]:
        dataset = payload.get("dataset")
        if isinstance(dataset, Mapping):
            return dataset
        dataset_id = task.get("dataset_id")
        for candidate in payload.get("datasets") or ():
            if isinstance(candidate, Mapping) and candidate.get("dataset_id") == dataset_id:
                return candidate
        raise ExternalHarnessError(
            "external_harness.config.invalid_dataset_params",
            f"task dataset_id '{dataset_id}' is not available for Harbor translation",
        )

    def _environment_binding(
        self,
        payload: Mapping[str, Any],
        role_adapter: Mapping[str, Any],
        harness: Mapping[str, Any],
    ):
        environment = payload.get("environment")
        env_id = role_adapter.get("env_id") or self.env_id
        if not isinstance(environment, Mapping):
            for candidate in payload.get("environments") or ():
                if isinstance(candidate, Mapping) and candidate.get("env_id") == env_id:
                    environment = candidate
                    break
        if not isinstance(environment, Mapping):
            raise ExternalHarnessError(
                "external_harness.translate.environment_bridge_failed",
                f"env_id '{env_id}' is not available for Harbor environment translation",
            )
        try:
            environment_override = (
                _mapping(harness.get("environment_override"))
                or _mapping(role_adapter.get("environment_override"))
                or None
            )
            return build_harbor_environment_binding(
                environment,
                environment_override=environment_override,
                validation_phase="environment",
            )
        except ExternalHarnessError as exc:
            raise ExternalHarnessError(
                "external_harness.translate.environment_bridge_failed",
                f"env_id '{env_id}' could not be translated to Harbor environment: {exc}",
            ) from exc

    def _harness(self, role_adapter: Mapping[str, Any]) -> Mapping[str, Any]:
        params = _mapping(role_adapter.get("params")) or self.params
        return _mapping(params.get("harness"))

    def _n_concurrent(self, task: Mapping[str, Any], harness: Mapping[str, Any]) -> int:
        task_value = _optional_int(
            task.get("concurrency"),
            code="external_harness.config.invalid_concurrency",
            label="tasks[].concurrency",
        )
        harness_value = _optional_int(
            harness.get("n_concurrent"),
            code="external_harness.config.invalid_concurrency",
            label="harness.n_concurrent",
        )
        if (task_value is not None and task_value < 1) or (harness_value is not None and harness_value < 1):
            raise ExternalHarnessError(
                "external_harness.config.invalid_concurrency",
                "tasks[].concurrency and harness.n_concurrent must be >= 1",
            )
        if task_value is not None and harness_value is not None and task_value != harness_value:
            raise ExternalHarnessError(
                "external_harness.config.invalid_concurrency",
                "tasks[].concurrency and harness.n_concurrent must match",
            )
        return task_value or harness_value or 1

    def _validated_model(self, backend: Mapping[str, Any]) -> str:
        backend_config = _mapping(backend.get("config"))
        model = backend_config.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ExternalHarnessError(
                "external_harness.translate.invalid_model",
                "backend.config.model must be a non-empty LiteLLM model string",
            )
        model = model.strip()
        if model.startswith(_HOSTED_VLLM_MODEL_PREFIX):
            hosted_name = model.removeprefix(_HOSTED_VLLM_MODEL_PREFIX)
            if not hosted_name:
                raise ExternalHarnessError(
                    "external_harness.translate.invalid_model",
                    "hosted_vllm model must include a canonical model name",
                )
            if "/" in hosted_name:
                raise ExternalHarnessError(
                    "external_harness.translate.invalid_model",
                    "hosted_vllm model must use a canonical model name without nested slashes",
                )
            return model
        provider = backend_config.get("custom_llm_provider") or backend_config.get("provider")
        if provider == "lm_studio" and not model.startswith(_LM_STUDIO_MODEL_PREFIX):
            raise ExternalHarnessError(
                "external_harness.translate.invalid_model",
                "LM Studio real model id must be translated to the LiteLLM model string "
                f"'{_LM_STUDIO_MODEL_PREFIX}{model}' before Harbor translation",
            )
        if "/" not in model:
            raise ExternalHarnessError(
                "external_harness.translate.invalid_model",
                f"backend.config.model '{model}' must include a LiteLLM provider prefix",
            )
        return model

    def _dataset_configs(
        self,
        dataset: Mapping[str, Any],
        task: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None]:
        loader = dataset.get("loader") or dataset.get("type")
        if loader == "harbor_registry":
            config = self._registry_dataset_config(dataset, task)
            expected = config.get("n_tasks")
            expected_tasks = _optional_int(
                expected,
                code="external_harness.config.invalid_dataset_params",
                label="DatasetConfig.n_tasks",
            )
            return [config], [], expected_tasks
        if loader == "harbor_local_path":
            return self._local_path_configs(dataset, task)
        raise ExternalHarnessError(
            "external_harness.config.invalid_loader",
            f"HarborAdapter v1 requires harbor_registry or harbor_local_path dataset, got '{loader}'",
        )

    def _registry_dataset_config(self, dataset: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
        params = _mapping(dataset.get("params"))
        name, version_or_ref, uses_package_ref = _registry_name_version(params)
        config: dict[str, Any] = {"name": name}
        if uses_package_ref:
            config["ref"] = version_or_ref
        else:
            config["version"] = version_or_ref
        if params.get("registry_url") and params.get("registry_path"):
            raise ExternalHarnessError(
                "external_harness.config.invalid_dataset_params",
                "harbor_registry dataset must not set both registry_url and registry_path",
            )
        if params.get("registry_url") is not None:
            config["registry_url"] = params["registry_url"]
        if params.get("registry_path") is not None:
            config["registry_path"] = str(Path(str(params["registry_path"])).expanduser().resolve())
        for key in ("task_names", "exclude_task_names"):
            if params.get(key) is not None:
                config[key] = params[key]
        n_tasks = params.get("n_tasks") if params.get("n_tasks") is not None else task.get("max_samples")
        if n_tasks is not None:
            config["n_tasks"] = _positive_int_or_none(
                n_tasks,
                "external_harness.config.invalid_dataset_params",
                label="DatasetConfig.n_tasks",
            )
        if self._registry_probe and not self._registry_probe(name, version_or_ref, config):
            raise ExternalHarnessError(
                "external_harness.runtime.registry_not_found",
                f"Harbor registry ref '{name}@{version_or_ref}' was not found",
            )
        return config

    def _local_path_configs(
        self,
        dataset: Mapping[str, Any],
        task: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None]:
        params = _mapping(dataset.get("params"))
        raw_path = params.get("path")
        path = Path(str(raw_path)).expanduser() if raw_path else None
        if path is None or not path.exists():
            raise ExternalHarnessError(
                "external_harness.runtime.local_path_not_found",
                f"Harbor local task path '{raw_path}' does not exist",
            )
        path = path.resolve()
        if not self._is_local_path_visible(path):
            raise ExternalHarnessError(
                "external_harness.runtime.local_path_not_visible",
                f"Harbor local task path '{path}' is not visible to the launcher",
            )
        path_kind = str(params.get("path_kind") or "auto")
        is_task_path = _is_harbor_task_path(path)
        if path_kind == "task" and not is_task_path:
            raise ExternalHarnessError(
                "external_harness.config.invalid_dataset_params",
                f"path_kind=task does not match Harbor TaskPaths validation for '{path}'",
            )
        if path_kind == "dataset" and is_task_path:
            raise ExternalHarnessError(
                "external_harness.config.invalid_dataset_params",
                f"path_kind=dataset does not match single Harbor task path '{path}'",
            )
        resolved_kind = "task" if (path_kind == "task" or (path_kind == "auto" and is_task_path)) else "dataset"
        if resolved_kind == "task":
            max_samples = _optional_int(
                task.get("max_samples"),
                code="external_harness.config.invalid_dataset_params",
                label="tasks[].max_samples",
            )
            if max_samples is not None and max_samples > 1:
                raise ExternalHarnessError(
                    "external_harness.config.invalid_dataset_params",
                    "single Harbor task path requires max_samples <= 1",
                )
            if params.get("task_names") or params.get("exclude_task_names"):
                raise ExternalHarnessError(
                    "external_harness.config.invalid_dataset_params",
                    "single Harbor task path must not set task filters",
                )
            task_config = {"path": str(path)}
            return [], [task_config], 1
        dataset_config: dict[str, Any] = {"path": str(path)}
        for key in ("task_names", "exclude_task_names"):
            if params.get(key) is not None:
                dataset_config[key] = params[key]
        if task.get("max_samples") is not None:
            dataset_config["n_tasks"] = _positive_int_or_none(
                task.get("max_samples"),
                "external_harness.config.invalid_dataset_params",
                label="tasks[].max_samples",
            )
        return [dataset_config], [], dataset_config.get("n_tasks")

    def _is_local_path_visible(self, path: Path) -> bool:
        if self._local_path_visible_probe is not None:
            return self._local_path_visible_probe(path)
        return os.access(path, os.R_OK | os.X_OK)

    def _agent_config(
        self,
        *,
        backend: Mapping[str, Any],
        harness: Mapping[str, Any],
        model: str,
    ) -> dict[str, Any]:
        agent = _mapping(harness.get("agent"))
        kind = agent.get("kind") or "base_agent"
        name = agent.get("name")
        import_path = agent.get("import_path")
        self._validate_agent_name(name, import_path)
        if model.startswith(_HOSTED_VLLM_MODEL_PREFIX):
            _require_hosted_vllm_model_info(_mapping(backend.get("config")))
        if kind == "base_agent":
            return self._base_agent_config(backend=backend, agent=agent, model=model)
        if kind == "installed_client":
            return self._installed_client_config(backend=backend, agent=agent, model=model)
        raise ExternalHarnessError(
            "external_harness.translate.invalid_agent",
            f"harness.agent.kind '{kind}' is unsupported for Harbor v1",
        )

    def _validate_agent_name(self, name: Any, import_path: Any) -> None:
        if bool(name) == bool(import_path):
            raise ExternalHarnessError(
                "external_harness.translate.invalid_agent",
                "Harbor agent must provide exactly one of name or import_path",
            )
        if not name:
            return
        try:
            from harbor.models.agent.name import AgentName
        except Exception as exc:
            raise ExternalHarnessError(
                "external_harness.runtime.harbor_unavailable",
                f"Harbor AgentName cannot be imported: {exc}",
            ) from exc
        if str(name) not in AgentName.values():
            raise ExternalHarnessError(
                "external_harness.translate.invalid_agent",
                f"Harbor agent name '{name}' is not in AgentName.values()",
            )

    def _base_agent_config(
        self,
        *,
        backend: Mapping[str, Any],
        agent: Mapping[str, Any],
        model: str,
    ) -> dict[str, Any]:
        backend_config = _mapping(backend.get("config"))
        api_base = backend_config.get("api_base")
        if not isinstance(api_base, str) or not api_base:
            raise ExternalHarnessError(
                "external_harness.translate.backend_agent_bridge_failed",
                "base_agent requires backend.config.api_base for Harbor launcher-side model calls",
            )
        kwargs = dict(_mapping(agent.get("kwargs")))
        if "api_base" in kwargs and kwargs["api_base"] != api_base:
            raise ExternalHarnessError(
                "external_harness.translate.backend_agent_bridge_failed",
                "base_agent kwargs.api_base conflicts with backend.config.api_base",
            )
        kwargs["api_base"] = api_base
        provider = backend_config.get("custom_llm_provider") or backend_config.get("provider")
        if provider:
            kwargs = _inject_base_agent_provider(kwargs, provider)
        kwargs = _merge_generation_parameters(kwargs, _mapping(backend_config.get("generation_parameters")))
        kwargs["model_info"] = _merged_model_info(
            _mapping_or_error(
                backend_config.get("model_info"),
                code="external_harness.translate.backend_agent_bridge_failed",
                label="backend.config.model_info",
            ),
            _mapping(kwargs.pop("model_info", None)),
        )
        config = _agent_identity(agent)
        config["model_name"] = model
        config["kwargs"] = _strip_empty(kwargs)
        extra_env = _non_secret_extra_env(agent.get("extra_env"))
        if extra_env:
            config["env"] = extra_env
        return config

    def _installed_client_config(
        self,
        *,
        backend: Mapping[str, Any],
        agent: Mapping[str, Any],
        model: str,
    ) -> dict[str, Any]:
        extra_env = _non_secret_extra_env(agent.get("extra_env"))
        if not any(key in extra_env for key in ("OPENAI_BASE_URL", "LLM_BASE_URL", "ANTHROPIC_BASE_URL")):
            raise ExternalHarnessError(
                "external_harness.translate.installed_client_incompatible",
                "installed_client requires trial-visible endpoint env configuration",
            )
        if self._installed_client_probe is not None:
            probe_ok = self._installed_client_probe(
                backend=backend,
                agent=agent,
                env=extra_env,
            )
        else:
            probe_ok = _installed_client_agent_available(agent)
        if not probe_ok:
            raise ExternalHarnessError(
                "external_harness.translate.installed_client_incompatible",
                "installed_client CLI/env preflight did not pass",
            )
        config = _agent_identity(agent)
        config["model_name"] = model
        kwargs = dict(_mapping(agent.get("kwargs")))
        kwargs.pop("api_base", None)
        if kwargs:
            config["kwargs"] = kwargs
        config["env"] = extra_env
        return config

    def _raise_if_job_config_contains_secret(self, job_config: Mapping[str, Any], secret_values: set[str]) -> None:
        if _contains_secret(job_config, secret_values):
            raise ExternalHarnessError(
                "external_harness.translate.secret_serialization_blocked",
                "Harbor JobConfig contains a secret-like key or value",
            )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return None


def _path(value: Any, *, default: Path) -> Path:
    if value is None:
        return default.expanduser().resolve()
    return Path(str(value)).expanduser().resolve()


def _safe_job_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe.replace("-", "_").strip("_") or "gage_harbor_job"


def _optional_int(value: Any, *, code: str, label: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, code=code, label=label)


def _positive_int_or_none(value: Any, code: str, *, label: str = "value") -> int:
    number = _coerce_int(value, code=code, label=label)
    if number < 1:
        raise ExternalHarnessError(code, f"{label} must be >= 1")
    return number


def _coerce_int(value: Any, *, code: str, label: str) -> int:
    if isinstance(value, bool):
        raise ExternalHarnessError(code, f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        raise ExternalHarnessError(code, f"{label} must be an integer")
    raise ExternalHarnessError(code, f"{label} must be an integer")


def _coerce_float(value: Any, *, code: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExternalHarnessError(code, f"{label} must be a number") from exc
    if not math.isfinite(number):
        raise ExternalHarnessError(code, f"{label} must be finite")
    return number


def _trial_attempts(trial_policy: Mapping[str, Any]) -> int:
    trials = _optional_int(
        trial_policy.get("trials"),
        code="external_harness.translate.invalid_trials",
        label="trial_policy.trials",
    )
    n_attempts = _optional_int(
        trial_policy.get("n_attempts"),
        code="external_harness.translate.invalid_trials",
        label="trial_policy.n_attempts",
    )
    if trials is not None and n_attempts is not None and trials != n_attempts:
        raise ExternalHarnessError(
            "external_harness.translate.invalid_trials",
            "trial_policy.trials and trial_policy.n_attempts must match when both are set",
        )
    value = trials if trials is not None else n_attempts
    attempts = value if value is not None else 1
    if attempts < 1:
        raise ExternalHarnessError(
            "external_harness.translate.invalid_trials",
            "trial_policy.trials must be >= 1",
        )
    return attempts


def _timeout_multiplier(trial_policy: Mapping[str, Any], harness: Mapping[str, Any]) -> float:
    job_options = _mapping(harness.get("job_options"))
    value = trial_policy.get("timeout_multiplier", job_options.get("timeout_multiplier", 1.0))
    return _coerce_float(
        value,
        code="external_harness.config.invalid_dataset_params",
        label="timeout_multiplier",
    )


def _max_retries(task: Mapping[str, Any], harness: Mapping[str, Any]) -> int:
    failure_policy = _mapping(task.get("failure_policy"))
    job_options = _mapping(harness.get("job_options"))
    value = failure_policy.get("max_retries", job_options.get("max_retries", 0))
    retries = _coerce_int(
        value,
        code="external_harness.config.invalid_dataset_params",
        label="max_retries",
    )
    if retries < 0:
        raise ExternalHarnessError(
            "external_harness.config.invalid_dataset_params",
            "max_retries must be >= 0",
        )
    return retries


def _apply_optional_job_options(job_config: dict[str, Any], harness: Mapping[str, Any]) -> None:
    job_options = _mapping(harness.get("job_options"))
    for source, target in (
        ("agent_timeout_multiplier", "agent_timeout_multiplier"),
        ("verifier_timeout_multiplier", "verifier_timeout_multiplier"),
        ("agent_setup_timeout_multiplier", "agent_setup_timeout_multiplier"),
        ("environment_build_timeout_multiplier", "environment_build_timeout_multiplier"),
    ):
        if job_options.get(source) is not None:
            job_config[target] = _coerce_float(
                job_options[source],
                code="external_harness.config.invalid_dataset_params",
                label=source,
            )


def _registry_name_version(params: Mapping[str, Any]) -> tuple[str, str, bool]:
    ref = params.get("ref")
    if ref is not None:
        text = str(ref)
        if "@" not in text:
            raise ExternalHarnessError(
                "external_harness.translate.invalid_ref",
                f"Harbor registry ref '{text}' must be name@version or package/name@ref",
            )
        name, version = text.rsplit("@", 1)
        if not name or not version:
            raise ExternalHarnessError(
                "external_harness.translate.invalid_ref",
                f"Harbor registry ref '{text}' must include both name and version",
            )
        return name, version, "/" in name
    name = params.get("name")
    version = params.get("version") or params.get("ref")
    if not name or not version:
        raise ExternalHarnessError(
            "external_harness.translate.invalid_ref",
            "harbor_registry dataset must provide params.ref or params.name with params.version/ref",
        )
    return str(name), str(version), "/" in str(name)


def _is_harbor_task_path(path: Path) -> bool:
    try:
        from harbor.models.task.paths import TaskPaths

        return bool(TaskPaths(path).is_valid())
    except Exception:
        return False


def _registry_ref_exists(dataset: Mapping[str, Any]) -> bool:
    try:
        from harbor.models.job.config import DatasetConfig

        config = DatasetConfig.model_validate(dataset)
        _run_async(config.get_task_configs(disable_verification=True))
        return True
    except Exception:
        return False


def _model_endpoint_reachable(api_base: str) -> bool:
    try:
        from urllib import request

        url = f"{api_base.rstrip('/')}/models"
        with request.urlopen(url, timeout=2.0) as response:
            return 200 <= int(response.status) < 500
    except Exception:
        return False


def _docker_info_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _jobs_dir_archivable(jobs_dir: Path | None) -> bool:
    if jobs_dir is None:
        return False
    try:
        jobs_dir.mkdir(parents=True, exist_ok=True)
        probe_path = jobs_dir / ".gage_artifact_probe"
        probe_path.write_text("ok", encoding="utf-8")
        if probe_path.read_text(encoding="utf-8") != "ok":
            return False
        probe_path.unlink(missing_ok=True)
        return os.access(jobs_dir, os.R_OK | os.W_OK | os.X_OK)
    except Exception:
        return False


def _preflight_root(context: Mapping[str, Any]) -> Path | None:
    invocation = context.get("invocation")
    if isinstance(invocation, HarborInvocation):
        return invocation.workdir / "_preflight"
    return None


def _run_async(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("Harbor preflight cannot run inside an active event loop")


def _resolve_secret_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    match = _ENV_REF_RE.match(text)
    if match:
        env_name = match.group("name")
        default = match.group("default")
        resolved = os.environ.get(env_name, default)
        if resolved is None or "${" in str(resolved):
            raise ExternalHarnessError(
                "external_harness.translate.unresolved_secret",
                f"secret reference '{text}' is unresolved",
            )
        return str(resolved)
    if "${" in text:
        raise ExternalHarnessError(
            "external_harness.translate.unresolved_secret",
            f"secret reference '{text}' is unresolved",
        )
    return text


def _invocation_environ(api_key: str | None, *, agent_env: Mapping[str, Any] | None = None) -> dict[str, str]:
    environ: dict[str, str] = {}
    pythonpath = _launcher_pythonpath()
    if pythonpath:
        environ["PYTHONPATH"] = pythonpath
    for key, value in (agent_env or {}).items():
        env_key = str(key)
        env_value = str(value)
        if _looks_secret_key(env_key) or _looks_secret_value(env_value):
            continue
        environ[env_key] = env_value
    if api_key is not None:
        environ["OPENAI_API_KEY"] = api_key
    return environ


def _launcher_pythonpath() -> str:
    repo_src = Path(__file__).resolve().parents[3]
    entries = [str(repo_src)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        entries.extend(part for part in existing.split(os.pathsep) if part)
    deduped = list(dict.fromkeys(entries))
    return os.pathsep.join(deduped)


def _merge_generation_parameters(
    kwargs: dict[str, Any],
    generation_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    if not generation_parameters:
        return kwargs
    llm_call_kwargs = dict(_mapping(kwargs.get("llm_call_kwargs")))
    for key, value in generation_parameters.items():
        if key in {"max_new_tokens", "max_tokens"}:
            llm_call_kwargs["max_tokens"] = value
        elif key in {"temperature", "top_p", "stop", "seed"}:
            kwargs[key] = value
        else:
            llm_call_kwargs[key] = value
    if llm_call_kwargs:
        kwargs["llm_call_kwargs"] = llm_call_kwargs
    return kwargs


def _inject_base_agent_provider(kwargs: dict[str, Any], provider: Any) -> dict[str, Any]:
    """Bridge provider naming through both Harbor BaseAgent and its LiteLLM call kwargs.

    Harbor's BaseAgent consumes the top-level value, while Terminus-2 forwards
    ``llm_kwargs`` into LiteLLM. Both layers need the same provider to avoid
    local OpenAI-compatible model names being interpreted as unknown providers.
    """

    if "custom_llm_provider" in kwargs and kwargs["custom_llm_provider"] != provider:
        raise ExternalHarnessError(
            "external_harness.translate.backend_agent_bridge_failed",
            "base_agent kwargs.custom_llm_provider conflicts with backend.config provider",
        )
    kwargs["custom_llm_provider"] = provider

    llm_kwargs = dict(
        _mapping_or_error(
            kwargs.get("llm_kwargs"),
            code="external_harness.translate.backend_agent_bridge_failed",
            label="base_agent kwargs.llm_kwargs",
        )
    )
    if "custom_llm_provider" in llm_kwargs and llm_kwargs["custom_llm_provider"] != provider:
        raise ExternalHarnessError(
            "external_harness.translate.backend_agent_bridge_failed",
            "base_agent kwargs.llm_kwargs.custom_llm_provider conflicts with backend.config provider",
        )
    llm_kwargs["custom_llm_provider"] = provider
    kwargs["llm_kwargs"] = llm_kwargs
    return kwargs


def _mapping_or_error(value: Any, *, code: str, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise ExternalHarnessError(code, f"{label} must be a mapping")


def _merged_model_info(
    backend_model_info: Mapping[str, Any],
    agent_model_info: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(backend_model_info)
    for key, value in agent_model_info.items():
        if key in result:
            existing = result[key]
            if isinstance(existing, Mapping) and isinstance(value, Mapping):
                result[key] = _merged_model_info(existing, value)
                continue
            if existing != value:
                raise ExternalHarnessError(
                    "external_harness.translate.model_info_conflict",
                    f"model_info key '{key}' differs between backend and agent kwargs",
                )
        else:
            result[key] = value
    return result


def _require_hosted_vllm_model_info(backend_config: Mapping[str, Any]) -> None:
    model_info = backend_config.get("model_info")
    if not isinstance(model_info, Mapping):
        missing = sorted(_HOSTED_VLLM_MODEL_INFO_KEYS)
    else:
        missing = sorted(_HOSTED_VLLM_MODEL_INFO_KEYS - set(model_info))
    if missing:
        raise ExternalHarnessError(
            "external_harness.translate.model_info_required",
            f"hosted_vllm backend requires model_info keys {missing}",
        )


def _agent_identity(agent: Mapping[str, Any]) -> dict[str, Any]:
    if agent.get("name"):
        return {"name": str(agent["name"])}
    return {"import_path": str(agent["import_path"])}


def _installed_client_agent_available(agent: Mapping[str, Any]) -> bool:
    try:
        from harbor.agents.factory import AgentFactory
        from harbor.agents.installed.base import BaseInstalledAgent
    except Exception:
        return False

    try:
        if agent.get("name"):
            from harbor.models.agent.name import AgentName

            agent_class = AgentFactory._AGENT_MAP.get(AgentName(str(agent["name"])))  # noqa: SLF001
        else:
            module_path, class_name = str(agent["import_path"]).split(":", 1)
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
    except Exception:
        return False
    return inspect.isclass(agent_class) and issubclass(agent_class, BaseInstalledAgent)


def _non_secret_extra_env(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExternalHarnessError(
            "external_harness.config.secret_agent_env_forbidden",
            "harness.agent.extra_env must be a mapping of non-secret environment values",
        )
    env: dict[str, str] = {}
    for key, raw_value in value.items():
        env_key = str(key)
        env_value = str(raw_value)
        if _looks_secret_key(env_key) or _looks_secret_value(env_value):
            raise ExternalHarnessError(
                "external_harness.config.secret_agent_env_forbidden",
                "harness.agent.extra_env must not contain secrets or template references",
            )
        env[env_key] = env_value
    return env


def _strip_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in ({}, [], None)}


def _expected_total_trials(expected_tasks: int | None, n_attempts: int) -> int | None:
    if expected_tasks is None:
        return None
    return expected_tasks * n_attempts


def _invocation_from_plan(plan: TaskBatchHarnessPlan) -> HarborInvocation:
    invocation = _mapping(plan.payload).get("invocation")
    if not isinstance(invocation, HarborInvocation):
        raise ExternalHarnessError(
            "external_harness.runtime.harbor_api_incompatible",
            "Harbor plan does not contain a HarborInvocation",
        )
    return invocation


def _write_launcher_input(invocation: HarborInvocation, *, adapter_id: str) -> None:
    payload = {
        "job_config": invocation.job_config,
        "jobs_dir": str(invocation.jobs_dir),
        "job_name": invocation.job_name,
        "metadata": {
            "adapter_id": adapter_id,
            "expected_total_trials": invocation.expected_total_trials,
        },
    }
    invocation.job_config_path.parent.mkdir(parents=True, exist_ok=True)
    invocation.job_config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _stage_local_registry_mirrors(invocation: HarborInvocation) -> None:
    """Rewrite local registry mirror task paths to absolute paths for Harbor launcher cwd."""

    datasets = invocation.job_config.get("datasets")
    if not isinstance(datasets, list):
        return

    staged_dir = invocation.workdir / "registry_mirrors"
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            continue
        registry_path_value = dataset.get("registry_path")
        if not registry_path_value:
            continue
        registry_path = Path(str(registry_path_value)).expanduser()
        if not registry_path.exists():
            continue

        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
        changed = False
        for registry_dataset in registry_payload if isinstance(registry_payload, list) else []:
            tasks = registry_dataset.get("tasks") if isinstance(registry_dataset, dict) else None
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if not isinstance(task, dict) or task.get("git_url") is not None:
                    continue
                raw_path = task.get("path")
                if not isinstance(raw_path, str) or not raw_path:
                    continue
                resolved = _resolve_local_registry_task_path(raw_path, registry_path=registry_path)
                if str(resolved) != raw_path:
                    task["path"] = str(resolved)
                    changed = True

        if changed:
            staged_dir.mkdir(parents=True, exist_ok=True)
            staged_path = staged_dir / f"{registry_path.stem}-{index}.json"
            staged_path.write_text(json.dumps(registry_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            dataset["registry_path"] = str(staged_path)


def _resolve_local_registry_task_path(raw_path: str, *, registry_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    registry_relative_candidate = (registry_path.parent / path).resolve()
    if registry_relative_candidate.exists():
        return registry_relative_candidate
    return path.resolve()


def _write_raw_input_artifacts(invocation: HarborInvocation) -> None:
    invocation.workdir.mkdir(parents=True, exist_ok=True)
    (invocation.workdir / "invocation.json").write_text(
        json.dumps(_report_safe_value(invocation.to_artifact_dict()), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (invocation.workdir / "job_config.json").write_text(
        json.dumps(_report_safe_value(invocation.job_config), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _write_cancelled_marker(invocation: HarborInvocation, *, reason: str) -> None:
    marker = {
        "status": "cancelled",
        "reason": reason,
        "cancelled_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "job_name": invocation.job_name,
        "jobs_dir": str(invocation.jobs_dir),
        "job_dir": str(invocation.jobs_dir / invocation.job_name),
        "workdir": str(invocation.workdir),
        "job_config_path": str(invocation.job_config_path),
        "launcher_result_path": str(_launcher_result_path_from_invocation(invocation)),
    }
    for marker_path in (
        invocation.workdir / "cancelled.json",
        invocation.jobs_dir / invocation.job_name / "cancelled.json",
    ):
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report_safe_value(value: Any) -> Any:
    return SecretFilter().redact(value).value


def _launcher_timeout_s(params: Mapping[str, Any]) -> float | None:
    harness = _mapping(params.get("harness"))
    launcher = _mapping(harness.get("launcher"))
    value = launcher.get("timeout_s")
    if value is None:
        return None
    return _coerce_float(
        value,
        code="external_harness.config.invalid_launcher",
        label="harness.launcher.timeout_s",
    )


def _launcher_live_log(params: Mapping[str, Any]) -> bool:
    """Resolve ``harness.launcher.live_log`` (default True).

    When True, the parent process mirrors Harbor subprocess stdout/stderr and
    Harbor's internal ``job.log`` lines to its own stderr while the job runs.
    Long Harbor jobs would otherwise appear silent on the operator terminal.
    """

    harness = _mapping(params.get("harness"))
    launcher = _mapping(harness.get("launcher"))
    if "live_log" not in launcher:
        return True
    value = launcher.get("live_log")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ExternalHarnessError(
        "external_harness.config.invalid_launcher",
        "harness.launcher.live_log must be a boolean",
    )


def _launcher_result_path_from_invocation(invocation: HarborInvocation) -> Path:
    argv = list(invocation.launcher_argv or [])
    if "--result-file" in argv:
        index = argv.index("--result-file")
        if index + 1 < len(argv):
            return Path(str(argv[index + 1]))
    return invocation.workdir / "launcher_result.json"


def _launcher_subprocess_payload(result: Any) -> dict[str, Any]:
    return {
        "argv": list(getattr(result, "argv", []) or []),
        "exit_code": int(getattr(result, "exit_code", 0)),
        "timed_out": bool(getattr(result, "timed_out", False)),
        "result_file": str(getattr(result, "result_file", "")),
        "stdout_path": str(getattr(result, "stdout_path", "")),
        "stderr_path": str(getattr(result, "stderr_path", "")),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _expected_parse_trials(payload: Mapping[str, Any], trial_policy: Mapping[str, Any]) -> int | None:
    for value in (
        payload.get("expected_total_trials"),
        trial_policy.get("trials"),
    ):
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _contains_secret(value: Any, secret_values: set[str]) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _looks_secret_key(str(key)):
                return True
            if _contains_secret(nested, secret_values):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_secret(item, secret_values) for item in value)
    if isinstance(value, str):
        if value in secret_values and value not in {"", "EMPTY"}:
            return True
        return _looks_secret_value(value)
    return False


def _looks_secret_key(key: str) -> bool:
    lowered = key.lower()
    if "api_key" in lowered or "apikey" in lowered:
        return True
    parts = {part for part in re.split(r"[^a-z0-9]+", lowered) if part}
    if any(part in parts for part in ("secret", "password", "credential")):
        return True
    return "token" in parts and "per_token" not in lowered


def _looks_secret_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return stripped.startswith(_SECRET_VALUE_PREFIXES) or "${" in stripped


__all__ = ["HarborAdapter", "HarborInvocation"]
