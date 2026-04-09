"""Staged builder for PipelineConfig objects."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.config.schema import SchemaValidationError, normalize_pipeline_payload


class PipelineConfigBuildError(ValueError):
    """Raised when a PipelineConfig fails to build in a specific stage."""

    def __init__(self, *, code: str, stage: str, field_path: str, message: str) -> None:
        self.code = code
        self.stage = stage
        self.field_path = field_path
        self.detail = message
        super().__init__(f"{code} [{stage}] {field_path}: {message}")


class PipelineConfigBuilder:
    """Incrementally materialize a :class:`PipelineConfig` from a raw payload."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self._normalized: Optional[Dict[str, Any]] = None
        self._state: Dict[str, Any] = {}

    def normalize(self) -> "PipelineConfigBuilder":
        try:
            self._normalized = normalize_pipeline_payload(self._payload)
        except SchemaValidationError as exc:
            raise PipelineConfigBuildError(
                code="E_NORMALIZE",
                stage="normalize",
                field_path="<payload>",
                message=str(exc),
            ) from exc
        return self

    def build_root(self) -> "PipelineConfigBuilder":
        normalized = self._require_normalized()

        from gage_eval.config.pipeline_config import (
            BuiltinPipelineSpec,
            CustomPipelineSpec,
            CustomPipelineStep,
        )

        metadata = normalized.get("metadata", {})
        builtin = None
        if normalized.get("builtin"):
            builtin_entry = normalized["builtin"]
            builtin = BuiltinPipelineSpec(
                pipeline_id=builtin_entry.get("pipeline_id"),
                params=builtin_entry.get("params") or builtin_entry.get("args", {}),
            )

        custom = None
        if normalized.get("custom"):
            steps = []
            for index, step in enumerate(normalized["custom"].get("steps", [])):
                try:
                    steps.append(
                        CustomPipelineStep(
                            step_type=step.get("step"),
                            adapter_id=step.get("adapter_id") or step.get("role_ref"),
                            params=step.get("params") or {},
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive wrap
                    raise self._build_error(
                        code="E_BUILD_CUSTOM_STEPS",
                        stage="build_root",
                        field_path=f"custom.steps[{index}]",
                        message=str(exc),
                    ) from exc
            custom = CustomPipelineSpec(steps=tuple(steps))

        self._state.update(
            {
                "metadata": metadata,
                "builtin": builtin,
                "custom": custom,
            }
        )
        return self

    def build_assets(self) -> "PipelineConfigBuilder":
        normalized = self._require_normalized()

        from gage_eval.config.pipeline_config import (
            AgentBackendSpec,
            BackendSpec,
            DatasetSpec,
            McpClientSpec,
            ModelSpec,
            PromptTemplateSpec,
            SandboxProfileSpec,
            _strip_known_keys,
        )

        self._state["models"] = tuple(
            ModelSpec(
                model_id=item.get("model_id"),
                source=item.get("source"),
                hub=item.get("hub"),
                hub_params=item.get("hub_params") or item.get("hub_args", {}),
                params=item.get("params") or {},
            )
            for item in normalized.get("models", [])
        )

        self._state["datasets"] = tuple(
            DatasetSpec(
                dataset_id=item.get("dataset_id"),
                loader=item.get("loader"),
                hub=item.get("hub"),
                hub_params=item.get("hub_params") or item.get("hub_args", {}),
                preprocess_chain=tuple(item.get("preprocess_chain", [])),
                params=item.get("params") or {},
                schema=item.get("schema"),
            )
            for item in normalized.get("datasets", [])
        )

        self._state["backends"] = tuple(
            BackendSpec(
                backend_id=item.get("backend_id"),
                type=item.get("type"),
                config=item.get("config", {}),
            )
            for item in normalized.get("backends", [])
        )

        self._state["agent_backends"] = tuple(
            AgentBackendSpec(
                agent_backend_id=item.get("agent_backend_id"),
                type=item.get("type"),
                config=item.get("config", {}),
                backend_id=item.get("backend_id"),
            )
            for item in normalized.get("agent_backends", [])
        )

        self._state["sandbox_profiles"] = tuple(
            SandboxProfileSpec(
                sandbox_id=item.get("sandbox_id") or item.get("template_name"),
                runtime=item.get("runtime"),
                image=item.get("image"),
                resources=item.get("resources") or {},
                runtime_configs=item.get("runtime_configs") or {},
                params=item.get("params") or {},
                extra=_strip_known_keys(
                    item,
                    {
                        "sandbox_id",
                        "template_name",
                        "runtime",
                        "image",
                        "resources",
                        "runtime_configs",
                        "params",
                    },
                ),
            )
            for item in normalized.get("sandbox_profiles", [])
        )

        self._state["mcp_clients"] = tuple(
            McpClientSpec(
                mcp_client_id=item.get("mcp_client_id"),
                transport=item.get("transport"),
                endpoint=item.get("endpoint"),
                timeout_s=item.get("timeout_s"),
                allowlist=tuple(item.get("allowlist", [])),
                params=item.get("params") or {},
            )
            for item in normalized.get("mcp_clients", [])
        )

        self._state["prompts"] = tuple(
            PromptTemplateSpec(
                prompt_id=item.get("prompt_id"),
                renderer=item.get("renderer"),
                template=item.get("template"),
                params=item.get("params") or {},
            )
            for item in normalized.get("prompts", [])
        )
        return self

    def build_role_adapters(self) -> "PipelineConfigBuilder":
        normalized = self._require_normalized()

        from gage_eval.config.pipeline_config import RoleAdapterSpec

        self._state["role_adapters"] = tuple(
            RoleAdapterSpec(
                adapter_id=item.get("adapter_id"),
                role_type=item.get("role_type"),
                class_path=item.get("class_path"),
                parallel=item.get("parallel", {}),
                sandbox=item.get("sandbox", {}),
                resource_requirement=item.get("resource_requirement", {}),
                capabilities=tuple(item.get("capabilities", [])),
                params=item.get("params") or {},
                backend_id=item.get("backend_id"),
                backend=dict(item.get("backend", {})) if item.get("backend") else None,
                agent_backend_id=item.get("agent_backend_id"),
                agent_backend=dict(item.get("agent_backend", {})) if item.get("agent_backend") else None,
                agent_runtime_id=item.get("agent_runtime_id"),
                compat_runtime_id=item.get("compat_runtime_id"),
                mcp_client_id=item.get("mcp_client_id"),
                prompt_id=item.get("prompt_id"),
                prompt_params=item.get("prompt_params") or item.get("prompt_args", {}),
            )
            for item in normalized.get("role_adapters", [])
        )
        return self

    def build_metrics(self) -> "PipelineConfigBuilder":
        normalized = self._require_normalized()

        from gage_eval.config.pipeline_config import MetricSpec, _normalize_metric_entry, _normalize_summary_generator_entry

        metrics = []
        for index, entry in enumerate(normalized.get("metrics", [])):
            try:
                metrics.append(MetricSpec(**_normalize_metric_entry(entry)))
            except Exception as exc:
                raise self._build_error(
                    code="E_BUILD_METRICS",
                    stage="build_metrics",
                    field_path=f"metrics[{index}]",
                    message=str(exc),
                ) from exc

        summary_generators = []
        for index, entry in enumerate(normalized.get("summary_generators", [])):
            try:
                summary_generators.append(_normalize_summary_generator_entry(entry))
            except Exception as exc:
                raise self._build_error(
                    code="E_BUILD_SUMMARY_GENERATORS",
                    stage="build_metrics",
                    field_path=f"summary_generators[{index}]",
                    message=str(exc),
                ) from exc

        self._state["metrics"] = tuple(metrics)
        self._state["summary_generators"] = tuple(summary_generators)
        return self

    def build_tasks(self) -> "PipelineConfigBuilder":
        normalized = self._require_normalized()

        from gage_eval.config.pipeline_config import CustomPipelineStep, MetricSpec, TaskSpec, _normalize_metric_entry

        tasks = []
        for task_index, item in enumerate(normalized.get("tasks", [])):
            try:
                steps = tuple(
                    CustomPipelineStep(
                        step_type=step.get("step"),
                        adapter_id=step.get("adapter_id") or step.get("role_ref"),
                        params=step.get("params") or {},
                    )
                    for step in item.get("steps", [])
                )
                metric_overrides = tuple(
                    MetricSpec(**_normalize_metric_entry(metric))
                    for metric in item.get("metric_overrides", [])
                )
                tasks.append(
                    TaskSpec(
                        task_id=item.get("task_id"),
                        dataset_id=item.get("dataset_id") or item.get("dataset_ref"),
                        steps=steps,
                        metric_overrides=metric_overrides,
                        reporting=item.get("reporting", {}),
                        max_samples=item.get("max_samples"),
                        shuffle=item.get("shuffle"),
                        shuffle_seed=item.get("shuffle_seed"),
                        shuffle_strategy=item.get("shuffle_strategy"),
                        shuffle_small_dataset_threshold=item.get(
                            "shuffle_small_dataset_threshold"
                        ),
                        keep_shuffle_artifacts=item.get("keep_shuffle_artifacts"),
                        concurrency=item.get("concurrency"),
                        prefetch_factor=item.get("prefetch_factor"),
                        max_inflight=item.get("max_inflight"),
                        failure_policy=item.get("failure_policy"),
                        metric_concurrency=item.get("metric_concurrency"),
                        report_partial_on_failure=item.get("report_partial_on_failure"),
                        support_payload_policy=item.get("support_payload_policy") or {},
                    )
                )
            except Exception as exc:
                raise self._build_error(
                    code="E_BUILD_TASKS",
                    stage="build_tasks",
                    field_path=f"tasks[{task_index}]",
                    message=str(exc),
                ) from exc

        self._state["tasks"] = tuple(tasks)
        return self

    def build(self):  # return PipelineConfig
        from gage_eval.config.pipeline_config import PipelineConfig
        normalized = self._require_normalized()

        return PipelineConfig(
            metadata=self._state.get("metadata", {}),
            builtin=self._state.get("builtin"),
            custom=self._state.get("custom"),
            datasets=self._state.get("datasets", ()),
            models=self._state.get("models", ()),
            backends=self._state.get("backends", ()),
            agent_backends=self._state.get("agent_backends", ()),
            sandbox_profiles=self._state.get("sandbox_profiles", ()),
            mcp_clients=self._state.get("mcp_clients", ()),
            prompts=self._state.get("prompts", ()),
            role_adapters=self._state.get("role_adapters", ()),
            metrics=self._state.get("metrics", ()),
            tasks=self._state.get("tasks", ()),
            summary_generators=self._state.get("summary_generators", ()),
            observability=normalized.get("observability") or {},
        )

    def _require_normalized(self) -> Dict[str, Any]:
        if self._normalized is None:
            raise RuntimeError("PipelineConfigBuilder.normalize() must run before build stages")
        return self._normalized

    def _build_error(self, *, code: str, stage: str, field_path: str, message: str) -> PipelineConfigBuildError:
        return PipelineConfigBuildError(
            code=code,
            stage=stage,
            field_path=field_path,
            message=message,
        )
