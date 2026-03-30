from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gage_eval.role.arena.resources.runtime_bridge import (
    RuntimeLease,
    build_runtime_bridge,
)
from gage_eval.role.arena.resources.specs import ArenaResources


class ResourceLifecycleError(RuntimeError):
    def __init__(self, message: str, *, errors: list[dict[str, object]]) -> None:
        super().__init__(message)
        self.error_code = "resource_lifecycle_error"
        self.errors = tuple(errors)


class ArenaResourceControl:
    def allocate(self, resource_spec: object) -> ArenaResources:
        normalized_spec = dict(resource_spec) if isinstance(resource_spec, Mapping) else resource_spec
        runtime = RuntimeLease(resource_spec=normalized_spec)
        bridge = build_runtime_bridge(resource_spec=normalized_spec, runtime=runtime)
        resources = ArenaResources(
            resource_spec=normalized_spec,
            resource_categories=(
                "game_runtime_resource",
                "game_bridge_resource",
            ),
            game_runtime=runtime,
            game_bridge=bridge,
        )
        resources.record_lifecycle("allocated", resource_category="game_runtime_resource")
        resources.record_lifecycle("allocated", resource_category="game_bridge_resource")
        return resources

    def release(self, resources: ArenaResources) -> None:
        if getattr(resources, "lifecycle_phase", "") == "released":
            return
        resources.record_lifecycle("releasing")
        errors: list[dict[str, object]] = []
        self._release_visualization(resources, errors)
        self._capture_resource_artifacts(resources)
        handle = resources.game_runtime
        if handle is not None:
            for action in (handle.close, handle.terminate, handle.reap):
                try:
                    action()
                except Exception as exc:
                    errors.append(
                        {
                            "error_code": "resource_lifecycle_error",
                            "resource_category": "game_runtime_resource",
                            "operation": getattr(action, "__name__", "unknown"),
                            "message": str(exc),
                        }
                    )
            resources.record_lifecycle("released", resource_category="game_runtime_resource")
        if errors:
            resources.errors.extend(errors)
            resources.record_lifecycle("release_failed")
            raise ResourceLifecycleError(errors[0]["message"], errors=errors)
        resources.record_lifecycle("released")

    @staticmethod
    def _capture_resource_artifacts(resources: ArenaResources) -> None:
        for candidate in (
            getattr(resources, "visualization", None),
            getattr(resources, "output", None),
        ):
            if candidate is None:
                continue
            for attr_name in ("resource_artifacts", "artifacts"):
                value = getattr(candidate, attr_name, None)
                if isinstance(value, Mapping):
                    resources.resource_artifacts.update(
                        {str(key): item for key, item in value.items()}
                    )

    @staticmethod
    def _release_visualization(
        resources: ArenaResources,
        errors: list[dict[str, object]],
    ) -> None:
        visualization = getattr(resources, "visualization", None)
        if visualization is None:
            return
        releaser = getattr(visualization, "release", None)
        if callable(releaser):
            try:
                releaser()
            except Exception as exc:
                errors.append(
                    {
                        "error_code": "resource_lifecycle_error",
                        "resource_category": "visualization_resource",
                        "operation": "release",
                        "message": str(exc),
                    }
                )
        resources.record_lifecycle("released", resource_category="visualization_resource")
