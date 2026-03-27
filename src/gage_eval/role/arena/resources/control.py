from __future__ import annotations

from collections.abc import Mapping

from gage_eval.role.arena.resources.runtime_bridge import (
    RuntimeLease,
    build_runtime_bridge,
)
from gage_eval.role.arena.resources.specs import ArenaResources


class ArenaResourceControl:
    def allocate(self, resource_spec: object) -> ArenaResources:
        normalized_spec = dict(resource_spec) if isinstance(resource_spec, Mapping) else resource_spec
        runtime = RuntimeLease(resource_spec=normalized_spec)
        bridge = build_runtime_bridge(resource_spec=normalized_spec, runtime=runtime)
        return ArenaResources(
            resource_spec=normalized_spec,
            game_runtime=runtime,
            game_bridge=bridge,
        )

    def release(self, resources: ArenaResources) -> None:
        handle = resources.game_runtime
        if handle is not None:
            cleanup_error: Exception | None = None
            for action in (handle.close, handle.terminate, handle.reap):
                try:
                    action()
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if cleanup_error is not None:
                raise cleanup_error
