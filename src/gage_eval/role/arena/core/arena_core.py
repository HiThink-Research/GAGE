from __future__ import annotations

import inspect

from gage_eval.role.arena.core.game_session import GameSession
from gage_eval.role.arena.resources.control import ResourceLifecycleError


class GameArenaCore:
    def __init__(self, *, resolver, resource_control, output_writer) -> None:
        self.resolver = resolver
        self.resource_control = resource_control
        self.output_writer = output_writer

    def run_sample(self, sample, *, invocation_context=None):
        resolved = self.resolver.resolve(sample)
        resources = None
        try:
            resources = self.resource_control.allocate(resolved.resource_spec)
            if _accepts_keyword(GameSession.from_resolved, "invocation_context"):
                session = GameSession.from_resolved(
                    sample,
                    resolved,
                    resources,
                    invocation_context=invocation_context,
                )
            else:
                session = GameSession.from_resolved(sample, resolved, resources)
            resolved.scheduler.run(session)
            session.finalize()
            if resources is not None:
                try:
                    self.resource_control.release(resources)
                except ResourceLifecycleError:
                    pass
            return self.output_writer.finalize(session)
        except Exception:
            if resources is not None:
                try:
                    self.resource_control.release(resources)
                except ResourceLifecycleError:
                    pass
            raise


def _accepts_keyword(callable_obj, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    if keyword in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
