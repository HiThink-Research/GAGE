from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from gage_eval.agent_runtime.tooling.contracts import ToolingError


@dataclass(frozen=True)
class SkillPolicy:
    """Restricts which skill ids may be exposed to one runtime registry."""

    allowed_skill_ids: frozenset[str] | None = None

    @classmethod
    def allow_all(cls) -> "SkillPolicy":
        return cls()

    @classmethod
    def from_iterable(cls, skill_ids: Iterable[str] | None) -> "SkillPolicy":
        if skill_ids is None:
            return cls.allow_all()
        return cls(frozenset(str(skill_id) for skill_id in skill_ids))

    def ensure_allowed(self, skill_id: str) -> None:
        if self.allowed_skill_ids is None or skill_id in self.allowed_skill_ids:
            return
        raise ToolingError(
            "client_execution.tool_registry.skill_policy_denied",
            f"skill is denied by kit policy: {skill_id}",
            details={"skill": skill_id},
        )
