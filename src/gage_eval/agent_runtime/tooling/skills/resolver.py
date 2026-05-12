from __future__ import annotations

from typing import Any

from gage_eval.agent_runtime.tooling.contracts import ToolSchemaIR, ToolingError
from gage_eval.agent_runtime.tooling.skills.manifest import normalize_skill_manifests
from gage_eval.agent_runtime.tooling.skills.policy import SkillPolicy


class SkillManifestResolver:
    """Resolves skill manifests into registry-ready ToolSchemaIR contributions."""

    def __init__(
        self,
        manifests: dict[str, dict[str, Any]],
        *,
        policy: SkillPolicy | None = None,
    ) -> None:
        self._manifests = normalize_skill_manifests(manifests)
        self._policy = policy or SkillPolicy.allow_all()

    def resolve(self, skill_name: str) -> list[ToolSchemaIR]:
        self._policy.ensure_allowed(skill_name)
        manifest = self._manifests.get(skill_name)
        if manifest is None:
            raise ToolingError(
                "client_execution.tool_registry.skill_unavailable",
                f"skill is unavailable: {skill_name}",
                details={"skill": skill_name},
            )
        return [
            ToolSchemaIR.from_provider_schema(dict(raw_tool), provider=f"skill:{skill_name}")
            for raw_tool in manifest.get("tools", [])
            if isinstance(raw_tool, dict)
        ]
