from __future__ import annotations

from typing import Any


SkillManifest = dict[str, Any]


def normalize_skill_manifests(manifests: dict[str, dict[str, Any]]) -> dict[str, SkillManifest]:
    """Normalize manifest ids and shallow-copy manifest payloads."""

    return {str(name): dict(value) for name, value in manifests.items()}
