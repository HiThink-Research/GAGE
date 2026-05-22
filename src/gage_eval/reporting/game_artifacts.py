from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class GameArtifactRef:
    name: str
    path: str
    role: str


GAME_ARTIFACT_REF_FIELDS = {
    "replay_ref": "replay",
    "replay_v1_ref": "replay",
    "visual_session_ref": "visual_session",
}


def iter_game_artifact_refs(sample: dict[str, Any]) -> Iterable[GameArtifactRef]:
    artifact_sources = [
        sample.get("artifacts"),
        _nested(sample, "sample", "artifacts"),
        _nested(sample, "model_output", "artifacts"),
        _nested(sample, "model_output", "result", "artifacts"),
        _nested(sample, "judge_output", "artifacts"),
        _nested(sample, "judge_output", "result", "artifacts"),
    ]
    for entry in iter_predict_result_entries(sample):
        artifact_sources.append(entry.get("artifacts"))
        artifact_sources.append(_nested(entry, "result", "artifacts"))

    for artifacts in artifact_sources:
        if not isinstance(artifacts, dict):
            continue
        for key, role in GAME_ARTIFACT_REF_FIELDS.items():
            path = _coerce_str(artifacts.get(key))
            if path:
                yield GameArtifactRef(name=key, path=path, role=role)


def iter_predict_result_entries(sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for container in (sample, sample.get("sample")):
        if not isinstance(container, dict):
            continue
        entries = container.get("predict_result")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                yield entry


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
