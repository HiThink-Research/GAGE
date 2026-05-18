from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from gage_eval.reporting.assembly.scenario_profiles import ProfileRefResolver
from gage_eval.reporting.game_artifacts import iter_game_artifact_refs, iter_predict_result_entries


class GameScenarioProfile:
    profile_name = "game"

    def build(self, index: Any, *, ref_resolver: ProfileRefResolver | None = None) -> dict[str, Any]:
        resolver = ref_resolver or ProfileRefResolver.from_index(index)
        illegal_games = 0
        illegal_total = 0
        move_count = 0
        replay_refs: list[str] = []
        game_kits: set[str] = set()
        for sample in getattr(index, "samples", []) or []:
            sample_payload = sample.get("sample") or sample
            metadata = sample_payload.get("metadata") or {}
            game_kit = (metadata.get("game_arena") or {}).get("game_kit")
            if not game_kit:
                dataset_id = sample_payload.get("_dataset_id") or sample_payload.get("dataset_id")
                if isinstance(dataset_id, str) and "gomoku" in dataset_id.lower():
                    game_kit = "gomoku"
            if not game_kit:
                game_kit = _first_text(
                    _iter_nested_values(sample, ("game_arena", "game_kit"), ("game_context", "game_kit"))
                )
            if game_kit:
                game_kits.add(str(game_kit))
            judge = sample.get("judge_output") or {}
            if "illegal_move_count" in judge:
                illegal = _coerce_int(judge.get("illegal_move_count")) or 0
            else:
                illegal = _coerce_int(judge.get("illegal_action_count")) or 0
            if illegal:
                illegal_games += 1
                illegal_total += illegal
            source_paths = {
                _normalize_source_path(getattr(index, "run_dir", None), path)
                for path in _iter_replay_paths(sample)
            }
            sample_move_count = _move_count_for_sample(sample)
            if sample_move_count is None:
                sample_move_count = _move_count_from_replay_manifests(index, sample)
            move_count += sample_move_count or 0
            for path in sorted(source_paths):
                ref_id = resolver.resolve(
                    path,
                    profile="game",
                    field="replay_refs",
                )
                if ref_id:
                    replay_refs.append(ref_id)
        return {
            "profile_version": "gage.scenario.game.v1",
            "game_kits": sorted(game_kits),
            "illegal_actions": {"games": illegal_games, "total": illegal_total},
            "move_count": move_count,
            "replay_refs": sorted(set(replay_refs)),
        }


def _iter_replay_paths(sample: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for artifact in _iter_artifact_ref_dicts(sample):
        name_or_path = str(artifact.get("name") or artifact.get("path") or "")
        if "replay" in name_or_path or "visual_session" in name_or_path:
            path = _coerce_str(artifact.get("path"))
            if path:
                paths.append(path)

    paths.extend(ref.path for ref in iter_game_artifact_refs(sample))

    return paths


def _iter_artifact_ref_dicts(sample: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    candidates = [
        sample.get("artifact_refs"),
        _nested(sample, "sample", "artifact_refs"),
        _nested(sample, "judge_output", "artifact_refs"),
        _nested(sample, "model_output", "artifact_refs"),
        _nested(sample, "model_output", "agent_eval", "trial_aggregate", "trial_result_refs"),
    ]
    for refs_value in candidates:
        if isinstance(refs_value, list):
            refs.extend(ref for ref in refs_value if isinstance(ref, dict))
    return refs


def _move_count_for_sample(sample: dict[str, Any]) -> int | None:
    values = list(
        _iter_nested_values(
            sample,
            ("judge_output", "move_count"),
            ("judge_output", "result", "move_count"),
            ("model_output", "move_count"),
            ("model_output", "result", "move_count"),
            ("model_output", "game_arena", "total_steps"),
            ("model_output", "result", "game_arena", "total_steps"),
            ("sample", "game_arena", "total_steps"),
        )
    )
    for entry in _iter_predict_result_entries(sample):
        values.extend(
            _iter_nested_values(
                entry,
                ("move_count",),
                ("result", "move_count"),
                ("game_arena", "total_steps"),
                ("footer", "total_steps"),
                ("result", "game_arena", "total_steps"),
                ("result", "footer", "total_steps"),
            )
        )
    coerced = [_coerce_int(value) for value in values]
    counts = [value for value in coerced if value is not None]
    for count in counts:
        if count > 0:
            return count
    return counts[0] if counts else None


def _move_count_from_replay_manifests(index: Any, sample: dict[str, Any]) -> int | None:
    run_dir = getattr(index, "run_dir", None)
    if not run_dir:
        return None
    evidence_paths = _evidence_ref_paths(index)
    for source_path in _iter_replay_manifest_paths(sample):
        ref_path = _normalize_source_path(run_dir, source_path)
        if ref_path not in evidence_paths:
            continue
        count = _read_replay_manifest_move_count(Path(run_dir) / ref_path)
        if count is not None:
            return count
    return None


def _iter_replay_manifest_paths(sample: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for artifact in _iter_artifact_ref_dicts(sample):
        name_or_path = str(artifact.get("name") or artifact.get("path") or "")
        if "replay" in name_or_path:
            path = _coerce_str(artifact.get("path"))
            if path:
                paths.append(path)

    paths.extend(ref.path for ref in iter_game_artifact_refs(sample) if ref.role == "replay")
    return paths


def _evidence_ref_paths(index: Any) -> set[str]:
    refs = getattr(index, "evidence_refs", {}) or {}
    values = refs.values() if isinstance(refs, dict) else refs
    paths: set[str] = set()
    for ref in values or []:
        path = ref.get("path") if isinstance(ref, dict) else getattr(ref, "path", None)
        if path:
            paths.add(str(path))
    return paths


def _read_replay_manifest_move_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        return None
    count = _coerce_int(stats.get("move_count"))
    return count if count is not None else None


def _iter_predict_result_entries(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return list(iter_predict_result_entries(sample))


def _iter_nested_values(value: dict[str, Any], *paths: tuple[str, ...]) -> list[Any]:
    values: list[Any] = []
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is not None:
            values.append(current)
    return values


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_text(values: list[Any]) -> str | None:
    for value in values:
        text = _coerce_str(value)
        if text:
            return text
    return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_source_path(run_dir: Any, value: str) -> str:
    if not run_dir:
        return value
    root = Path(run_dir)
    if value.startswith("~"):
        relative = _relative_to_root(root, Path(value).expanduser())
        return relative or value
    if value.startswith("/"):
        relative = _relative_to_root(root, Path(value))
        return relative or value
    if any(part == ".." for part in PurePosixPath(value).parts):
        return value
    if (root / value).exists():
        return value
    parts = PurePosixPath(value).parts
    for index, part in enumerate(parts):
        if part != root.name:
            continue
        candidate = PurePosixPath(*parts[index + 1 :]).as_posix()
        has_parent_segment = any(segment == ".." for segment in PurePosixPath(candidate).parts)
        if candidate and not has_parent_segment and (root / candidate).exists():
            return candidate
    return value


def _relative_to_root(root: Path, path: Path) -> str | None:
    try:
        return path.expanduser().resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return None
