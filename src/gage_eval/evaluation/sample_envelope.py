"""Helpers for mutating and snapshotting the canonical Sample envelope."""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from gage_eval.role.arena.types import ArenaFooter, ArenaHeader, ArenaTraceStep

_MESSAGE_EXTRAS = ("tool_calls", "tool_use", "model_output", "name", "path")
_ARENA_RESULT_DUPLICATE_FIELDS = (
    "winner",
    "reason",
    "move_count",
    "ranks",
    "final_scores",
    "episode_returns",
)


def append_predict_result(sample: Dict[str, Any], model_output: Optional[Dict[str, Any]]) -> None:
    """Append a DUT output entry to `sample["predict_result"]`.

    The original fields are preserved, and a canonical `message` field is added
    if missing. Multi-sample outputs are split into per-candidate entries when
    `_sample_n` is present and `answer` is a list.
    """

    if not isinstance(model_output, dict) or not model_output:
        return

    predict_result = sample.setdefault("predict_result", [])
    if not isinstance(predict_result, list):
        predict_result = sample["predict_result"] = []

    answer = model_output.get("answer")
    if _should_split_predict_result(model_output, answer):
        _append_split_predict_results(predict_result, model_output, answer)
        return

    entry = copy.deepcopy(model_output)
    entry.setdefault("index", len(predict_result))
    if "message" not in entry:
        entry["message"] = _build_message(entry)
    predict_result.append(entry)


def ensure_arena_header(
    sample: Dict[str, Any],
    *,
    start_time_ms: Optional[int] = None,
) -> None:
    """Ensures `sample.metadata.game_arena` follows the frozen header contract.

    Args:
        sample: Mutable sample payload.
        start_time_ms: Optional explicit run start timestamp.
    """

    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        sample["metadata"] = metadata
    if start_time_ms is None:
        start_time_ms = int(time.time() * 1000)
    metadata["game_arena"] = _build_arena_header(metadata, start_time_ms=start_time_ms)


def append_arena_contract(
    sample: Dict[str, Any],
    model_output: Optional[Mapping[str, Any]],
    *,
    end_time_ms: Optional[int] = None,
) -> None:
    """Normalizes arena output into the frozen sample paths.

    This function enforces:
    - `sample.predict_result[0].arena_trace`
    - `sample.predict_result[0].game_arena`

    Args:
        sample: Mutable sample payload.
        model_output: Raw arena model output.
        end_time_ms: Optional explicit run end timestamp.
    """

    predict_result = sample.setdefault("predict_result", [])
    if not isinstance(predict_result, list):
        predict_result = sample["predict_result"] = []

    # STEP 1: Locate the arena result entry and move it to predict_result[0].
    arena_index = _resolve_arena_entry_index(predict_result, model_output)
    if arena_index < 0:
        arena_entry = (
            copy.deepcopy(dict(model_output))
            if isinstance(model_output, Mapping)
            else {}
        )
    else:
        source_entry = predict_result.pop(arena_index)
        arena_entry = dict(source_entry) if isinstance(source_entry, Mapping) else {}
        if isinstance(model_output, Mapping):
            for key, value in model_output.items():
                arena_entry.setdefault(key, copy.deepcopy(value))
    if "message" not in arena_entry:
        message_source = dict(model_output) if isinstance(model_output, Mapping) else arena_entry
        arena_entry["message"] = _build_message(message_source)
    contract_header = _resolve_arena_contract_mapping(arena_entry, model_output, "header")
    if contract_header is not None:
        _merge_arena_header_contract(sample, contract_header)
    contract_sample = _resolve_arena_contract_mapping(arena_entry, model_output, "sample")
    if contract_sample is not None:
        _merge_arena_runtime_metadata(
            sample,
            contract_sample=contract_sample,
            contract_header=contract_header,
        )

    # STEP 2: Normalize trace/footer payloads under the frozen keys.
    fallback_ts_ms = int(time.time() * 1000)
    raw_trace = _resolve_arena_trace_payload(arena_entry, model_output)
    trace_steps = _normalize_arena_trace_steps(raw_trace, fallback_timestamp_ms=fallback_ts_ms)
    arena_entry["arena_trace"] = trace_steps
    if _has_arena_contract_field(arena_entry, model_output, "trace"):
        arena_entry["trace"] = copy.deepcopy(trace_steps)

    footer = _build_arena_footer(
        arena_entry,
        trace_steps,
        end_time_ms=end_time_ms,
        contract_footer=_resolve_arena_contract_mapping(arena_entry, model_output, "footer"),
    )
    arena_entry["game_arena"] = footer
    if _has_arena_contract_field(arena_entry, model_output, "footer"):
        arena_entry["footer"] = copy.deepcopy(footer)

    artifacts = _build_arena_artifacts(
        arena_entry,
        contract_artifacts=_resolve_arena_contract_mapping(arena_entry, model_output, "artifacts"),
    )
    if artifacts is not None:
        arena_entry["artifacts"] = artifacts

    # STEP 3: Remove duplicate semantics from the arena root payload.
    for key in _ARENA_RESULT_DUPLICATE_FIELDS:
        arena_entry.pop(key, None)

    predict_result.insert(0, arena_entry)
    _reindex_predict_result(predict_result)


def _normalize_arena_trace_steps(
    raw_trace: Any,
    *,
    fallback_timestamp_ms: int,
) -> list[dict[str, Any]]:
    trace_source = raw_trace
    if isinstance(trace_source, Mapping):
        # NOTE: Keep compatibility with legacy {"schema": "...", "steps": [...]} payloads.
        legacy_steps = trace_source.get("steps")
        if isinstance(legacy_steps, Sequence) and not isinstance(legacy_steps, (str, bytes)):
            trace_source = legacy_steps
        else:
            return []

    if not isinstance(trace_source, Sequence) or isinstance(trace_source, (str, bytes)):
        return []

    normalized: list[dict[str, Any]] = []

    # STEP 1: Canonicalize every step to the frozen required field set.
    for idx, item in enumerate(trace_source):
        source = dict(item) if isinstance(item, Mapping) else {}
        timestamp = _coerce_int(source.get("timestamp"), fallback_timestamp_ms)
        obs_ready_ms = _coerce_int(source.get("t_obs_ready_ms"), timestamp)
        action_submitted_ms = _coerce_int(source.get("t_action_submitted_ms"), obs_ready_ms)
        state = str(source.get("trace_state") or "done")
        if state not in {"in_progress", "done"}:
            state = "done"

        step = ArenaTraceStep(
            step_index=_coerce_int(source.get("step_index"), idx),
            trace_state=state,
            timestamp=timestamp,
            player_id=str(source.get("player_id") or "unknown"),
            action_raw=source.get("action_raw"),
            action_applied=source.get("action_applied", source.get("action_raw")),
            t_obs_ready_ms=obs_ready_ms,
            t_action_submitted_ms=action_submitted_ms,
            timeout=_coerce_bool(source.get("timeout"), default=False),
            is_action_legal=_coerce_bool(source.get("is_action_legal"), default=True),
            retry_count=max(0, _coerce_int(source.get("retry_count"), 0)),
            illegal_reason=source.get("illegal_reason"),
            info=source.get("info"),
            reward=source.get("reward"),
            timeline_id=source.get("timeline_id"),
            deadline_ms=source.get("deadline_ms"),
        )
        normalized.append(step.to_dict())

    return normalized


def _build_arena_footer(
    output: Mapping[str, Any],
    trace_steps: Sequence[Mapping[str, Any]],
    *,
    end_time_ms: Optional[int] = None,
    contract_footer: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    footer_source = contract_footer if isinstance(contract_footer, Mapping) else {}
    if end_time_ms is None:
        end_time_ms = int(time.time() * 1000)
    end_time_ms = _coerce_int(footer_source.get("end_time_ms"), end_time_ms)

    result = _resolve_arena_result_payload(output)
    explicit_total_steps = "total_steps" in footer_source or "move_count" in footer_source
    total_steps = _coerce_int(
        footer_source.get("total_steps", footer_source.get("move_count")),
        _coerce_int(result.get("move_count"), len(trace_steps)),
    )
    if not explicit_total_steps and trace_steps:
        total_steps = max(total_steps, len(trace_steps))
    winner = (
        _coerce_optional_str(footer_source.get("winner_player_id"))
        or _coerce_optional_str(footer_source.get("winner"))
        or _coerce_optional_str(result.get("winner"))
    )
    termination_reason = str(
        footer_source.get("termination_reason")
        or footer_source.get("reason")
        or result.get("reason")
        or result.get("result")
        or "unknown"
    )
    ranks = _coerce_sequence_list(footer_source.get("ranks"))
    if ranks is None:
        ranks = _coerce_sequence_list(result.get("ranks"))
    final_scores = _coerce_float_map(footer_source.get("final_scores"))
    if final_scores is None:
        final_scores = _coerce_float_map(result.get("final_scores"))
    if final_scores is None:
        final_scores = _coerce_float_map(result.get("scores"))
    episode_returns = _coerce_float_map(footer_source.get("episode_returns"))
    if episode_returns is None:
        episode_returns = _coerce_float_map(result.get("episode_returns"))
    if episode_returns is None:
        episode_returns = _derive_episode_returns(trace_steps)
    footer = ArenaFooter(
        end_time_ms=end_time_ms,
        total_steps=max(0, total_steps),
        winner_player_id=winner,
        termination_reason=termination_reason,
        ranks=ranks,
        final_scores=final_scores,
        episode_returns=episode_returns,
    )
    return footer.to_dict()


def _resolve_arena_result_payload(output: Mapping[str, Any]) -> Mapping[str, Any]:
    result = output.get("result")
    if isinstance(result, Mapping):
        return result
    return output


def _build_arena_header(
    metadata: Mapping[str, Any],
    *,
    start_time_ms: int,
) -> dict[str, Any]:
    existing_header = metadata.get("game_arena")
    header_source = existing_header if isinstance(existing_header, Mapping) else {}
    players = _normalize_arena_players(header_source.get("players"), metadata)
    mode = str(header_source.get("mode") or ("competitive" if len(players) > 1 else "single"))
    engine_id = str(
        header_source.get("engine_id")
        or metadata.get("engine_id")
        or metadata.get("env_impl")
        or metadata.get("game_type")
        or "unknown_engine"
    )
    seed = _coerce_int(header_source.get("seed"), _coerce_int(metadata.get("seed"), 0))
    header = ArenaHeader(
        engine_id=engine_id,
        seed=seed,
        mode=mode,
        players=players,
        start_time_ms=_coerce_int(header_source.get("start_time_ms"), start_time_ms),
    )
    return header.to_dict()


def _merge_arena_header_contract(
    sample: Dict[str, Any],
    contract_header: Mapping[str, Any],
) -> None:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        sample["metadata"] = metadata

    existing_header = metadata.get("game_arena")
    if isinstance(existing_header, Mapping):
        start_time_ms = _coerce_int(
            existing_header.get("start_time_ms"),
            int(time.time() * 1000),
        )
    else:
        start_time_ms = int(time.time() * 1000)

    player_ids = contract_header.get("player_ids")
    if (
        "player_ids" not in metadata
        and isinstance(player_ids, Sequence)
        and not isinstance(player_ids, (str, bytes))
    ):
        metadata["player_ids"] = [str(player_id) for player_id in player_ids if player_id not in (None, "")]

    canonical_header = _build_arena_header(metadata, start_time_ms=start_time_ms)

    engine_id = contract_header.get("engine_id")
    if engine_id in (None, "") and canonical_header.get("engine_id") in (None, "", "unknown_engine"):
        engine_id = contract_header.get("env") or contract_header.get("game_kit")
    if engine_id not in (None, "") and canonical_header.get("engine_id") in (None, "", "unknown_engine"):
        canonical_header["engine_id"] = str(engine_id)

    mode = contract_header.get("mode")
    if mode not in (None, "") and canonical_header.get("mode") in (None, ""):
        canonical_header["mode"] = str(mode)

    seed = contract_header.get("seed")
    if seed not in (None, "") and canonical_header.get("seed") in (None, ""):
        canonical_header["seed"] = _coerce_int(seed, 0)

    for key in ("game_kit", "env", "scheduler"):
        value = contract_header.get(key)
        if value not in (None, ""):
            canonical_header[key] = str(value)

    metadata["game_arena"] = canonical_header


def _merge_arena_runtime_metadata(
    sample: Dict[str, Any],
    *,
    contract_sample: Mapping[str, Any],
    contract_header: Optional[Mapping[str, Any]] = None,
) -> None:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        sample["metadata"] = metadata

    runtime_overrides = contract_sample.get("runtime_overrides")
    if isinstance(runtime_overrides, Mapping):
        for key, value in runtime_overrides.items():
            if value is None:
                continue
            metadata[str(key)] = copy.deepcopy(value)

    runtime_player_ids, runtime_player_names = _extract_runtime_player_metadata(contract_sample)
    if not runtime_player_ids and isinstance(contract_header, Mapping):
        header_player_ids = contract_header.get("player_ids")
        if isinstance(header_player_ids, Sequence) and not isinstance(header_player_ids, (str, bytes)):
            runtime_player_ids = [
                str(player_id)
                for player_id in header_player_ids
                if player_id not in (None, "")
            ]

    if runtime_player_ids:
        metadata["player_ids"] = runtime_player_ids
        if runtime_player_names:
            metadata["player_names"] = runtime_player_names
        elif "player_names" in metadata:
            metadata["player_names"] = {
                player_id: str(existing_name)
                for player_id in runtime_player_ids
                if (existing_name := (
                    metadata.get("player_names", {}).get(player_id)
                    if isinstance(metadata.get("player_names"), Mapping)
                    else None
                )) not in (None, "")
            } or {player_id: player_id for player_id in runtime_player_ids}
        current_start_player = metadata.get("start_player_id")
        if current_start_player not in runtime_player_ids:
            metadata["start_player_id"] = runtime_player_ids[0]


def _extract_runtime_player_metadata(
    contract_sample: Mapping[str, Any],
) -> tuple[list[str], dict[str, str]]:
    players = contract_sample.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return [], {}

    runtime_player_ids: list[str] = []
    runtime_player_names: dict[str, str] = {}
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_id = (
            player.get("player_id")
            or player.get("id")
            or player.get("name")
            or player.get("seat")
        )
        if player_id in (None, ""):
            continue
        normalized_player_id = str(player_id)
        runtime_player_ids.append(normalized_player_id)
        player_name = (
            player.get("player_name")
            or player.get("display_name")
            or player.get("name")
            or normalized_player_id
        )
        runtime_player_names[normalized_player_id] = str(player_name)
    return runtime_player_ids, runtime_player_names


def _normalize_arena_players(
    existing_players: Any,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalized_existing = _coerce_players(existing_players)
    if normalized_existing:
        return normalized_existing

    normalized_metadata_players = _coerce_players(metadata.get("players"))
    if normalized_metadata_players:
        return normalized_metadata_players

    player_ids = metadata.get("player_ids")
    if isinstance(player_ids, Mapping):
        player_ids = list(player_ids.values())
    if isinstance(player_ids, Sequence) and not isinstance(player_ids, (str, bytes)):
        by_ids: list[dict[str, Any]] = []
        for player_id in player_ids:
            if player_id in (None, ""):
                continue
            by_ids.append(
                {
                    "player_id": str(player_id),
                    "controller_type": "unknown",
                    "model_id": None,
                    "policy_id": None,
                }
            )
        return by_ids
    return []


def _coerce_players(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    players: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        player_id = item.get("player_id")
        if player_id in (None, ""):
            continue
        players.append(
            {
                "player_id": str(player_id),
                "controller_type": str(item.get("controller_type") or "unknown"),
                "model_id": item.get("model_id"),
                "policy_id": item.get("policy_id"),
            }
        )
    return players


def _derive_episode_returns(trace_steps: Sequence[Mapping[str, Any]]) -> Optional[dict[str, float]]:
    totals: dict[str, float] = {}
    for step in trace_steps:
        reward_map = _coerce_float_map(step.get("reward"))
        if reward_map is None:
            continue
        for player_id, value in reward_map.items():
            totals[player_id] = totals.get(player_id, 0.0) + value
    return totals or None


def _coerce_float_map(value: Any) -> Optional[dict[str, float]]:
    if not isinstance(value, Mapping):
        return None
    parsed: dict[str, float] = {}
    for key, raw in value.items():
        try:
            parsed[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return parsed or None


def _coerce_sequence_list(value: Any) -> Optional[list[Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return None


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def ensure_predict_result_slot(sample: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """Ensure `sample["predict_result"][index]` exists and return it.

    Args:
        sample: Mutable sample envelope.
        index: Target predict_result index.

    Returns:
        The mutable predict_result entry at `index`.
    """

    normalized_index = max(0, int(index))
    predict_result = sample.setdefault("predict_result", [])
    if not isinstance(predict_result, list):
        predict_result = sample["predict_result"] = []
    while len(predict_result) <= normalized_index:
        predict_result.append({})
    slot = predict_result[normalized_index]
    if not isinstance(slot, dict):
        slot = {}
        predict_result[normalized_index] = slot
    return slot


def set_arena_trace(sample: Dict[str, Any], arena_trace: Any, index: int = 0) -> None:
    """Write canonical arena_trace steps to `sample["predict_result"][index]`.

    Args:
        sample: Mutable sample envelope.
        arena_trace: Arena trace payload (preferred: list[dict], legacy: {"steps": [...]}).
        index: Target predict_result index.
    """

    normalized = _normalize_arena_trace_steps(
        arena_trace,
        fallback_timestamp_ms=int(time.time() * 1000),
    )
    slot = ensure_predict_result_slot(sample, index=index)
    slot["arena_trace"] = normalized


def get_arena_trace(sample: Mapping[str, Any], index: int = 0) -> Optional[list[dict[str, Any]]]:
    """Read canonical arena_trace steps from `sample["predict_result"][index]`.

    Args:
        sample: Sample mapping.
        index: Target predict_result index.

    Returns:
        Canonical trace step list when present, otherwise None.
    """

    predict_result = sample.get("predict_result")
    if not isinstance(predict_result, list):
        return None
    normalized_index = max(0, int(index))
    if normalized_index >= len(predict_result):
        return None
    slot = predict_result[normalized_index]
    if not isinstance(slot, Mapping):
        return None
    if "arena_trace" not in slot:
        return None
    return _normalize_arena_trace_steps(
        slot.get("arena_trace"),
        fallback_timestamp_ms=int(time.time() * 1000),
    )


def resolve_arena_trace(
    sample: Optional[Mapping[str, Any]],
    model_output: Optional[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve canonical arena_trace steps from model_output or sample envelope.

    Args:
        sample: Sample mapping that may contain predict_result entries.
        model_output: Explicit model_output payload when provided.

    Returns:
        Canonical trace step list, preferring explicit model_output first.
    """

    fallback_timestamp_ms = int(time.time() * 1000)

    if isinstance(model_output, Mapping):
        if "trace" in model_output:
            return _normalize_arena_trace_steps(
                model_output.get("trace"),
                fallback_timestamp_ms=fallback_timestamp_ms,
            )
        if "arena_trace" in model_output:
            return _normalize_arena_trace_steps(
                model_output.get("arena_trace"),
                fallback_timestamp_ms=fallback_timestamp_ms,
            )
        raw_index = model_output.get("index")
        if isinstance(raw_index, int):
            resolved = get_arena_trace(sample or {}, index=raw_index)
            if resolved is not None:
                return resolved

    resolved_primary = get_arena_trace(sample or {}, index=0)
    if resolved_primary is not None:
        return resolved_primary

    if isinstance(sample, Mapping):
        legacy = sample.get("arena_trace")
        if legacy is not None:
            return _normalize_arena_trace_steps(
                legacy,
                fallback_timestamp_ms=fallback_timestamp_ms,
            )

    return []


def _should_split_predict_result(model_output: Mapping[str, Any], answer: Any) -> bool:
    if not isinstance(answer, list):
        return False
    sample_n = model_output.get("_sample_n")
    if isinstance(sample_n, int):
        return sample_n > 1
    if isinstance(sample_n, str) and sample_n.isdigit():
        return int(sample_n) > 1
    return False


def _append_split_predict_results(
    predict_result: List[Dict[str, Any]],
    model_output: Dict[str, Any],
    answers: List[Any],
) -> None:
    for idx, answer in enumerate(answers):
        entry = copy.deepcopy(model_output)
        entry["answer"] = answer
        entry.pop("message", None)
        entry["index"] = len(predict_result)
        entry["candidate_index"] = idx
        entry["message"] = _build_message(entry)
        predict_result.append(entry)


def latest_predict_result(sample: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the latest predict_result entry if available.

    Args:
        sample: Sample mapping that may contain predict_result.

    Returns:
        The most recent predict_result dict, or None when unavailable.
    """

    if not isinstance(sample, Mapping):
        return None
    predict_result = sample.get("predict_result")
    if isinstance(predict_result, list) and predict_result:
        latest = predict_result[-1]
        if isinstance(latest, Mapping):
            return dict(latest)
    return None


def resolve_selected_predict_result(
    sample: Optional[Mapping[str, Any]],
    *,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve one selected predict_result entry with backward-compatible fallback."""

    if not isinstance(sample, Mapping):
        return {}
    predict_result = sample.get("predict_result")
    if not isinstance(predict_result, Sequence) or isinstance(
        predict_result,
        (str, bytes),
    ):
        return {}

    selected_entry = _resolve_selected_predict_result_entry(
        sample,
        predict_result,
        domain=domain,
    )
    return dict(selected_entry) if isinstance(selected_entry, Mapping) else {}


def resolve_model_output(
    sample: Optional[Mapping[str, Any]],
    model_output: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve model_output with predict_result and legacy fallbacks.

    Args:
        sample: Sample mapping that may contain predict_result or model_output.
        model_output: Explicit model_output payload when provided.

    Returns:
        A dict-shaped model_output, preferring explicit payload then predict_result.
    """

    if isinstance(model_output, Mapping) and model_output:
        return dict(model_output)
    latest = latest_predict_result(sample)
    if latest:
        return latest
    if isinstance(sample, Mapping):
        legacy = sample.get("model_output")
        if isinstance(legacy, Mapping) and legacy:
            return dict(legacy)
    return {}


def _resolve_selected_predict_result_entry(
    sample: Mapping[str, Any],
    predict_result: Sequence[Any],
    *,
    domain: Optional[str],
) -> Mapping[str, Any]:
    selected_index = _resolve_explicit_selected_predict_result_index(sample, predict_result)
    if selected_index is not None:
        entry = predict_result[selected_index]
        if isinstance(entry, Mapping):
            return entry

    selected_id = _resolve_explicit_selected_predict_result_id(sample)
    if selected_id is not None:
        for entry in predict_result:
            if not isinstance(entry, Mapping):
                continue
            entry_id = entry.get("id")
            if entry_id is not None and str(entry_id) == selected_id:
                return entry

    for selector in _selector_sources(sample, domain=domain):
        selector_entry = _resolve_selector_entry(selector, predict_result)
        if selector_entry:
            return selector_entry

    if predict_result:
        first = predict_result[0]
        if isinstance(first, Mapping):
            return first
    return {}


def _resolve_explicit_selected_predict_result_index(
    sample: Mapping[str, Any],
    predict_result: Sequence[Any],
) -> Optional[int]:
    for raw_value in (sample.get("selected_predict_result_index"),):
        try:
            resolved = int(raw_value)
        except (TypeError, ValueError):
            continue
        if 0 <= resolved < len(predict_result):
            return resolved
    return None


def _resolve_explicit_selected_predict_result_id(
    sample: Mapping[str, Any],
) -> Optional[str]:
    raw_value = sample.get("selected_predict_result_id")
    if raw_value in (None, ""):
        return None
    return str(raw_value)


def _resolve_selector_entry(
    selector: Mapping[str, Any],
    predict_result: Sequence[Any],
) -> Mapping[str, Any]:
    raw_index = selector.get("selected_predict_result_index", selector.get("index"))
    try:
        selected_index = int(raw_index)
    except (TypeError, ValueError):
        selected_index = None
    if selected_index is not None and 0 <= selected_index < len(predict_result):
        entry = predict_result[selected_index]
        if isinstance(entry, Mapping):
            return entry

    selected_id = (
        selector.get("selected_predict_result_id")
        or selector.get("predict_result_id")
        or selector.get("id")
    )
    if selected_id not in (None, ""):
        for entry in predict_result:
            if not isinstance(entry, Mapping):
                continue
            entry_id = entry.get("id")
            if entry_id is not None and str(entry_id) == str(selected_id):
                return entry
    return {}


def _selector_sources(
    sample: Mapping[str, Any],
    *,
    domain: Optional[str],
) -> list[Mapping[str, Any]]:
    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        return []

    sources: list[Mapping[str, Any]] = []
    raw_selector = metadata.get("result_selector")
    if isinstance(raw_selector, Mapping):
        sources.append(raw_selector)
        if domain and isinstance(raw_selector.get(domain), Mapping):
            sources.append(raw_selector.get(domain))

    if domain == "arena":
        arena_meta = metadata.get("game_arena")
        if isinstance(arena_meta, Mapping):
            arena_selector = arena_meta.get("result_selector")
            if isinstance(arena_selector, Mapping):
                sources.append(arena_selector)
    return sources


def update_eval_result(sample: Dict[str, Any], judge_output: Optional[Dict[str, Any]]) -> None:
    """Merges judge output into `sample["eval_result"]`."""

    if not isinstance(judge_output, dict) or not judge_output:
        return
    eval_result = sample.setdefault("eval_result", {})
    if not isinstance(eval_result, dict):
        eval_result = sample["eval_result"] = {}
    for key, value in judge_output.items():
        eval_result[key] = copy.deepcopy(value)


def resolve_judge_output(
    sample: Optional[Mapping[str, Any]],
    judge_output: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve judge_output with eval_result and legacy fallbacks.

    Args:
        sample: Sample mapping that may contain eval_result or judge_output.
        judge_output: Explicit judge_output payload when provided.

    Returns:
        A dict-shaped judge_output, preferring explicit payload then eval_result.
    """

    if isinstance(judge_output, Mapping) and judge_output:
        return dict(judge_output)
    if isinstance(sample, Mapping):
        eval_result = sample.get("eval_result")
        if isinstance(eval_result, Mapping) and eval_result:
            return dict(eval_result)
        legacy = sample.get("judge_output")
        if isinstance(legacy, Mapping) and legacy:
            return dict(legacy)
    return {}


def snapshot_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a deep snapshot of a sample for safe persistence."""

    try:
        return json.loads(json.dumps(sample, ensure_ascii=False))
    except (TypeError, ValueError):
        # Fall back to deepcopy. This allows non-JSON-serializable fields and lets
        # the writer handle serialization later.
        return copy.deepcopy(sample)


def _build_message(output: Dict[str, Any]) -> Dict[str, Any]:
    message = output.get("message")
    if isinstance(message, dict):
        return _normalize_message(message)

    messages = output.get("messages")
    if isinstance(messages, Sequence) and messages:
        candidate = messages[-1]
        if isinstance(candidate, dict):
            return _normalize_message(candidate)

    answer = output.get("answer") or output.get("text") or output.get("content")
    if isinstance(answer, dict):
        return _normalize_message({"role": "assistant", "content": answer.get("content") or answer})
    if isinstance(answer, list):
        return _normalize_message({"role": "assistant", "content": answer})
    return {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "" if answer is None else str(answer),
            }
        ],
    }


def _normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    role = message.get("role") or "assistant"
    content = message.get("content")
    normalized = {
        "role": role,
        "content": _normalize_content_list(content),
    }
    for extra in _MESSAGE_EXTRAS:
        if extra in message:
            normalized[extra] = copy.deepcopy(message[extra])
    return normalized


def _normalize_content_list(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        return [_normalize_content(fragment) for fragment in content if fragment is not None]
    if isinstance(content, dict):
        return [_normalize_content(content)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def _normalize_content(fragment: Any) -> Dict[str, Any]:
    if isinstance(fragment, dict):
        fragment_type = fragment.get("type")
        if fragment_type in {"text", "image_url", "audio_url", "video_url", "file_url"}:
            payload = fragment.get(fragment_type) or fragment.get("url")
            if fragment_type == "text":
                text_value = fragment.get("text")
                if text_value is not None:
                    return {"type": "text", "text": str(text_value)}
                return {"type": "text", "text": ""}
            if isinstance(payload, dict):
                return {"type": fragment_type, fragment_type: payload}
            if payload is not None:
                return {"type": fragment_type, fragment_type: {"url": str(payload)}}
        if "text" in fragment and not fragment.get("type"):
            return {"type": "text", "text": str(fragment["text"])}
        try:
            return {"type": "text", "text": json.dumps(fragment, ensure_ascii=False)}
        except (TypeError, ValueError):
            return {"type": "text", "text": str(fragment)}

    if isinstance(fragment, str):
        return {"type": "text", "text": fragment}

    return {"type": "text", "text": str(fragment)}


def _resolve_arena_entry_index(
    predict_result: Sequence[Any],
    model_output: Optional[Mapping[str, Any]],
) -> int:
    if not predict_result:
        return -1

    # STEP 1: Prioritize entries already carrying arena contract fields.
    for idx in range(len(predict_result) - 1, -1, -1):
        entry = predict_result[idx]
        if isinstance(entry, Mapping) and _looks_like_arena_output(entry):
            return idx

    # STEP 2: Fall back to model_output index when available.
    if isinstance(model_output, Mapping):
        raw_index = model_output.get("index")
        if isinstance(raw_index, int) and 0 <= raw_index < len(predict_result):
            return raw_index

    # STEP 3: No explicit arena entry found.
    return -1


def _looks_like_arena_output(entry: Mapping[str, Any]) -> bool:
    if str(entry.get("output_kind") or "").strip().lower() == "arena":
        return True
    if str(entry.get("domain") or "").strip().lower() == "arena":
        return True

    legacy_markers = (
        "arena_trace",
        "game_arena",
        "move_count",
        "illegal_move_count",
        "final_board",
        "replay_path",
        "game_log",
    )
    if any(marker in entry for marker in legacy_markers):
        return True

    footer = entry.get("footer")
    trace = entry.get("trace")
    sample_payload = entry.get("sample")
    return _looks_like_arena_footer_contract(footer) and (
        _looks_like_arena_trace_contract(trace)
        or _looks_like_arena_sample_contract(sample_payload)
    )


def _looks_like_arena_footer_contract(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(
        key in value
        for key in (
            "winner_player_id",
            "termination_reason",
            "total_steps",
            "winner",
            "reason",
            "move_count",
        )
    )


def _looks_like_arena_trace_contract(value: Any) -> bool:
    trace_source = value
    if isinstance(trace_source, Mapping):
        trace_source = trace_source.get("steps")
    if not isinstance(trace_source, Sequence) or isinstance(trace_source, (str, bytes)):
        return False
    for item in trace_source:
        if not isinstance(item, Mapping):
            continue
        if "player_id" in item or "step_index" in item or "trace_state" in item:
            return True
    return False


def _looks_like_arena_sample_contract(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    game_kit = value.get("game_kit")
    return isinstance(game_kit, str) and bool(game_kit.strip())


def _resolve_arena_trace_payload(
    output: Mapping[str, Any],
    model_output: Optional[Mapping[str, Any]],
) -> Any:
    contract_trace = _resolve_arena_contract_field(output, model_output, "trace")
    if contract_trace is not None:
        return contract_trace
    return _resolve_arena_contract_field(output, model_output, "arena_trace")


def _build_arena_artifacts(
    output: Mapping[str, Any],
    *,
    contract_artifacts: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    artifacts = dict(contract_artifacts) if isinstance(contract_artifacts, Mapping) else {}
    replay_ref = artifacts.get("replay_ref")
    if replay_ref in (None, ""):
        replay_ref = output.get("replay_path")
    if replay_ref in (None, ""):
        replay_ref = _resolve_arena_result_payload(output).get("replay_path")
    if replay_ref not in (None, ""):
        artifacts["replay_ref"] = str(replay_ref)
    replay_v1_ref = output.get("replay_v1_path")
    if replay_v1_ref not in (None, "") and artifacts.get("replay_v1_ref") in (None, ""):
        artifacts["replay_v1_ref"] = str(replay_v1_ref)
    game_log_ref = output.get("game_log_path")
    if game_log_ref not in (None, "") and artifacts.get("game_log_ref") in (None, ""):
        artifacts["game_log_ref"] = str(game_log_ref)
    return artifacts or None


def _resolve_arena_contract_mapping(
    output: Mapping[str, Any],
    model_output: Optional[Mapping[str, Any]],
    field: str,
) -> Optional[Mapping[str, Any]]:
    value = _resolve_arena_contract_field(output, model_output, field)
    if isinstance(value, Mapping):
        return value
    return None


def _has_arena_contract_field(
    output: Mapping[str, Any],
    model_output: Optional[Mapping[str, Any]],
    field: str,
) -> bool:
    return _resolve_arena_contract_field(output, model_output, field) is not None


def _resolve_arena_contract_field(
    output: Mapping[str, Any],
    model_output: Optional[Mapping[str, Any]],
    field: str,
) -> Any:
    value = output.get(field)
    if value is not None:
        return value
    if isinstance(model_output, Mapping):
        return model_output.get(field)
    return None


def _reindex_predict_result(predict_result: Sequence[Any]) -> None:
    for idx, entry in enumerate(predict_result):
        if isinstance(entry, dict):
            entry["index"] = idx
