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

    # STEP 2: Normalize trace/footer payloads under the frozen keys.
    fallback_ts_ms = int(time.time() * 1000)
    raw_trace = arena_entry.get("arena_trace")
    if raw_trace is None and isinstance(model_output, Mapping):
        raw_trace = model_output.get("arena_trace")
    trace_steps = _normalize_arena_trace_steps(raw_trace, fallback_timestamp_ms=fallback_ts_ms)
    arena_entry["arena_trace"] = trace_steps

    footer = _build_arena_footer(arena_entry, trace_steps, end_time_ms=end_time_ms)
    arena_entry["game_arena"] = footer

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
    if not isinstance(raw_trace, Sequence) or isinstance(raw_trace, (str, bytes)):
        return []

    normalized: list[dict[str, Any]] = []

    # STEP 1: Canonicalize every step to the frozen required field set.
    for idx, item in enumerate(raw_trace):
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
) -> dict[str, Any]:
    if end_time_ms is None:
        end_time_ms = int(time.time() * 1000)

    total_steps = _coerce_int(output.get("move_count"), len(trace_steps))
    winner = _coerce_optional_str(output.get("winner"))
    termination_reason = str(output.get("reason") or "unknown")
    ranks = _coerce_sequence_list(output.get("ranks"))
    final_scores = _coerce_float_map(output.get("final_scores"))
    episode_returns = _coerce_float_map(output.get("episode_returns"))
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
    arena_markers = (
        "arena_trace",
        "game_arena",
        "move_count",
        "illegal_move_count",
        "final_board",
        "replay_path",
        "game_log",
    )
    return any(marker in entry for marker in arena_markers)


def _reindex_predict_result(predict_result: Sequence[Any]) -> None:
    for idx, entry in enumerate(predict_result):
        if isinstance(entry, dict):
            entry["index"] = idx
