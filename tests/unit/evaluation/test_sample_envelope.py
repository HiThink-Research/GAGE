from __future__ import annotations

import pytest

from gage_eval.evaluation import sample_envelope as envelope


def test_append_predict_result_handles_invalid_and_split_payloads() -> None:
    sample: dict = {}
    envelope.append_predict_result(sample, None)
    envelope.append_predict_result(sample, {})
    assert sample == {}

    sample = {"predict_result": {"bad": "shape"}}
    envelope.append_predict_result(sample, {"answer": ["a", "b"], "_sample_n": "2"})
    predict_result = sample["predict_result"]
    assert isinstance(predict_result, list)
    assert [entry["candidate_index"] for entry in predict_result] == [0, 1]
    assert [entry["index"] for entry in predict_result] == [0, 1]
    assert [entry["answer"] for entry in predict_result] == ["a", "b"]
    assert all("message" in entry for entry in predict_result)

    assert envelope._should_split_predict_result({"_sample_n": "x"}, ["a"]) is False
    assert envelope._should_split_predict_result({"_sample_n": 1}, ["a"]) is False
    assert envelope._should_split_predict_result({"_sample_n": 3}, "a") is False


def test_message_normalization_variants() -> None:
    normalized = envelope._build_message({"message": {"role": "assistant", "content": "hello"}})
    assert normalized["content"] == [{"type": "text", "text": "hello"}]

    normalized = envelope._build_message({"messages": [{"role": "assistant", "content": {"type": "text", "text": "x"}}]})
    assert normalized["content"][0]["type"] == "text"

    normalized = envelope._build_message({"answer": {"content": [{"type": "text", "text": "dict"}]}})
    assert normalized["content"][0]["text"] == "dict"

    normalized = envelope._build_message({"answer": [{"type": "text", "text": "list"}]})
    assert normalized["content"][0]["text"] == "list"

    normalized = envelope._build_message({"answer": None})
    assert normalized["content"][0]["text"] == ""

    message = envelope._normalize_message(
        {
            "role": "assistant",
            "content": "body",
            "tool_calls": [{"id": 1}],
            "name": "arena-bot",
        }
    )
    assert message["tool_calls"] == [{"id": 1}]
    assert message["name"] == "arena-bot"

    assert envelope._normalize_content_list("x") == [{"type": "text", "text": "x"}]
    assert envelope._normalize_content_list({"text": "y"}) == [{"type": "text", "text": "y"}]
    assert envelope._normalize_content_list(None) == []
    assert envelope._normalize_content_list(123) == [{"type": "text", "text": "123"}]

    assert envelope._normalize_content({"type": "text", "text": "z"}) == {"type": "text", "text": "z"}
    assert envelope._normalize_content({"type": "text"}) == {"type": "text", "text": ""}
    assert envelope._normalize_content({"type": "image_url", "url": "http://img"}) == {
        "type": "image_url",
        "image_url": {"url": "http://img"},
    }
    assert envelope._normalize_content({"type": "audio_url", "audio_url": {"url": "u"}}) == {
        "type": "audio_url",
        "audio_url": {"url": "u"},
    }
    assert envelope._normalize_content({"text": "legacy"}) == {"type": "text", "text": "legacy"}

    non_json = envelope._normalize_content({"bad": {1, 2, 3}})
    assert non_json["type"] == "text"

    assert envelope._normalize_content("plain") == {"type": "text", "text": "plain"}
    assert envelope._normalize_content(9) == {"type": "text", "text": "9"}


def test_model_and_judge_resolution_helpers() -> None:
    sample = {"predict_result": [{"index": 0, "answer": "a"}], "model_output": {"answer": "legacy"}}
    assert envelope.latest_predict_result(sample) == {"index": 0, "answer": "a"}
    assert envelope.latest_predict_result({"predict_result": [1]}) is None
    assert envelope.latest_predict_result(None) is None

    assert envelope.resolve_model_output(sample, {"answer": "explicit"}) == {"answer": "explicit"}
    assert envelope.resolve_model_output(sample, None)["answer"] == "a"
    assert envelope.resolve_model_output({"model_output": {"x": 1}}, None) == {"x": 1}
    assert envelope.resolve_model_output({}, None) == {}

    out = {"score": 1}
    target = {"eval_result": []}
    envelope.update_eval_result(target, out)
    assert target["eval_result"] == out
    envelope.update_eval_result(target, None)
    assert target["eval_result"] == out

    assert envelope.resolve_judge_output({"eval_result": {"ok": True}}, None) == {"ok": True}
    assert envelope.resolve_judge_output({"judge_output": {"legacy": 1}}, None) == {"legacy": 1}
    assert envelope.resolve_judge_output({}, {"explicit": 1}) == {"explicit": 1}
    assert envelope.resolve_judge_output({}, None) == {}

    serializable = {"x": [1, 2]}
    snap = envelope.snapshot_sample(serializable)
    assert snap == serializable and snap is not serializable

    non_serializable = {"x": object()}
    snap2 = envelope.snapshot_sample(non_serializable)
    assert snap2.keys() == non_serializable.keys()


def test_resolve_selected_predict_result_prefers_explicit_selector_then_falls_back() -> None:
    sample = {
        "selected_predict_result_index": 1,
        "selected_predict_result_id": "ignored-by-index",
        "metadata": {
            "result_selector": {"arena": {"id": "arena-b"}},
            "game_arena": {"result_selector": {"index": 0}},
        },
        "predict_result": [
            {"index": 0, "id": "arena-a"},
            {"index": 1, "id": "arena-b"},
        ],
    }

    assert envelope.resolve_selected_predict_result(sample, domain="arena") == {
        "index": 1,
        "id": "arena-b",
    }

    sample.pop("selected_predict_result_index")
    sample.pop("selected_predict_result_id")
    assert envelope.resolve_selected_predict_result(sample, domain="arena") == {
        "index": 1,
        "id": "arena-b",
    }

    sample["metadata"]["result_selector"] = {"arena": {"index": 99}}
    sample["metadata"]["game_arena"]["result_selector"] = {"id": "missing"}
    assert envelope.resolve_selected_predict_result(sample, domain="arena") == {
        "index": 0,
        "id": "arena-a",
    }


def test_ensure_arena_header_populates_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = {
        "id": "s1",
        "metadata": {
            "player_ids": ["p0", "p1"],
            "env_impl": "gomoku_local_v1",
            "seed": 7,
        },
    }
    envelope.ensure_arena_header(sample, start_time_ms=1000)
    header = sample["metadata"]["game_arena"]
    assert header["engine_id"] == "gomoku_local_v1"
    assert header["seed"] == 7
    assert header["mode"] == "competitive"
    assert header["start_time_ms"] == 1000
    assert [item["player_id"] for item in header["players"]] == ["p0", "p1"]

    monkeypatch.setattr(envelope.time, "time", lambda: 2.5)
    sample2 = {"metadata": "invalid"}
    envelope.ensure_arena_header(sample2)
    assert sample2["metadata"]["game_arena"]["start_time_ms"] == 2500


def test_build_arena_header_and_player_normalization_sources() -> None:
    metadata = {
        "game_type": "tic_env",
        "seed": "5",
        "players": [
            {"player_id": "p0", "controller_type": "agent", "model_id": None, "policy_id": None},
        ],
    }
    header = envelope._build_arena_header(metadata, start_time_ms=111)
    assert header["engine_id"] == "tic_env"
    assert header["mode"] == "single"
    assert header["seed"] == 5

    mapped = envelope._normalize_arena_players(None, {"player_ids": {"a": "p0", "b": "p1"}})
    assert [item["player_id"] for item in mapped] == ["p0", "p1"]

    assert envelope._coerce_players("bad") == []
    assert envelope._coerce_players([{"player_id": "", "controller_type": "x"}]) == []


def test_append_arena_contract_normalizes_paths_and_prunes_duplicates() -> None:
    sample = {
        "id": "s2",
        "metadata": {
            "player_ids": ["p0", "p1"],
            "game_arena": {
                "engine_id": "retro_env_v1",
                "seed": 11,
                "mode": "competitive",
                "players": [
                    {"player_id": "p0", "controller_type": "human", "model_id": None, "policy_id": None},
                    {"player_id": "p1", "controller_type": "agent", "model_id": "m1", "policy_id": None},
                ],
                "start_time_ms": 2000,
            },
        },
        "predict_result": [
            {"index": 0, "message": {"role": "assistant", "content": [{"type": "text", "text": "baseline"}]}},
            {"index": 1, "answer": "other"},
        ],
    }
    output = {
        "winner": "p0",
        "reason": "finished",
        "move_count": 2,
        "final_scores": {"p0": 10, "p1": 8, "drop": "x"},
        "arena_trace": [
            {
                "step_index": 0,
                "trace_state": "in_progress",
                "timestamp": 2001,
                "player_id": "p0",
                "action_raw": "R",
                "action_applied": "R",
                "t_obs_ready_ms": 2001,
                "t_action_submitted_ms": 2003,
                "timeout": False,
                "is_action_legal": True,
                "retry_count": 0,
                "info": {"turn": 0},
                "timeline_id": "t0",
            },
            {
                "step_index": "1",
                "trace_state": "bad_state",
                "timestamp": 2004,
                "player_id": "p1",
                "action_raw": "L",
                "t_obs_ready_ms": 2004,
                "t_action_submitted_ms": 2007,
                "timeout": "yes",
                "is_action_legal": "off",
                "retry_count": -2,
                "illegal_reason": "blocked",
                "reward": {"p0": 1.0, "p1": -1.0},
                "deadline_ms": 2020,
            },
        ],
    }

    envelope.append_arena_contract(sample, output, end_time_ms=5000)

    entry = sample["predict_result"][0]
    footer = entry["game_arena"]
    assert footer["winner_player_id"] == "p0"
    assert footer["termination_reason"] == "finished"
    assert footer["total_steps"] == 2
    assert footer["end_time_ms"] == 5000
    assert footer["final_scores"] == {"p0": 10.0, "p1": 8.0}

    assert entry["arena_trace"][1]["trace_state"] == "done"
    assert entry["arena_trace"][1]["retry_count"] == 0
    assert entry["arena_trace"][1]["action_applied"] == "L"
    assert entry["arena_trace"][1]["timeout"] is True
    assert entry["arena_trace"][1]["is_action_legal"] is False

    assert "winner" not in entry and "reason" not in entry
    assert "move_count" not in entry and "final_scores" not in entry
    assert sample["predict_result"][0]["index"] == 0
    assert sample["predict_result"][1]["index"] == 1
    assert sample["predict_result"][2]["index"] == 2


def test_append_arena_contract_handles_missing_entry_and_footer_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envelope.time, "time", lambda: 3.0)

    sample: dict = {"predict_result": []}
    envelope.append_arena_contract(sample, None)
    assert sample["predict_result"][0]["arena_trace"] == []
    assert sample["predict_result"][0]["game_arena"]["end_time_ms"] == 3000
    assert sample["predict_result"][0]["game_arena"]["total_steps"] == 0

    output = {
        "reason": "",
        "arena_trace": [
            {
                "step_index": 0,
                "trace_state": "done",
                "timestamp": 1,
                "player_id": "p0",
                "action_raw": "x",
                "action_applied": "x",
                "t_obs_ready_ms": 1,
                "t_action_submitted_ms": 2,
                "timeout": False,
                "is_action_legal": True,
                "retry_count": 0,
                "reward": {"p0": 2.5, "p1": "bad"},
            }
        ],
    }
    sample2 = {"predict_result": [{"index": 0, "message": {"role": "assistant", "content": []}}]}
    envelope.append_arena_contract(sample2, output)
    footer = sample2["predict_result"][0]["game_arena"]
    assert footer["termination_reason"] == "unknown"
    assert footer["episode_returns"] == {"p0": 2.5}
    assert sample2["predict_result"][0]["index"] == 0
    assert sample2["predict_result"][1]["index"] == 1


def test_append_arena_contract_derives_footer_from_nested_v2_result() -> None:
    sample = {"predict_result": []}
    output = {
        "output_kind": "arena",
        "sample": {"game_kit": "gomoku", "env": "gomoku_standard"},
        "result": {
            "winner": "Black",
            "reason": "five_in_row",
            "result": "win",
            "move_count": 5,
            "final_scores": {"Black": 1, "White": 0},
        },
        "arena_trace": [
            {
                "step_index": 0,
                "trace_state": "done",
                "timestamp": 1,
                "player_id": "Black",
                "action_raw": "A1",
                "action_applied": "A1",
                "t_obs_ready_ms": 1,
                "t_action_submitted_ms": 1,
                "timeout": False,
                "is_action_legal": True,
                "retry_count": 0,
            }
        ],
    }

    envelope.append_arena_contract(sample, output, end_time_ms=9000)

    footer = sample["predict_result"][0]["game_arena"]
    assert footer["winner_player_id"] == "Black"
    assert footer["termination_reason"] == "five_in_row"
    assert footer["total_steps"] == 5
    assert footer["end_time_ms"] == 9000
    assert footer["final_scores"] == {"Black": 1.0, "White": 0.0}


def test_append_arena_contract_keeps_non_arena_stale_entries_with_artifacts() -> None:
    sample = {
        "metadata": {"player_ids": ["p0", "p1"]},
        "predict_result": [
            {
                "index": 0,
                "answer": "stale-agent-output",
                "artifacts": [{"kind": "image", "path": "/tmp/stale.png"}],
                "message": {"role": "assistant", "content": [{"type": "text", "text": "stale"}]},
            }
        ],
    }
    output = {
        "output_kind": "arena",
        "header": {"game_kit": "gomoku", "env": "gomoku_standard"},
        "footer": {
            "winner_player_id": "p0",
            "termination_reason": "finished",
            "total_steps": 1,
        },
        "trace": [
            {
                "step_index": 0,
                "trace_state": "done",
                "timestamp": 1,
                "player_id": "p0",
                "action_raw": "A1",
                "action_applied": "A1",
                "t_obs_ready_ms": 1,
                "t_action_submitted_ms": 1,
                "timeout": False,
                "is_action_legal": True,
                "retry_count": 0,
            }
        ],
    }

    envelope.ensure_arena_header(sample, start_time_ms=1000)
    envelope.append_arena_contract(sample, output, end_time_ms=2000)

    assert sample["predict_result"][0]["output_kind"] == "arena"
    assert sample["predict_result"][1]["answer"] == "stale-agent-output"
    assert sample["predict_result"][1]["artifacts"] == [{"kind": "image", "path": "/tmp/stale.png"}]
    assert sample["metadata"]["game_arena"]["game_kit"] == "gomoku"
    assert envelope._looks_like_arena_output({"artifacts": [{"kind": "image"}]}) is False


def test_resolve_arena_entry_paths() -> None:
    assert envelope._resolve_arena_entry_index([], None) == -1
    assert envelope._resolve_arena_entry_index([{"answer": "x"}, {"game_arena": {}}], None) == 1
    assert envelope._resolve_arena_entry_index([{"answer": "x"}, {"answer": "y"}], {"index": 0}) == 0
    assert envelope._resolve_arena_entry_index([{"answer": "x"}, {"answer": "y"}], {"index": 9}) == -1
    assert envelope._resolve_arena_entry_index([{"answer": "x"}, {"answer": "y"}], None) == -1
    assert envelope._looks_like_arena_output({"replay_path": "/tmp/a"}) is True
    assert envelope._looks_like_arena_output({"output_kind": "arena"}) is True
    assert envelope._looks_like_arena_output({"answer": "x"}) is False


def test_misc_coercion_helpers() -> None:
    assert envelope._coerce_float_map(None) is None
    assert envelope._coerce_float_map({"a": 1, "b": "2", "c": "x"}) == {"a": 1.0, "b": 2.0}
    assert envelope._coerce_sequence_list("abc") is None
    assert envelope._coerce_sequence_list((1, 2)) == [1, 2]
    assert envelope._coerce_optional_str("") is None
    assert envelope._coerce_optional_str(7) == "7"
    assert envelope._coerce_int("x", 5) == 5
    assert envelope._coerce_bool("true", default=False) is True
    assert envelope._coerce_bool("no", default=True) is False
    assert envelope._coerce_bool(0, default=True) is False

    trace_steps = [
        {"player_id": "p0", "reward": {"p0": 1.0}},
        {"player_id": "p0", "reward": {"p0": 2.0, "p1": "x"}},
        {"player_id": "p1"},
    ]
    assert envelope._derive_episode_returns(trace_steps) == {"p0": 3.0}
