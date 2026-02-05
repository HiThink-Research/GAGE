from gage_eval.role.arena.games.retro import observation as retro_observation


def test_action_schema_prompt_includes_constraints_and_legal_moves():
    schema = retro_observation.ActionSchema(hold_ticks_min=2, hold_ticks_max=5, default_hold_ticks=3)
    prompt = schema.format_prompt(["noop", "right"])

    assert "Output ONE JSON object" in prompt
    assert "hold_ticks range: 2-5 (default 3)" in prompt
    assert "Legal moves: noop, right." in prompt


def test_json_dumps_falls_back_for_unserializable_objects():
    class Unserializable:
        def __repr__(self) -> str:
            return "<unserializable>"

    dumped = retro_observation._json_dumps({"x": Unserializable()})
    assert "<unserializable>" in dumped


def test_truncate_text_handles_zero_budget_and_long_payloads():
    assert retro_observation._truncate_text("hello", 0) == ""
    assert retro_observation._truncate_text("x" * 10, 1) == "xxxx..."


def test_numeric_delta_keeps_only_changed_numeric_keys():
    previous = {"score": 1, "lives": 3, "name": "mario"}
    current = {"score": 2, "lives": 3, "name": "mario"}

    assert retro_observation._numeric_delta(previous, current) == {"score": 1.0}


def test_info_delta_feeder_builds_delta_payload_and_text():
    feeder = retro_observation.InfoDeltaFeeder(window_size=2)
    text, payload = feeder.build(
        info_history=[{"score": 1, "name": "mario"}, {"score": 3, "name": "mario"}],
        raw_info={"score": 3, "name": "mario"},
        token_budget=20,
    )

    assert "delta" in text
    assert payload["window_size"] == 2
    assert payload["info_delta"] == {"score": 2.0}


def test_info_feeder_base_build_returns_text_and_empty_payload():
    feeder = retro_observation.InfoFeeder()
    text, payload = feeder.build(info_history=[{"tick": 0}], raw_info={"tick": 1}, token_budget=10)

    assert '"tick": 1' in text
    assert payload == {}


def test_observation_builder_build_populates_structured_fields():
    builder = retro_observation.ObservationBuilder(
        info_feeder=retro_observation.InfoLastFeeder(),
        action_schema=retro_observation.ActionSchema(hold_ticks_min=1, hold_ticks_max=3, default_hold_ticks=2),
        token_budget=50,
    )

    obs = builder.build(
        player_id="player_0",
        active_player="player_0",
        legal_moves=["noop", "right"],
        last_move=None,
        tick=1,
        decision_count=2,
        info_history=[],
        raw_info={"score": 7},
        reward_total=1.25,
        controls={
            "keys_hint": "Use keys",
            "buttons": ["LEFT", "RIGHT"],
            "move_aliases": {"right": {"keys_combo": "d", "buttons": ["RIGHT"]}},
        },
    )

    assert obs.view is not None
    assert obs.view.get("text") == obs.view_text
    assert obs.context == {"mode": "tick", "step": 2, "tick": 1}
    assert "Controls:" in (obs.view_text or "")
    assert isinstance(obs.extra.get("info_projection"), dict)
    assert isinstance(obs.extra.get("controls"), dict)


def test_format_controls_block_skips_invalid_alias_payloads_and_handles_empty_controls():
    assert retro_observation.ObservationBuilder._format_controls_block(None) == ""  # noqa: SLF001

    block = retro_observation.ObservationBuilder._format_controls_block(  # noqa: SLF001
        {"move_aliases": {"bad": "nope", "ok": {"keys_combo": "d"}}}
    )
    assert "ok=d" in block
    assert "bad=" not in block
