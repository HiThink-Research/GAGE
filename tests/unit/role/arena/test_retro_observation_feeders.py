from gage_eval.game_kits.real_time_game.retro_platformer import observation as retro_observation


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
    current = {"score": 2, "lives": 3, "name": "mario", "new_key": 9}

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


def test_info_none_feeder_disables_info_projection():
    feeder = retro_observation.InfoNoneFeeder()
    text, payload = feeder.build(info_history=[{"tick": 0}], raw_info={"tick": 1}, token_budget=10)

    assert text == ""
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
    assert obs.metadata.get("game_type") == "retro"
    assert obs.prompt is not None
    assert obs.prompt.payload.get("mode") == "tick"
    assert obs.prompt.payload.get("scheduler_mode") == "tick"
    schema_config = obs.metadata.get("action_schema_config") or {}
    assert schema_config.get("hold_ticks_min") == 1
    assert schema_config.get("hold_ticks_max") == 3
    assert schema_config.get("hold_ticks_default") == 2
    assert "Controls:" in (obs.view_text or "")
    extra = obs.metadata.get("observation_extra") or {}
    assert isinstance(extra.get("info_projection"), dict)
    assert isinstance(extra.get("action_schema_config"), dict)
    assert isinstance(extra.get("controls"), dict)


def test_observation_builder_omits_info_section_when_info_text_is_empty():
    builder = retro_observation.ObservationBuilder(
        info_feeder=retro_observation.InfoNoneFeeder(),
        action_schema=retro_observation.ActionSchema(),
        token_budget=32,
    )

    obs = builder.build(
        player_id="player_0",
        active_player="player_0",
        legal_moves=["noop"],
        last_move=None,
        tick=2,
        decision_count=3,
        info_history=[],
        raw_info={"score": 99},
        reward_total=4.0,
    )

    view_text = obs.view_text or ""
    assert "Reward total: 4.000" in view_text
    assert "Info:" not in view_text


def test_format_controls_block_skips_invalid_alias_payloads_and_handles_empty_controls():
    assert retro_observation.ObservationBuilder._format_controls_block(None) == ""  # noqa: SLF001

    block = retro_observation.ObservationBuilder._format_controls_block(  # noqa: SLF001
        {"move_aliases": {"bad": "nope", "ok": {"keys_combo": "d"}}}
    )
    assert "ok=d" in block
    assert "bad=" not in block


def test_observation_builder_build_attaches_image_payload():
    builder = retro_observation.ObservationBuilder(
        info_feeder=retro_observation.InfoLastFeeder(),
        action_schema=retro_observation.ActionSchema(),
        token_budget=16,
    )

    obs = builder.build(
        player_id="player_0",
        active_player="player_0",
        legal_moves=["noop"],
        last_move=None,
        tick=0,
        decision_count=0,
        info_history=[],
        raw_info={},
        reward_total=0.0,
        image={"encoding": "raw_base64", "data": "abc", "shape": [1, 1, 3], "dtype": "uint8"},
    )

    assert isinstance(obs.view, dict)
    assert obs.view.get("image", {}).get("encoding") == "raw_base64"


def test_observation_builder_build_includes_env_id_when_provided():
    builder = retro_observation.ObservationBuilder(
        info_feeder=retro_observation.InfoLastFeeder(),
        action_schema=retro_observation.ActionSchema(),
        token_budget=16,
    )

    obs = builder.build(
        player_id="player_0",
        active_player="player_0",
        legal_moves=["noop"],
        last_move=None,
        tick=0,
        decision_count=0,
        info_history=[],
        raw_info={},
        reward_total=0.0,
        game_type="SuperMarioBros3-Nes-v0",
        env_id="SuperMarioBros3-Nes-v0",
    )

    assert obs.metadata.get("game_type") == "SuperMarioBros3-Nes-v0"
    assert obs.metadata.get("env_id") == "SuperMarioBros3-Nes-v0"
    assert obs.prompt is not None
    assert obs.prompt.payload.get("env_id") == "SuperMarioBros3-Nes-v0"
