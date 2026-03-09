import base64
from unittest.mock import MagicMock

from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderResult, PromptRenderer
from gage_eval.role.arena.players.llm_player import LLMPlayer
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, ArenaPromptSpec


def test_llm_player_uses_game_owned_prompt_instruction() -> None:
    sample = {"messages": [], "metadata": {}}
    parser = MagicMock()
    role_manager = MagicMock()
    player = LLMPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
    )

    observation = ArenaObservation(
        board_text="Board",
        legal_moves=["NOOP", "FIRE", "LEFT"],
        active_player="player_0",
        metadata={},
        view={"text": "Board"},
        legal_actions={"items": ["NOOP", "FIRE", "LEFT"]},
        prompt=ArenaPromptSpec(
            instruction="Game prompt instruction",
            payload={},
        ),
    )

    prompt = player._format_observation(observation)

    assert prompt == "Game prompt instruction"


def test_llm_player_falls_back_to_legacy_grid_prompt_without_game_prompt() -> None:
    sample = {"messages": [], "metadata": {}}
    parser = MagicMock()
    role_manager = MagicMock()
    player = LLMPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
    )

    observation = ArenaObservation(
        board_text="Board",
        legal_moves=["A1", "A2"],
        active_player="player_0",
        metadata={},
        view={"text": "Board"},
        legal_actions={"items": ["A1", "A2"]},
    )

    prompt = player._format_observation(observation)

    assert "Current Board:" in prompt
    assert "A1, A2" in prompt


def test_llm_player_prompt_payload_keeps_retro_schema_and_scheduler_mode() -> None:
    sample = {"messages": [{"role": "system", "content": "legacy"}], "metadata": {}}
    parser = MagicMock()
    role_manager = MagicMock()
    player = LLMPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
        scheduler_mode="tick",
    )
    observation = ArenaObservation(
        board_text="Retro board",
        legal_moves=["noop", "right_run_jump"],
        active_player="player_0",
        metadata={
            "game_type": "retro",
        },
        view={"text": "Retro board"},
        legal_actions={"items": ["noop", "right_run_jump"]},
        context={"mode": "turn", "step": 10, "tick": 200},
        prompt=ArenaPromptSpec(
            instruction="base",
            payload={
                "game_type": "SuperMarioBros3-Nes-v0",
                "env_id": "SuperMarioBros3-Nes-v0",
                "legal_moves": ["noop", "right_run_jump"],
                "action_schema": 'Output ONE JSON object: {"move":"<legal_move>","hold_ticks":<int>}',
                "action_schema_config": {
                    "hold_ticks_min": 1,
                    "hold_ticks_max": 12,
                    "hold_ticks_default": 6,
                },
                "hold_ticks": {"min": 1, "max": 12, "default": 6},
                "arena_observation": {
                    "metadata": {"source": "retro_game_builder"},
                },
            },
        ),
    )

    payload = player._build_prompt_payload(
        observation=observation,
        prompt_text="choose",
        retry_reason=None,
        last_output=None,
        legacy_messages=[{"role": "system", "content": "legacy"}],
        has_image=False,
    )

    assert payload["game_type"] == "SuperMarioBros3-Nes-v0"
    assert payload["env_id"] == "SuperMarioBros3-Nes-v0"
    assert payload["mode"] == "tick"
    assert payload["scheduler_mode"] == "tick"
    assert payload["observation_mode"] == "turn"
    assert payload["legal_moves"] == ["noop", "right_run_jump"]
    assert payload["hold_ticks"] == {"min": 1, "max": 12, "default": 6}
    assert payload["arena_observation"]["metadata"]["source"] == "retro_game_builder"


def test_llm_player_retro_prompt_with_renderer_avoids_schema_duplication() -> None:
    sample = {"messages": [], "metadata": {}}
    parser = MagicMock()
    role_manager = MagicMock()
    renderer = _CaptureRenderer()
    player = LLMPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
        prompt_renderer=renderer,
    )
    observation = ArenaObservation(
        board_text="Retro board",
        legal_moves=["noop", "right"],
        active_player="player_0",
        metadata={"game_type": "SuperMarioBros3-Nes-v0"},
        view={"text": "Retro board"},
        legal_actions={"items": ["noop", "right"]},
        context={"mode": "tick", "step": 3, "tick": 12},
        prompt=ArenaPromptSpec(
            instruction='Output ONE JSON object: {"move":"<legal_move>","hold_ticks":<int>}',
            renderer_instruction="Follow the system prompt for output format and policy.",
            payload={},
        ),
    )

    prompt = player._format_observation(observation)

    assert "Follow the system prompt for output format and policy." in prompt
    assert "Output ONE JSON object" not in prompt


class _CaptureRenderer(PromptRenderer):
    def __init__(self) -> None:
        self.last_payload = {}

    def render(self, context: PromptContext) -> PromptRenderResult:
        self.last_payload = context.payload
        return PromptRenderResult(messages=[{"role": "system", "content": "arena-system"}])


def _build_test_data_url(token: str) -> str:
    payload = base64.b64encode(token.encode("utf-8")).decode("ascii")
    return f"data:image/png;base64,{payload}"


def test_llm_player_turn_messages_use_prompt_renderer_payload() -> None:
    sample = {"messages": [{"role": "system", "content": "legacy"}], "metadata": {"game_type": "gomoku"}}
    parser = MagicMock()
    role_manager = MagicMock()
    renderer = _CaptureRenderer()
    player = LLMPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
        prompt_renderer=renderer,
    )
    observation = ArenaObservation(
        board_text="Board",
        legal_moves=["A1", "A2"],
        active_player="player_0",
        metadata={"game_type": "gomoku"},
        view={"text": "Board"},
        legal_actions={"items": ["A1", "A2"]},
    )

    messages = player._build_turn_messages(
        observation=observation,
        prompt_text="Choose one move.",
        image_fragment=None,
    )

    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "arena-system"}
    assert messages[1]["role"] == "user"
    assert "Choose one move." in messages[1]["content"][0]["text"]
    assert renderer.last_payload["player_id"] == "player_0"
    assert renderer.last_payload["arena_observation"]["active_player"] == "player_0"
    assert renderer.last_payload["arena_observation"]["legal_moves"] == ["A1", "A2"]
    assert renderer.last_payload["messages"] == [{"role": "system", "content": "legacy"}]
    assert len(renderer.last_payload["legacy_messages"]) == 2


def test_llm_player_s4_turn_messages_attach_ordered_image_history() -> None:
    sample = {"messages": [], "metadata": {"game_type": "vizdoom"}}
    parser = MagicMock()
    role_manager = MagicMock()
    renderer = _CaptureRenderer()
    player = LLMPlayer(
        name="p1",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
        prompt_renderer=renderer,
        scheme_id="S4_text_image_4f_action_hist",
        scheme_params={"action_history_len": 4, "image_history_len": 4},
    )
    image_history = [
        {"data_url": _build_test_data_url("frame_0")},
        {"data_url": _build_test_data_url("frame_1")},
        {"data_url": _build_test_data_url("frame_2")},
        {"data_url": _build_test_data_url("frame_3")},
    ]
    observation = ArenaObservation(
        board_text="",
        legal_moves=["1", "2", "3"],
        active_player="p1",
        metadata={"game_type": "vizdoom", "player_id": "p1", "player_ids": ["p0", "p1"]},
        view={
            "text": "Tick 10",
            "image": image_history[-1],
            "image_history": image_history,
        },
        legal_actions={"items": ["1", "2", "3"]},
    )

    prompt_text = player._format_observation(observation)
    image_fragments = player._build_image_fragments(observation)
    messages = player._build_turn_messages(
        observation=observation,
        prompt_text=prompt_text,
        image_fragments=image_fragments,
    )

    assert "Attached images:" in prompt_text
    user_content = messages[1]["content"]
    assert len([part for part in user_content if part["type"] == "image_url"]) == 4
    assert renderer.last_payload["image_count"] == 4
    assert renderer.last_payload["vizdoom_strategy"]["has_image_history_block"] is True
    assert renderer.last_payload["vizdoom_strategy"]["image_history_count"] == 4


def test_llm_player_dumps_one_directory_per_request(tmp_path) -> None:
    sample = {"messages": [], "metadata": {"game_type": "vizdoom"}}
    parser = MagicMock()
    role_manager = MagicMock()
    renderer = _CaptureRenderer()
    player = LLMPlayer(
        name="p1",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
        prompt_renderer=renderer,
        scheme_id="S4_text_image_4f_action_hist",
        scheme_params={
            "action_history_len": 4,
            "image_history_len": 4,
            "debug_image_dump_dir": str(tmp_path),
            "debug_image_dump_max": 10,
        },
    )
    image_history = [
        {"data_url": _build_test_data_url("frame_0")},
        {"data_url": _build_test_data_url("frame_1")},
        {"data_url": _build_test_data_url("frame_2")},
        {"data_url": _build_test_data_url("frame_3")},
    ]
    observation = ArenaObservation(
        board_text="",
        legal_moves=["1", "2", "3"],
        active_player="p1",
        metadata={"game_type": "vizdoom", "player_id": "p1", "player_ids": ["p0", "p1"]},
        view={
            "text": "Tick 10",
            "image": image_history[-1],
            "image_history": image_history,
        },
        legal_actions={"items": ["1", "2", "3"]},
    )

    prompt_text = player._format_observation(observation)
    image_fragments = player._build_image_fragments(observation)
    messages = player._build_turn_messages(
        observation=observation,
        prompt_text=prompt_text,
        image_fragments=image_fragments,
    )
    player._maybe_dump_request_images(
        observation=observation,
        messages=messages,
        phase="primary",
    )

    request_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(request_dirs) == 1
    frame_files = sorted(request_dirs[0].glob("frame_*"))
    assert len(frame_files) == 4


def test_llm_player_turn_messages_avoid_duplicate_image_fragment() -> None:
    sample = {"messages": [], "metadata": {"game_type": "gomoku"}}
    parser = MagicMock()
    role_manager = MagicMock()
    renderer = _CaptureRenderer()
    player = LLMPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
        prompt_renderer=renderer,
    )
    image_fragment = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}
    rendered_messages = [
        {"role": "system", "content": "arena-system"},
        {"role": "user", "content": [{"type": "text", "text": "Choose one move."}, image_fragment]},
    ]

    messages = player._append_image_fragment(  # noqa: SLF001
        rendered_messages,
        image_fragment=image_fragment,
        fallback_text="Choose one move.",
    )

    user_content = messages[1]["content"]
    image_count = sum(1 for part in user_content if isinstance(part, dict) and part.get("type") == "image_url")
    assert image_count == 1


def test_llm_player_turn_messages_fallback_to_backward_prompt() -> None:
    sample = {"messages": [{"role": "system", "content": "legacy"}], "metadata": {}}
    parser = MagicMock()
    role_manager = MagicMock()
    player = LLMPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
    )
    observation = ArenaObservation(
        board_text="Board",
        legal_moves=["A1", "A2"],
        active_player="player_0",
        metadata={},
        view={"text": "Board"},
        legal_actions={"items": ["A1", "A2"]},
    )

    messages = player._build_turn_messages(
        observation=observation,
        prompt_text="Choose one move.",
        image_fragment=None,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Choose one move." in messages[1]["content"][0]["text"]
    assert "missing_prompt_renderer" in player._backward_prompt_logged_reasons


def test_summarize_messages_for_log_includes_data_url_image_reference() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "frame attached"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
            ],
        }
    ]

    summary = LLMPlayer._summarize_messages_for_log(messages)  # noqa: SLF001

    assert "frame attached" in summary
    assert "<image_ref:data:image/jpeg;base64;sha1=" in summary
    assert "chars=" in summary


def test_summarize_messages_for_log_includes_http_image_reference() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "remote image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/sample.jpg"}},
            ],
        }
    ]

    summary = LLMPlayer._summarize_messages_for_log(messages)  # noqa: SLF001

    assert "remote image" in summary
    assert "<image_ref:https://example.com/sample.jpg>" in summary


def test_llm_player_scheme_prompt_overrides_game_owned_instruction_for_vizdoom() -> None:
    player = LLMPlayer(
        name="p0",
        adapter_id="dummy",
        role_manager=MagicMock(),
        sample={"messages": [], "metadata": {"game_type": "vizdoom"}},
        parser=MagicMock(),
        scheme_id="S3_text_image_current",
    )
    observation = _build_vizdoom_observation(step=3, health=100, reward=0.0)
    observation.prompt = ArenaPromptSpec(
        instruction="Game-owned instruction should not dominate scheme prompt.",
        payload={},
    )

    prompt = player._format_observation(observation)

    assert "Perspective:" in prompt
    assert "Reason: <short reason>" in prompt
    assert "Game-owned instruction should not dominate scheme prompt." not in prompt


def test_llm_player_scheme_s1_suppresses_image_payload() -> None:
    player = LLMPlayer(
        name="p0",
        adapter_id="dummy",
        role_manager=MagicMock(),
        sample={"messages": [], "metadata": {"game_type": "vizdoom"}},
        parser=MagicMock(),
        scheme_id="S1_rich_text_only",
    )

    image_fragment = player._build_image_fragment(_build_vizdoom_observation(step=1, health=100, reward=0.0))

    assert image_fragment is None


def test_llm_player_scheme_s6_v2_includes_action_outcome_history() -> None:
    player = LLMPlayer(
        name="p0",
        adapter_id="dummy",
        role_manager=MagicMock(),
        sample={"messages": [], "metadata": {"game_type": "vizdoom"}},
        parser=MagicMock(),
        scheme_id="S6_v2_text_image_action_outcome_hist",
    )
    first_observation = _build_vizdoom_observation(step=4, health=100, reward=0.0)
    second_observation = _build_vizdoom_observation(step=5, health=95, reward=0.5)
    player._remember_interaction(
        first_observation,
        ArenaAction(player="p0", move="2", raw="2", metadata={"player_type": "backend"}),
    )
    player._remember_interaction(
        second_observation,
        ArenaAction(player="p0", move="1", raw="1", metadata={"player_type": "backend"}),
    )

    prompt = player._format_observation(_build_vizdoom_observation(step=6, health=94, reward=1.0))

    assert "Recent action and outcome history (oldest to newest):" in prompt
    assert "step=4: action=2 -> outcome@step=5: HEALTH:-5" in prompt
    assert "Action: <action_id>" in prompt
    assert "Reason: <short reason>" in prompt


def test_llm_player_scheme_prompt_in_renderer_mode_uses_observation_snapshot() -> None:
    player = LLMPlayer(
        name="p0",
        adapter_id="dummy",
        role_manager=MagicMock(),
        sample={"messages": [], "metadata": {"game_type": "vizdoom"}},
        parser=MagicMock(),
        prompt_renderer=MagicMock(),
        scheme_id="S3_text_image_current",
    )
    prompt = player._format_observation(_build_vizdoom_observation(step=6, health=94, reward=1.0))

    assert "Observation snapshot for the current ViZDoom turn." in prompt
    assert "Current state:" in prompt
    assert "Instructions:" not in prompt


def test_llm_player_vizdoom_prompt_payload_contains_strategy_blocks() -> None:
    player = LLMPlayer(
        name="p0",
        adapter_id="dummy",
        role_manager=MagicMock(),
        sample={"messages": [], "metadata": {"game_type": "vizdoom"}},
        parser=MagicMock(),
        scheme_id="S6_v2_text_image_action_outcome_hist",
    )
    first_observation = _build_vizdoom_observation(step=4, health=100, reward=0.0)
    second_observation = _build_vizdoom_observation(step=5, health=95, reward=0.5)
    player._remember_interaction(
        first_observation,
        ArenaAction(player="p0", move="2", raw="2", metadata={"player_type": "backend"}),
    )
    player._remember_interaction(
        second_observation,
        ArenaAction(player="p0", move="1", raw="1", metadata={"player_type": "backend"}),
    )
    payload = player._build_prompt_payload(
        observation=_build_vizdoom_observation(step=6, health=94, reward=1.0),
        prompt_text="placeholder",
        retry_reason=None,
        last_output=None,
        legacy_messages=[],
        has_image=True,
    )

    assert payload["scheme_id"] == "S6_v2_text_image_action_outcome_hist"
    assert payload["scheme_supports_image"] is True
    strategy = payload["vizdoom_strategy"]
    assert strategy["scheme_id"] == "S6_v2_text_image_action_outcome_hist"
    assert "Perspective:" not in strategy["perspective_block"]
    assert strategy["has_action_outcome_history_block"] is True
    assert "step=4: action=2 -> outcome@step=5: HEALTH:-5" in strategy["action_outcome_history_block"]


def test_build_action_metadata_contains_decision_reason() -> None:
    player = LLMPlayer(
        name="p0",
        adapter_id="dummy",
        role_manager=MagicMock(),
        sample={"messages": [], "metadata": {"game_type": "vizdoom"}},
        parser=MagicMock(),
        scheme_id="S3_text_image_current",
    )
    parse_result = type("ParseResult", (), {"reason": "move left then fire", "chat_text": None, "hold_ticks": None})()

    metadata = player._build_action_metadata(parse_result, retry_count=1)

    assert metadata["scheme_id"] == "S3_text_image_current"
    assert metadata["decision_reason"] == "move left then fire"
    assert metadata["retry_count"] == 1


def _build_image_payload() -> dict[str, str | list[int]]:
    raw = base64.b64encode(bytes([0, 0, 0])).decode("ascii")
    return {
        "encoding": "raw_base64",
        "data": raw,
        "shape": [1, 1, 3],
        "dtype": "uint8",
    }


def _build_vizdoom_observation(*, step: int, health: int, reward: float) -> ArenaObservation:
    return ArenaObservation(
        board_text="",
        legal_moves=["1", "2", "3"],
        active_player="p0",
        metadata={
            "game_type": "vizdoom",
            "reward": reward,
            "t": step,
        },
        view={
            "text": f"Tick {step}. Legal actions: 1, 2, 3",
            "vector": {"HEALTH": health, "FRAGCOUNT": 1},
            "image": _build_image_payload(),
        },
        legal_actions={"items": ["1", "2", "3"]},
        context={"step": step, "mode": "tick"},
    )
