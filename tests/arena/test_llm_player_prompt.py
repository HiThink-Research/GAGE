from unittest.mock import MagicMock

from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderResult, PromptRenderer
from gage_eval.role.arena.players.llm_player import LLMPlayer
from gage_eval.role.arena.types import ArenaObservation, ArenaPromptSpec


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
