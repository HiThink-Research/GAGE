from unittest.mock import MagicMock

from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderResult, PromptRenderer
from gage_eval.role.arena.players.llm_player import LLMPlayer
from gage_eval.role.arena.types import ArenaObservation


def test_llm_player_uses_pettingzoo_prompt():
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
        board_text="PettingZoo env: pettingzoo.atari.space_invaders_v2",
        legal_moves=["NOOP", "FIRE", "LEFT"],
        active_player="player_0",
        metadata={"env_id": "pettingzoo.atari.space_invaders_v2"},
        view={"text": "PettingZoo env: pettingzoo.atari.space_invaders_v2"},
        legal_actions={"items": ["NOOP", "FIRE", "LEFT"]},
    )

    prompt = player._format_observation(observation)

    assert "Legal moves:" in prompt
    assert "NOOP" in prompt
    assert "Output ONLY the action label or id" in prompt
    assert "Current Board:" not in prompt


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

    assert messages == [{"role": "system", "content": "arena-system"}]
    assert renderer.last_payload["player_id"] == "player_0"
    assert renderer.last_payload["arena_observation"]["active_player"] == "player_0"
    assert renderer.last_payload["arena_observation"]["legal_moves"] == ["A1", "A2"]


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
